"""Shadow mode: the detector watches a match the farm loop is already playing.

``PlaySession`` is driven once per controller tick, the way ``MatchRecorder`` is. A daemon
thread behind it starts the frame source, loads the detector, reads the newest frame at
``RATE_HZ`` at most and writes one JSON line per inference under the instance's data folder. It
sends no input, it never touches the datalog, and it never raises into the loop: the thread hands
its events to a list under a lock and the controller thread drains them from ``observe()``.

Each row also carries what ``brawlfarm.play.reader`` read from that frame and what
``brawlfarm.play.rules`` would call on it. Those fields only ever add to a row: ``ms`` is still
the detector alone, and a reader or rules error costs the fields of that row, never the session.

The controller never waits for that thread either. The end of a match sets the stop flag and
returns; the thread stops the source, closes the file and queues its summary on its way out, and
the controller drains that row on a later tick. Only ``close()``, at process shutdown, joins.

The source is the emulator's own screen captures (``brawlfarm.play.capture``), not a live video
stream: one capture costs 180 to 280 ms, so a frame is 0.2 to 0.3 s old by the time the model
sees it.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from brawlfarm.core.states import State
from brawlfarm.play import matchrec, rules

log = logging.getLogger("brawlfarm.play.session")

# RATE_HZ rarely binds: a capture costs 180 to 280 ms, so the real rate is about 5 a second
RATE_HZ = 5.0  # most inferences per second; a farm tick is about 0.75 s
STALE_LIMIT = 5.0  # seconds without a fresh frame before the session gives up
MAX_SECONDS = 360.0  # as matchrec: a backstop, not the rule
KEEP_FILES = 50  # newest shadow files kept per instance
JOIN_TIMEOUT = 3.0  # seconds close() waits for the thread; no other caller waits at all

Event = tuple[str, dict]


def _plain(value: Any) -> Any:
    """A reader value as JSON takes it: a numpy scalar would stop the row at json.dumps."""
    if value is None:
        return None
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, float | np.floating):
        return round(float(value), 3)
    return int(value)


def _default_source() -> Any:
    from brawlfarm.play import capture  # kept lazy so the module stays cheap to import

    return capture.ScreencapSource()


def _default_detector() -> Any:
    from brawlfarm.play import detect  # onnxruntime is heavy and shadow mode alone pays

    return detect.Detector.load()


def _default_folder() -> Path:
    from brawlfarm.core import config

    return config.DATA_DIR / "shadow"


class PlaySession:
    """One shadow session per match. Watches, writes, and sends nothing.

    ``observe()`` and ``close()`` belong to the controller thread; ``_begin()`` and ``_tick()``
    run on the session thread (or are called by a test with ``threaded=False``). The event
    queue, the failure reason and the counters the summary reads are the only shared state and
    one lock guards all of them.

    Frames come from the emulator's screen captures, so a frame is 0.2 to 0.3 s old by the time
    the model sees it.

    Shadow mode needs no optional extra. It captures the screen through adb and runs the model on
    onnxruntime, which rapidocr-onnxruntime already requires, so a plain install can run it.
    """

    def __init__(
        self,
        *,
        folder: Path | None = None,
        source_factory: Callable[[], Any] | None = None,
        detector_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        threaded: bool = True,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._folder = Path(folder) if folder is not None else None
        self._source_factory = source_factory or _default_source
        self._detector_factory = detector_factory or _default_detector
        self._clock = clock
        self._sleep = sleep
        self._threaded = threaded
        self._now = now

        self._lock = threading.Lock()
        self._events: list[Event] = []
        self._failed: str | None = None
        self._frames = 0
        self._ms: list[float] = []
        self._stale_ticks = 0
        self._counts: dict[str, int] = {}
        self._rules_ms: list[float] = []
        self._read_hits: dict[str, int] = {}
        self._rule_calls: dict[str, int] = {}
        self._rules_errors = 0
        self._farm: tuple[str, str, float] | None = None  # state, phase, clock at the last tick
        self._enemy_frames: deque[bool] = deque(maxlen=rules.ENEMY_WINDOW)
        self._file: Any = None
        self._file_name = ""

        self._running = False
        self._closed = False
        self._off = False  # latched for the life of the process: no usable model
        self._await_stop = False  # a failed session waits for the match to be over
        self._torn_down = True  # no session in flight, so there is nothing to tear down
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._source: Any = None
        self._detector: Any = None  # loaded once and kept across matches
        self._started_at = 0.0
        self._last_fresh = 0.0

    @property
    def active(self) -> bool:
        return self._running

    # -- the controller thread ---------------------------------------------------

    def observe(self, state: State, phase: str) -> list[Event]:
        """One controller tick. Returns the feed rows the controller writes."""
        try:
            self._observe(state, phase)
        except Exception as exc:  # shadow mode must never stop the farm loop
            log.warning("shadow session error: %s", exc)
            self._await_stop = True
            self._running = False
            self._queue("play_fallback", reason="session_error", detail=repr(exc))
        return self._drain()

    def close(self) -> list[Event]:
        """End a live session and drain. Closing twice is a no-op returning [].

        Shutdown is the one place that may wait: the controller calls this from its finally,
        where there is no tick left to hold up.
        """
        if self._closed:
            return []
        self._closed = True
        try:
            if self._running:
                self._end()
            self._finish()
        except Exception as exc:
            log.warning("shadow session did not close cleanly: %s", exc)
            self._queue("play_fallback", reason="session_error", detail=repr(exc))
        return self._drain()

    def _observe(self, state: State, phase: str) -> None:
        with self._lock:
            self._farm = (state.value, phase, self._clock())
        playing = phase == "playing"
        match_over = not playing or state in matchrec.STOP_STATES
        if self._running:
            if (
                match_over
                or self._clock() - self._started_at >= MAX_SECONDS
                or self._reason() is not None
            ):
                self._end()
            if match_over:
                self._await_stop = False  # this tick is the end of the match, not a wait for it
            return
        if match_over:
            self._await_stop = False  # the match this session gave up on is over
        if self._closed or self._off or self._await_stop:
            return
        if not playing or state != State.IN_MATCH:
            return
        if self._winding_down():
            return  # the last thread is still ending; this tick does not wait for it
        self._start()

    def _start(self) -> None:
        with self._lock:
            self._failed = None
            self._frames = 0
            self._ms = []
            self._stale_ticks = 0
            self._counts = {}
            self._rules_ms = []
            self._read_hits = {}
            self._rule_calls = {}
            self._rules_errors = 0
            self._torn_down = False
        self._enemy_frames.clear()
        self._started_at = self._clock()
        self._last_fresh = self._started_at
        self._stop_flag.clear()
        self._running = True
        if self._threaded:
            self._thread = threading.Thread(target=self._run, name="play-shadow", daemon=True)
            self._thread.start()

    def _end(self) -> None:
        """Signal the session and leave. The thread tears its own session down.

        A join here would hold a controller tick for seconds, twice over: the thread may be
        inside a capture, and stopping the source joins its pump thread as well. So the
        controller only sets the flag and marks the session done. Whichever thread owns the
        session stops the source, closes the file and queues the summary, and the controller
        drains that row on a later tick.
        """
        self._stop_flag.set()
        self._running = False
        self._await_stop = True
        if self._winding_down():
            return
        self._teardown()

    def _finish(self) -> None:
        """Shutdown only: wait a bounded time for the thread, then tear down what is left.

        ``close()`` is the one caller, and the controller calls it from its own finally, with
        no tick left to hold up. Nothing inside ``observe()`` reaches this.
        """
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(JOIN_TIMEOUT)
            if thread.is_alive():  # the source is stopped under it on purpose
                log.warning("the shadow thread is still running after %.0f s", JOIN_TIMEOUT)
        self._teardown()

    def _teardown(self) -> None:
        """Stop the source, close the file, queue the summary. Once per session.

        It runs on whichever thread gets there first: the session thread on its way out, or
        the controller when there is no thread to do it (``close()``, or ``threaded=False``).
        Every row is flushed as it is written, so a file this never closes is still readable.
        """
        with self._lock:
            if self._torn_down:
                return
            self._torn_down = True
        self._stop_source()
        self._close_file()
        with self._lock:
            frames, ms, stale = self._frames, list(self._ms), self._stale_ticks
            counts, name, failed = dict(self._counts), self._file_name, self._failed
            rules_ms, hits = list(self._rules_ms), dict(self._read_hits)
            rule_calls, rules_errors = dict(self._rule_calls), self._rules_errors
        if frames == 0:
            # nothing was ever inferred: a failed start says so in its fallback already
            return
        seconds = max(self._clock() - self._started_at, 0.0)
        self._queue(
            "play_summary",
            frames=frames,
            seconds=round(seconds, 1),
            fps=round(frames / seconds, 2) if seconds > 0 else 0.0,
            ms_p50=round(float(np.percentile(ms, 50)), 1) if ms else 0.0,
            ms_p95=round(float(np.percentile(ms, 95)), 1) if ms else 0.0,
            stale_ticks=stale,
            boxes=counts,
            file=name,
            rules_ms_p50=round(float(np.percentile(rules_ms, 50)), 1) if rules_ms else 0.0,
            rules_ms_p95=round(float(np.percentile(rules_ms, 95)), 1) if rules_ms else 0.0,
            coverage={field: round(hit / frames, 3) for field, hit in hits.items()},
            rule_calls=rule_calls,
            rules_errors=rules_errors,
        )
        if failed is not None:
            log.info("shadow session ended early (%s) after %d frames", failed, frames)

    # -- the session thread ------------------------------------------------------

    def _run(self) -> None:
        try:
            self._begin()
            while not self._stop_flag.is_set():
                due = self._clock() + 1.0 / RATE_HZ
                if not self._tick() or self._stop_flag.is_set():
                    return
                nap = due - self._clock()
                if nap > 0:
                    self._sleep(nap)
        except Exception as exc:  # the thread dies quietly, the loop never hears about it
            log.warning("the shadow thread stopped: %s", exc)
            self._fail("session_error", repr(exc))
        finally:
            try:  # the thread owns the teardown: the controller signalled and left
                self._teardown()
            except Exception as exc:
                log.warning("the shadow thread did not end cleanly: %s", exc)

    def _begin(self) -> None:
        """Detector, source, file, header: what the thread does before its first frame.

        The detector comes first so that a worker with no model in its models folder never
        starts a source it is about to throw away.
        """
        from brawlfarm.play import detect

        began = self._clock()
        if self._detector is None:
            try:
                self._detector = self._detector_factory()
            except detect.ModelMissing as exc:
                log.info("shadow mode off: %s", exc)
                self._disable("model_missing")
                return
            except detect.ModelInvalid as exc:
                log.warning("shadow mode off: %s", exc)
                self._disable("model_invalid", repr(exc))
                return
            except Exception as exc:
                self._fail("detector_error", repr(exc))
                return
        try:
            source = self._source_factory()
            source.start()
        except Exception as exc:
            self._fail("source_start", repr(exc))
            return
        self._source = source
        if self._stop_flag.is_set():
            # the match ended while the source was starting: hand the source back, say nothing
            self._stop_source()
            return
        try:
            self._open_file()
        except OSError as exc:
            self._fail("session_error", repr(exc))
            return
        self._last_fresh = self._clock()
        self._queue(
            "play_on",
            model=self._detector.name,
            provider=self._detector.provider,
            smoke=bool(self._detector.smoke),
            start_ms=round((self._clock() - began) * 1000.0, 1),
        )

    def _tick(self) -> bool:
        """One frame through the model. False when the session is over."""
        source = self._source  # a local: the controller may stop and drop it mid-tick
        if source is None or self._stop_flag.is_set():
            return False
        error = getattr(source, "error", None)
        if error is not None:
            self._fail("source_error", str(error))
            return False
        frame, age = source.latest()
        if frame is None or age is None:
            with self._lock:
                self._stale_ticks += 1
            if self._clock() - self._last_fresh >= STALE_LIMIT:
                self._fail("stale")
                return False
            return True
        self._last_fresh = self._clock()
        at = self._clock()
        try:
            boxes = self._detector.detect(frame)
        except Exception as exc:
            self._fail("detector_error", repr(exc))
            return False
        ms = (self._clock() - at) * 1000.0
        row = {
            "t": round(self._clock() - self._started_at, 3),
            "age": round(age, 3),
            "ms": round(ms, 1),
            # boxes, timings and HUD numbers only: no pixels, no names, no tags
            "boxes": [
                [box.cls, round(box.score, 3), int(box.x), int(box.y), int(box.w), int(box.h)]
                for box in boxes
            ],
        }
        row.update(self._judge(frame, boxes))
        self._write(row)
        with self._lock:
            self._frames += 1
            self._ms.append(ms)
            for box in boxes:
                self._counts[box.cls] = self._counts.get(box.cls, 0) + 1
        return True

    def _judge(self, frame: np.ndarray, boxes: list[Any]) -> dict:
        """The reader and the rules on this tick's frame: the row's added fields.

        Everything that can raise stays inside the try, the turn into plain values included,
        because an error here must cost this row's fields and not reach ``_run``, which would
        end the session. ``read`` and ``rules`` are None for a row they did not finish. Only
        the first error of a session is logged: at 5 Hz the rest would bury the log.
        """
        at = self._clock()
        read: dict[str, Any] | None = None
        calls: dict[str, Any] | None = None
        failed = None
        try:
            from brawlfarm.play import reader  # cv2 is heavy and shadow mode alone pays

            reading = reader.read(frame, boxes)
            read = {field: _plain(value) for field, value in asdict(reading).items()}
            self._enemy_frames.append(any(box.cls == "enemy" for box in boxes))
            context = rules.Context(enemy_recent=sum(self._enemy_frames))
            calls = asdict(rules.evaluate(reading, boxes, context))
        except Exception as exc:
            failed = exc
        ms = (self._clock() - at) * 1000.0
        with self._lock:
            self._rules_ms.append(ms)
            for field, value in (read or {}).items():
                self._read_hits[field] = self._read_hits.get(field, 0) + (value not in (None, []))
            for name, call in (calls or {}).items():
                self._rule_calls[name] = self._rule_calls.get(name, 0) + (call is not None)
            first = failed is not None and self._rules_errors == 0
            self._rules_errors += failed is not None
            farm = self._farm
        if first:
            log.warning("shadow reader or rules error, rows go on without them: %r", failed)
        added: dict[str, Any] = {
            "rules_ms": round(ms, 1),
            "read": read,
            "rules": calls,
            "farm": None,
        }
        if farm is not None:
            # the label the farm's last tick gave; what it then tapped never reaches this session
            state, phase, seen = farm
            added["farm"] = {"state": state, "phase": phase, "age": round(self._clock() - seen, 3)}
        return added

    # -- the file ----------------------------------------------------------------

    def _open_file(self) -> None:
        """Prune to the newest KEEP_FILES - 1, then open this session's file and head it."""
        folder = self._folder if self._folder is not None else _default_folder()
        folder.mkdir(parents=True, exist_ok=True)
        old = sorted(folder.glob("*.jsonl"), key=lambda path: (path.stat().st_mtime, path.name))
        for path in old[: max(len(old) - (KEEP_FILES - 1), 0)]:
            try:
                path.unlink()
            except OSError as exc:
                log.debug("old shadow file not removed: %s", exc)
        started = self._now()
        stem = started.strftime("%Y%m%d-%H%M%S")
        path = folder / f"{stem}.jsonl"
        suffix = 2
        while path.exists():  # two sessions in the same second
            path = folder / f"{stem}-{suffix}.jsonl"
            suffix += 1
        handle = path.open("w", encoding="utf-8")
        with self._lock:
            self._file = handle
            self._file_name = path.name
        self._write(
            {
                "model": self._detector.name,
                "training_set_hash": self._detector.training_set_hash,
                "provider": self._detector.provider,
                "smoke": bool(self._detector.smoke),
                "classes": list(self._detector.classes),
                "rate_hz": RATE_HZ,
                "started": started.isoformat(timespec="seconds"),
            }
        )

    def _write(self, row: dict) -> None:
        """One line, flushed: a worker that is killed mid-match still leaves a readable file."""
        with self._lock:
            handle = self._file
        if handle is None:  # closed under the thread by an ending controller tick
            return
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        handle.flush()

    def _close_file(self) -> None:
        with self._lock:
            handle, self._file = self._file, None
        if handle is None:
            return
        try:
            handle.close()
        except OSError as exc:
            log.debug("shadow file not closed cleanly: %s", exc)

    # -- shared bits -------------------------------------------------------------

    def _winding_down(self) -> bool:
        """True while a session thread is still ending. Reaps a finished one, never waits."""
        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return False
        if thread.is_alive():
            return True
        self._thread = None
        return False

    def _stop_source(self) -> None:
        source, self._source = self._source, None
        if source is None:
            return
        try:
            source.stop()
        except Exception as exc:
            log.warning("the play source did not stop cleanly: %s", exc)

    def _fail(self, reason: str, detail: str | None = None) -> None:
        """Give up on this session: one fallback row, then wait for the match to end."""
        with self._lock:
            if self._failed is not None or self._stop_flag.is_set():
                return  # already failing, or the controller is ending the session anyway
            self._failed = reason
        log.warning("shadow session gave up: %s%s", reason, f" ({detail})" if detail else "")
        self._stop_flag.set()
        self._stop_source()
        self._close_file()
        fields = {"reason": reason}
        if detail is not None:
            fields["detail"] = detail
        self._queue("play_fallback", **fields)

    def _disable(self, reason: str, detail: str | None = None) -> None:
        """A failure no other match in this process can recover from."""
        self._off = True
        self._fail(reason, detail)

    def _reason(self) -> str | None:
        with self._lock:
            return self._failed

    def _queue(self, kind: str, **fields: Any) -> None:
        with self._lock:
            self._events.append((kind, {"shadow": True, **fields}))

    def _drain(self) -> list[Event]:
        with self._lock:
            events, self._events = self._events, []
        return events
