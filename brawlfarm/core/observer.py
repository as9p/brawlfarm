"""Observe-only recording: the owner plays by hand while this process watches.

The loop screencaps, writes the panel's preview frame, classifies the frame and hands it
to the same Recorder the farm uses, so the next recalibration corpus comes from real play
instead of from the bot's own narrow path through the game. It records at the native
1600x900 size, so a template crop can be cut straight out of a session folder.

It imports adb for connect, screencap and screen_size and nothing that touches the screen:
controller, brawlers, quests, rewards and core/settings are all outside its import graph,
so no tap function is reachable from here and tests/test_observer.py asserts exactly that.

It runs as ``python -m brawlfarm.worker --observe``, which puts it in the one worker slot
per instance the supervisor already guards, so an observer and a farming worker can never
drive one display at the same time. Observe mode is recording by definition, so it raises
record.flag itself and drops it on the way out, including when the loop raises: one switch
for the owner, not two.
"""

from __future__ import annotations

import logging
import os
import time

from brawlfarm.core import adb, config, preview, states, status
from brawlfarm.core.recorder import Recorder
from brawlfarm.play.matchrec import MatchRecorder

log = logging.getLogger("brawlfarm.core.observer")

POLL_INTERVAL_S = 0.5  # about two frames a second; the recorder keeps at most one
STATUS_EVERY = 10  # heartbeat and flag poll roughly every five seconds
ADB_ERROR_LIMIT = 8  # the same streak the farm loop tolerates before it gives up

RECORD_FLAG = "record.flag"
STOP_FLAG = "stop.flag"
# Recorder reasons that mean the session is over for good. Nothing is left to record
# after one of these, so the loop stops instead of spinning on a closed session.
_SPENT = ("frame_cap", "disk_cap", "error")


class Observer:
    """Watch one instance and record labeled frames until stop.flag or a cap."""

    def __init__(self, *, max_minutes: float | None = None) -> None:
        self.max_minutes = max_minutes
        self.frames = 0
        self.start = time.monotonic()
        self.recorder = Recorder(
            config.HOME_DIR / "calibration",
            config.DATA_DIR.name,
            config.DATA_DIR / RECORD_FLAG,
            full_size=True,
        )
        self.matches = MatchRecorder()

    # --- the flag the owner never has to touch ------------------------------------

    def _set_record_flag(self) -> None:
        flag = config.DATA_DIR / RECORD_FLAG
        try:
            flag.parent.mkdir(parents=True, exist_ok=True)
            flag.touch()
        except OSError as exc:
            log.warning("record.flag not raised: %s", getattr(exc, "strerror", None) or exc)

    def _clear_record_flag(self) -> None:
        try:
            (config.DATA_DIR / RECORD_FLAG).unlink(missing_ok=True)
        except OSError:
            pass

    # --- the heartbeat ------------------------------------------------------------

    def _write_status(self, *, running: bool = True) -> None:
        """The same status.json slot a farming worker writes, so the supervisor sees one
        process per instance whichever kind it is. ``mode`` is what tells them apart."""
        status.write_status(
            config.DATA_DIR,
            {
                "pid": os.getpid(),
                "running": running,
                "mode": "observe",
                "phase": "observing",
                "frames": self.frames,
                "minutes_elapsed": round((time.monotonic() - self.start) / 60, 1),
                "max_minutes": self.max_minutes,
                "port": config.ADB_PORT,
            },
        )

    # --- the loop -----------------------------------------------------------------

    def _done(self) -> bool:
        try:
            if (config.DATA_DIR / STOP_FLAG).exists():
                return True
        except OSError:
            pass
        if self.max_minutes is not None:
            if (time.monotonic() - self.start) / 60 >= self.max_minutes:
                return True
        return self.recorder.status()["reason"] in _SPENT

    def run(self) -> None:
        adb.connect()
        wrong = adb.emulator_mismatch()
        if wrong:
            raise SystemExit(wrong)
        w, h = adb.screen_size()
        if (w, h) != (config.SCREEN_W, config.SCREEN_H):
            raise SystemExit(
                f"Display is {w}x{h}, expected {config.SCREEN_W}x{config.SCREEN_H}. "
                f"Lock the emulator's resolution before recording."
            )
        # A leftover stop.flag belongs to the worker that stopped before this one, and
        # whoever launched us wants us running.
        try:
            (config.DATA_DIR / STOP_FLAG).unlink(missing_ok=True)
        except OSError:
            pass
        self._set_record_flag()
        self.recorder.poll()
        self._write_status()
        log.info("observing; turn the switch off to end the session")
        errors = 0
        i = 0
        try:
            while True:
                try:
                    screen = adb.screencap()
                    errors = 0
                except adb.AdbError as exc:
                    errors += 1
                    log.warning("adb error (streak %d): %s", errors, exc)
                    if errors >= ADB_ERROR_LIMIT:
                        raise
                    time.sleep(POLL_INTERVAL_S)
                    continue
                preview.maybe_write(screen)
                # No farm phase to hint with, and PHASE_ORDER only reorders anchors: it
                # never changes the label a frame gets.
                state = states.classify(screen, phase=None)
                self.matches.observe(state, self.recorder.session_dir)
                if self.recorder.observe(screen, state, "observe"):
                    self.frames += 1
                i += 1
                if i % STATUS_EVERY == 0:
                    self._write_status()
                    self.recorder.poll()
                if self._done():
                    break
                time.sleep(POLL_INTERVAL_S)
        finally:
            self._clear_record_flag()
            self.matches.close()
            self.recorder.close()
            self._write_status(running=False)
            log.info("observe session ended after %d frames", self.frames)
