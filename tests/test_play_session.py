"""The shadow session: a detector runs beside an untouched farm loop and sends nothing.

Everything here drives the session by hand (``threaded=False``) except the one threaded test,
so the reads are deterministic: the fake clock only moves when a test moves it.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from brawlfarm import play
from brawlfarm.core.states import State
from brawlfarm.play import capture, detect, reader, rules, session

HUNG_LIMIT = 10.0  # seconds a hung fake capture waits before it gives up on the test
TICK_BOUND = 1.0  # seconds: what a controller tick may cost, well under one JOIN_TIMEOUT

HEADER_KEYS = {
    "model",
    "training_set_hash",
    "provider",
    "smoke",
    "classes",
    "rate_hz",
    "started",
}


class FakeSource:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.started = False
        self.stopped = False
        self.error: str | None = None
        self.frame: np.ndarray | None = np.zeros((900, 1600, 3), dtype=np.uint8)
        self.age = 0.25

    def start(self) -> None:
        if self.fail:
            raise RuntimeError("no device")
        self.started = True

    def latest(self):
        if self.frame is None or self.error is not None:
            return None, None
        return self.frame, self.age

    def stop(self) -> None:
        self.stopped = True


class FakeDetector:
    name = "play-v1"
    provider = "CPUExecutionProvider"
    smoke = False
    training_set_hash = "abc123"
    classes = ("enemy", "bush")

    def __init__(self) -> None:
        self.fail = False
        self.boxes = [
            detect.Box(cls="enemy", score=0.9123, x=10.4, y=20.6, w=30.2, h=40.8),
            detect.Box(cls="bush", score=0.5551, x=1.0, y=2.0, w=3.0, h=4.0),
        ]
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.fail:
            raise RuntimeError("bad input")
        return list(self.boxes)


class Harness:
    """A session and the doubles behind it, plus the fake clock the tests move."""

    def __init__(self, folder: Path, *, threaded: bool = False, sleep=None, source=None) -> None:
        self.sources: list[FakeSource] = []
        self.detector = FakeDetector()
        self.detector_calls = 0
        self.source_fails = False
        self.factory_error: Exception | None = None
        self.now = [100.0]
        self.session = session.PlaySession(
            folder=folder,
            source_factory=source or self._source,
            detector_factory=self._detector,
            clock=lambda: self.now[0],
            sleep=sleep or (lambda seconds: None),
            threaded=threaded,
            now=lambda: datetime(2026, 9, 20, 10, 11, 12),
        )

    def _source(self) -> FakeSource:
        if self.factory_error is not None and isinstance(self.factory_error, RuntimeError):
            raise self.factory_error
        made = FakeSource(fail=self.source_fails)
        self.sources.append(made)
        return made

    def _detector(self) -> FakeDetector:
        self.detector_calls += 1
        if self.factory_error is not None and not isinstance(self.factory_error, RuntimeError):
            raise self.factory_error
        return self.detector

    def run(self, ticks: int = 1) -> None:
        """What the thread would do: begin, then a few ticks."""
        self.session._begin()
        for _ in range(ticks):
            self.session._tick()


@pytest.fixture()
def h(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


def reasons(events) -> list[str]:
    return [fields["reason"] for kind, fields in events if kind == "play_fallback"]


def kinds(events) -> list[str]:
    return [kind for kind, _ in events]


def drain_for(h: Harness, kind: str, events: list) -> list:
    """Keep ticking the controller until the winding-down thread's row is drained."""
    for _ in range(500):
        if kind in kinds(events):
            break
        time.sleep(0.01)
        events = events + h.session.observe(State.RESULTS, "playing")
    return events


class HungCapture:
    """One frame, then a capture that hangs: what a stuck adb.screencap() does to a source.

    It is handed to the real ScreencapSource, so stopping that source joins a pump thread
    that will not return until the test lets go.
    """

    def __init__(self) -> None:
        self.held = threading.Event()
        self.calls = 0
        self.sources: list[capture.ScreencapSource] = []

    def __call__(self) -> np.ndarray:
        self.calls += 1
        if self.calls > 1:
            self.held.wait(HUNG_LIMIT)
        return np.zeros((900, 1600, 3), dtype=np.uint8)

    def source(self) -> capture.ScreencapSource:
        made = capture.ScreencapSource(capture=self, min_interval=0.0)
        self.sources.append(made)
        return made


def wait_for_a_frame(h: Harness) -> None:
    for _ in range(500):
        if h.detector.calls >= 1:
            return
        time.sleep(0.01)
    raise AssertionError("the session never read a frame")


# -- starting and stopping ---------------------------------------------------------


def test_the_default_source_is_a_screencap_source_and_captures_nothing_yet(monkeypatch) -> None:
    def never() -> np.ndarray:
        raise AssertionError("building the source must not capture")

    monkeypatch.setattr(capture.adb, "screencap", never)

    made = session._default_source()

    assert isinstance(made, capture.ScreencapSource)
    assert made.frames == 0 and made.error is None


def test_a_session_starts_only_on_a_match_in_the_playing_phase(h: Harness) -> None:
    assert h.session.observe(State.IN_MATCH, "returning") == []
    assert not h.session.active
    assert h.session.observe(State.MENU, "playing") == []
    assert not h.session.active
    assert h.session.observe(State.IN_MATCH, "playing") == []
    assert h.session.active


def test_the_happy_path_writes_a_file_and_reports_on_and_a_summary(h: Harness, tmp_path) -> None:
    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=3)
    h.now[0] += 12.0

    events = h.session.observe(State.RESULTS, "playing")

    assert kinds(events) == ["play_on", "play_summary"]
    on = events[0][1]
    assert on["shadow"] is True and on["model"] == "play-v1" and on["smoke"] is False
    assert on["provider"] == "CPUExecutionProvider" and isinstance(on["start_ms"], float)
    done = events[1][1]
    assert done["shadow"] is True and done["frames"] == 3
    assert done["boxes"] == {"enemy": 3, "bush": 3}
    assert done["stale_ticks"] == 0 and done["seconds"] == 12.0
    assert set(done) == {
        "shadow",
        "frames",
        "seconds",
        "fps",
        "ms_p50",
        "ms_p95",
        "stale_ticks",
        "boxes",
        "file",
        "rules_ms_p50",
        "rules_ms_p95",
        "coverage",
        "rule_calls",
        "rules_errors",
    }

    written = tmp_path / done["file"]
    assert written.parent == tmp_path  # the file field is a name, never a path
    lines = [json.loads(line) for line in written.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 4
    assert set(lines[0]) == HEADER_KEYS
    assert lines[0]["classes"] == ["enemy", "bush"] and lines[0]["rate_hz"] == session.RATE_HZ
    assert set(lines[1]) == {"t", "age", "ms", "boxes", "rules_ms", "read", "rules", "farm"}
    assert lines[1]["boxes"] == [["enemy", 0.912, 10, 20, 30, 40], ["bush", 0.555, 1, 2, 3, 4]]
    assert h.sources[0].stopped and not h.session.active


@pytest.mark.parametrize(
    ("state", "phase"),
    [
        (State.IN_MATCH, "returning"),
        (State.IN_MATCH, "at_menu"),
        (State.MENU, "playing"),
        (State.DISCONNECT, "playing"),
    ],
)
def test_every_way_out_of_a_match_ends_the_session(h: Harness, state, phase) -> None:
    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=1)

    events = h.session.observe(state, phase)

    assert kinds(events) == ["play_on", "play_summary"]
    assert not h.session.active and h.sources[0].stopped


def test_the_time_cap_ends_the_session(h: Harness) -> None:
    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=1)
    h.now[0] += session.MAX_SECONDS

    events = h.session.observe(State.IN_MATCH, "playing")

    assert kinds(events) == ["play_on", "play_summary"]
    assert not h.session.active and h.sources[0].stopped


def test_unknown_and_popup_inside_the_match_keep_the_session_open(h: Harness) -> None:
    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=1)

    assert kinds(h.session.observe(State.UNKNOWN, "playing")) == ["play_on"]
    assert h.session.observe(State.POPUP, "playing") == []
    assert h.session.active


# -- the ways it gives up ----------------------------------------------------------


def test_a_source_factory_that_raises_gives_one_fallback(h: Harness) -> None:
    h.factory_error = RuntimeError("no adb")
    h.session.observe(State.IN_MATCH, "playing")
    h.session._begin()

    events = h.session.observe(State.IN_MATCH, "playing")

    assert reasons(events) == ["source_start"]
    assert kinds(events) == ["play_fallback"]  # nothing was inferred, so no summary
    assert not h.session.active


def test_a_source_that_will_not_start_gives_one_fallback(h: Harness) -> None:
    h.source_fails = True
    h.session.observe(State.IN_MATCH, "playing")
    h.session._begin()

    events = h.session.observe(State.IN_MATCH, "playing")

    assert reasons(events) == ["source_start"] and not h.session.active


def test_a_source_error_inside_the_match_gives_one_fallback_and_a_summary(h: Harness) -> None:
    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=1)
    h.sources[0].error = "server gone"
    assert h.session._tick() is False
    assert h.session._tick() is False  # the thread may tick again; still one row

    events = h.session.observe(State.IN_MATCH, "playing")

    assert reasons(events) == ["source_error"]
    assert kinds(events) == ["play_on", "play_fallback", "play_summary"]
    assert h.sources[0].stopped


def test_a_stale_source_gives_up_only_after_the_limit(h: Harness) -> None:
    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=1)
    h.sources[0].frame = None
    h.now[0] += session.STALE_LIMIT - 1.0
    assert h.session._tick() is True
    h.now[0] += 1.0
    assert h.session._tick() is False

    events = h.session.observe(State.IN_MATCH, "playing")

    assert reasons(events) == ["stale"]
    assert dict(events)["play_summary"]["stale_ticks"] == 2


def test_a_detector_that_raises_gives_one_fallback_with_the_detail(h: Harness) -> None:
    h.session.observe(State.IN_MATCH, "playing")
    h.session._begin()
    h.detector.fail = True
    assert h.session._tick() is False

    events = h.session.observe(State.IN_MATCH, "playing")

    assert reasons(events) == ["detector_error"]
    detail = [f for k, f in events if k == "play_fallback"][0]["detail"]
    assert "bad input" in detail


def test_a_missing_model_is_one_quiet_fallback_and_no_session_ever_again(h: Harness) -> None:
    h.factory_error = detect.ModelMissing("no play.onnx")
    h.session.observe(State.IN_MATCH, "playing")
    h.session._begin()

    events = h.session.observe(State.IN_MATCH, "playing")
    assert events == [("play_fallback", {"reason": "model_missing", "shadow": True})]
    assert h.sources == []  # the source never starts when the model is not there

    assert h.session.observe(State.RESULTS, "playing") == []
    assert h.session.observe(State.IN_MATCH, "playing") == []
    assert not h.session.active


def test_an_invalid_model_is_one_fallback_with_the_detail(h: Harness) -> None:
    h.factory_error = detect.ModelInvalid("threshold for enemy is missing")
    h.session.observe(State.IN_MATCH, "playing")
    h.session._begin()

    events = h.session.observe(State.IN_MATCH, "playing")

    assert reasons(events) == ["model_invalid"]
    assert "threshold for enemy" in events[0][1]["detail"]
    assert h.session.observe(State.RESULTS, "playing") == []
    assert h.session.observe(State.IN_MATCH, "playing") == []


def test_a_missing_play_extra_no_longer_stops_shadow_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(play, "available", lambda: False)
    h = Harness(tmp_path)

    h.session.observe(State.IN_MATCH, "playing")
    assert h.session.active
    h.run(ticks=1)

    events = h.session.observe(State.RESULTS, "playing")

    assert kinds(events) == ["play_on", "play_summary"]
    assert reasons(events) == []


def test_a_mid_match_failure_waits_for_the_match_to_end_before_trying_again(h: Harness) -> None:
    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=1)
    h.sources[0].error = "server gone"
    h.session._tick()
    h.session.observe(State.IN_MATCH, "playing")

    assert h.session.observe(State.IN_MATCH, "playing") == []
    assert not h.session.active

    h.session.observe(State.RESULTS, "playing")
    h.session.observe(State.IN_MATCH, "playing")
    assert h.session.active
    h.run(ticks=1)
    assert len(h.sources) == 2 and h.sources[1].started


def test_the_detector_is_loaded_once_across_two_matches(h: Harness) -> None:
    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=1)
    h.session.observe(State.RESULTS, "playing")
    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=1)
    h.session.observe(State.RESULTS, "playing")

    assert h.detector_calls == 1 and len(h.sources) == 2


def test_old_shadow_files_are_pruned_to_the_cap(h: Harness, tmp_path: Path) -> None:
    for index in range(55):
        (tmp_path / f"20260919-00{index:04d}.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("not mine", encoding="utf-8")

    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=1)
    h.session.observe(State.RESULTS, "playing")

    assert len(list(tmp_path.glob("*.jsonl"))) == session.KEEP_FILES
    assert (tmp_path / "keep.txt").exists()


def test_a_second_session_in_the_same_second_gets_a_numbered_name(h: Harness) -> None:
    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=1)
    first = h.session.observe(State.RESULTS, "playing")[1][1]["file"]
    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=1)
    second = h.session.observe(State.RESULTS, "playing")[1][1]["file"]

    assert first == "20260920-101112.jsonl" and second == "20260920-101112-2.jsonl"


def test_close_ends_a_live_session_and_a_second_close_says_nothing(h: Harness) -> None:
    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=2)

    events = h.session.close()

    assert kinds(events) == ["play_on", "play_summary"]
    assert h.sources[0].stopped and not h.session.active
    assert h.session.close() == []


def test_close_without_a_session_says_nothing(h: Harness) -> None:
    assert h.session.close() == []


# -- the reader and the rules on each row --------------------------------------------


READING = reader.Reading(
    teams_left=4,
    self_health=np.int64(3200),
    self_health_fraction=np.float32(0.5),
    self_ammo=0,
    cube_count=2,
    enemy_health=(np.int64(1000), 800),
)
SELF_AND_CUBE = [
    detect.Box(cls="self", score=0.9, x=780.0, y=430.0, w=40.0, h=40.0),
    detect.Box(cls="power_cube", score=0.8, x=840.0, y=430.0, w=30.0, h=30.0),
]


def rows_of(folder: Path, done: dict) -> list[dict]:
    lines = (folder / done["file"]).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[1:]]


def reading_is(monkeypatch, *readings) -> None:
    """The reader hands these out in turn, the last one for good."""
    queue = list(readings)
    monkeypatch.setattr(
        reader, "read", lambda frame, boxes: queue.pop(0) if len(queue) > 1 else queue[0]
    )


def boom(*args, **kwargs):
    raise RuntimeError("bad read")


def test_a_row_carries_the_reader_the_rules_and_the_farm_label(h, tmp_path, monkeypatch) -> None:
    reading_is(monkeypatch, READING)
    h.detector.boxes = SELF_AND_CUBE
    h.session.observe(State.IN_MATCH, "playing")
    h.now[0] += 0.25
    h.run()

    done = dict(h.session.observe(State.RESULTS, "playing"))["play_summary"]

    row = rows_of(tmp_path, done)[0]
    assert row["read"] == {
        "teams_left": 4,
        "self_health": 3200,
        "self_health_fraction": 0.5,
        "self_ammo": 0,
        "cube_count": 2,
        "enemy_health": [1000, 800],
        "teammates_alive": None,
    }
    assert row["rules"] == {
        "centre": None,
        "cubes": {"label": "take_cube", "reason": "cube_near"},
        "exit": None,
    }
    assert row["farm"] == {"state": "in_match", "phase": "playing", "age": 0.25}
    assert row["boxes"][0] == ["self", 0.9, 780, 430, 40, 40]


def test_the_real_reader_runs_on_a_blank_frame_and_reads_nothing(h, tmp_path) -> None:
    h.session.observe(State.IN_MATCH, "playing")
    h.run()

    done = dict(h.session.observe(State.RESULTS, "playing"))["play_summary"]

    row = rows_of(tmp_path, done)[0]
    read = row["read"]
    assert read.pop("enemy_health") == [] and set(read.values()) == {None}
    assert row["rules"] == {"centre": None, "cubes": None, "exit": None}
    assert done["rules_errors"] == 0


def test_rules_ms_times_the_reader_and_the_rules_and_ms_stays_the_detector(
    h, tmp_path, monkeypatch
) -> None:
    def slow_read(frame, boxes):
        h.now[0] += 0.04
        return READING

    real = rules.evaluate

    def slow_rules(reading, boxes, context):
        h.now[0] += 0.01
        return real(reading, boxes, context)

    monkeypatch.setattr(reader, "read", slow_read)
    monkeypatch.setattr(rules, "evaluate", slow_rules)
    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=2)

    done = dict(h.session.observe(State.RESULTS, "playing"))["play_summary"]

    row = rows_of(tmp_path, done)[0]
    assert row["rules_ms"] == pytest.approx(50.0, abs=0.2) and row["ms"] == 0.0
    assert done["rules_ms_p50"] == pytest.approx(50.0, abs=0.2)
    assert done["rules_ms_p95"] == pytest.approx(50.0, abs=0.2)
    assert done["ms_p50"] == 0.0


def test_the_summary_has_coverage_per_field_and_calls_per_rule(h, monkeypatch) -> None:
    bare = reader.Reading(teams_left=4)
    reading_is(monkeypatch, READING, bare, bare, bare)
    h.detector.boxes = SELF_AND_CUBE
    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=4)

    done = dict(h.session.observe(State.RESULTS, "playing"))["play_summary"]

    assert done["coverage"] == {
        "teams_left": 1.0,
        "self_health": 0.25,
        "self_health_fraction": 0.25,
        "self_ammo": 0.25,  # a read of zero ammo is a read
        "cube_count": 0.25,
        "enemy_health": 0.25,  # an empty tuple is nothing read
        "teammates_alive": 0.0,
    }
    assert done["rule_calls"] == {"centre": 0, "cubes": 4, "exit": 0}
    assert done["rules_errors"] == 0


def test_a_lone_enemy_box_waits_for_the_frames_after_it_and_a_new_match_forgets(
    h, tmp_path, monkeypatch
) -> None:
    reading_is(monkeypatch, reader.Reading())  # no enemy bar: only the frames can back the box
    h.detector.boxes = [
        detect.Box(cls="self", score=0.9, x=780.0, y=430.0, w=40.0, h=40.0),
        detect.Box(cls="enemy", score=0.7, x=1080.0, y=430.0, w=40.0, h=40.0),
        detect.Box(cls="bush", score=0.7, x=580.0, y=430.0, w=40.0, h=40.0),
    ]
    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=rules.ENEMY_FRAMES)
    done = dict(h.session.observe(State.RESULTS, "playing"))["play_summary"]
    h.session.observe(State.MENU, "at_menu")

    calls = [row["rules"]["exit"] for row in rows_of(tmp_path, done)]
    assert calls[:-1] == [None] * (rules.ENEMY_FRAMES - 1)
    assert calls[-1] == {"label": "engage_ok", "reason": "bush_exit"}

    h.now[0] += 2.0
    h.session.observe(State.IN_MATCH, "playing")
    h.run()
    done = dict(h.session.observe(State.RESULTS, "playing"))["play_summary"]
    assert rows_of(tmp_path, done)[0]["rules"]["exit"] is None


def test_a_reader_that_raises_costs_the_fields_and_not_the_session(
    h, tmp_path, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(reader, "read", boom)
    h.session.observe(State.IN_MATCH, "playing")
    h.session._begin()

    with caplog.at_level("WARNING", logger="brawlfarm.play.session"):
        assert all(h.session._tick() for _ in range(3))
    assert h.session.active

    events = h.session.observe(State.RESULTS, "playing")

    assert kinds(events) == ["play_on", "play_summary"]
    done = dict(events)["play_summary"]
    assert done["frames"] == 3 and done["rules_errors"] == 3
    assert done["coverage"] == {} and done["rule_calls"] == {}
    rows = rows_of(tmp_path, done)
    assert [(row["read"], row["rules"]) for row in rows] == [(None, None)] * 3
    assert rows[0]["boxes"] and rows[0]["farm"]["state"] == "in_match"
    assert len([r for r in caplog.records if "rules error" in r.getMessage()]) == 1


def test_rules_that_raise_keep_the_reading_and_the_session(h, tmp_path, monkeypatch) -> None:
    reading_is(monkeypatch, READING)
    monkeypatch.setattr(rules, "evaluate", boom)
    h.session.observe(State.IN_MATCH, "playing")
    h.session._begin()

    assert h.session._tick() is True

    events = h.session.observe(State.RESULTS, "playing")
    assert "play_fallback" not in kinds(events)
    done = dict(events)["play_summary"]
    assert done["rules_errors"] == 1 and done["coverage"]["teams_left"] == 1.0
    row = rows_of(tmp_path, done)[0]
    assert row["read"]["teams_left"] == 4 and row["rules"] is None


def test_a_reading_json_cannot_hold_is_caught_before_the_row_is_written(
    h, tmp_path, monkeypatch
) -> None:
    reading_is(monkeypatch, reader.Reading(teams_left=object()))
    h.session.observe(State.IN_MATCH, "playing")
    h.session._begin()

    assert h.session._tick() is True

    done = dict(h.session.observe(State.RESULTS, "playing"))["play_summary"]
    assert done["rules_errors"] == 1 and rows_of(tmp_path, done)[0]["read"] is None


def test_a_threaded_session_outlives_a_reader_that_always_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(session, "RATE_HZ", 200.0)
    monkeypatch.setattr(reader, "read", boom)
    h = Harness(tmp_path, threaded=True, sleep=time.sleep)
    h.session.observe(State.IN_MATCH, "playing")
    for _ in range(500):
        if h.detector.calls >= 3:
            break
        time.sleep(0.01)
    assert h.detector.calls >= 3

    events = h.session.observe(State.IN_MATCH, "playing")
    events += h.session.observe(State.RESULTS, "playing")
    events = drain_for(h, "play_summary", events)

    assert "play_fallback" not in kinds(events)
    assert dict(events)["play_summary"]["rules_errors"] >= 3


# -- the real thread ---------------------------------------------------------------


def test_a_threaded_session_runs_and_ends_itself_when_the_match_ends(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(session, "RATE_HZ", 500.0)  # no real waiting in the test
    h = Harness(tmp_path, threaded=True, sleep=time.sleep)

    h.session.observe(State.IN_MATCH, "playing")
    for _ in range(300):  # the ticks happened; how fast they happened is not asserted
        if h.detector.calls >= 3:
            break
        time.sleep(0.01)

    events = drain_for(h, "play_summary", h.session.observe(State.RESULTS, "playing"))
    h.session.close()  # shutdown: the one place that waits for the thread

    assert kinds(events) == ["play_on", "play_summary"]
    assert dict(events)["play_summary"]["frames"] >= 1
    assert h.sources[0].stopped
    assert [t for t in threading.enumerate() if t.name == "play-shadow"] == []


def test_the_end_of_a_match_does_not_wait_for_a_hung_source(tmp_path: Path, monkeypatch):
    """The tick that ends a match joined the session thread and then the source's pump
    thread, up to JOIN_TIMEOUT each. It now signals and leaves, and the summary it used to
    queue itself arrives on a later tick."""
    monkeypatch.setattr(session, "RATE_HZ", 500.0)
    hung = HungCapture()
    h = Harness(tmp_path, threaded=True, sleep=time.sleep, source=hung.source)
    h.session.observe(State.IN_MATCH, "playing")
    wait_for_a_frame(h)

    began = time.monotonic()
    events = h.session.observe(State.RESULTS, "playing")
    ending = time.monotonic() - began
    began = time.monotonic()
    h.session.observe(State.IN_MATCH, "playing")  # the next match, with the last thread alive
    starting = time.monotonic() - began
    alive = [t for t in threading.enumerate() if t.name == "play-shadow"]
    hung.held.set()
    events = drain_for(h, "play_summary", events)
    h.session.close()

    assert ending < TICK_BOUND and starting < TICK_BOUND
    assert len(alive) == 1  # no second session under the one that is still winding down
    assert kinds(events) == ["play_on", "play_summary"]
    assert dict(events)["play_summary"]["frames"] >= 1
    assert hung.sources[0].frames >= 1  # it captured, and the thread that owned it
    assert [t for t in threading.enumerate() if t.name == "play-capture"] == []  # stopped it


def test_close_waits_for_a_hung_thread_but_only_once(tmp_path: Path, monkeypatch):
    """Shutdown is the one place a wait is allowed, and it is bounded by JOIN_TIMEOUT."""
    monkeypatch.setattr(session, "RATE_HZ", 500.0)
    monkeypatch.setattr(session, "JOIN_TIMEOUT", 0.25)
    hung = HungCapture()
    h = Harness(tmp_path, threaded=True, sleep=time.sleep, source=hung.source)
    h.session.observe(State.IN_MATCH, "playing")
    wait_for_a_frame(h)

    began = time.monotonic()
    events = h.session.close()
    closing = time.monotonic() - began
    hung.held.set()

    assert closing < TICK_BOUND
    assert kinds(events) == ["play_on"]  # the summary belongs to the thread still in its stop
    assert not h.session.active and h.session.close() == []


# -- rails -------------------------------------------------------------------------


def test_no_module_under_play_reaches_for_an_input_helper() -> None:
    banned = ("tap", "swipe", "input_", "keyevent")
    folder = Path(session.__file__).resolve().parent
    for source in sorted(folder.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "brawlfarm.core.adb":
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                names = [node.attr] if node.value.id == "adb" else []
            else:
                continue
            for name in names:
                assert not any(word in name for word in banned), f"{source.name}: {name}"


def test_a_whole_session_with_the_rules_sends_no_input(h, monkeypatch) -> None:
    sent: list[str] = []
    for name in ("tap", "swipe", "tap_hold", "keyevent", "input_text", "go_home", "_write_touch"):
        monkeypatch.setattr(capture.adb, name, lambda *a, name=name, **k: sent.append(name))
    reading_is(monkeypatch, READING)
    h.detector.boxes = SELF_AND_CUBE

    h.session.observe(State.IN_MATCH, "playing")
    h.run(ticks=3)
    events = h.session.observe(State.RESULTS, "playing")

    assert dict(events)["play_summary"]["rule_calls"]["cubes"] == 3
    assert sent == []


def test_the_session_and_the_rules_stay_off_the_stream() -> None:
    folder = Path(session.__file__).resolve().parent
    for name in ("session.py", "rules.py"):
        tree = ast.parse((folder / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = [node.module or ""] + [alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            else:
                continue
            assert not any("stream" in word or "scrcpy" in word for word in imported), name


def test_importing_the_session_pulls_nothing_heavy() -> None:
    code = (
        "import sys, brawlfarm.play.session; "
        "print([m for m in ('onnxruntime', 'av') if m in sys.modules])"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"
