"""core/preview.py: the small JPEG the worker writes once a second from the frame the
main loop already holds, so the panel can watch an instance without a second screencap.

The rail these tests exist for is harmlessness. maybe_write runs inside the farm's hot
loop, so a failed encode, a data dir that is a file, or a disk that will not take the
bytes must return False and nothing else -- no exception, no half-written preview.jpg,
no .tmp left behind, and at most one log line a minute.
"""

from __future__ import annotations

import logging
import pathlib

import cv2
import numpy as np
import pytest

from brawlfarm.core import config, preview
from brawlfarm.core import controller as controller_mod
from brawlfarm.core.controller import Controller
from brawlfarm.core.states import State

JPEG_SIGNATURE = b"\xff\xd8\xff"


@pytest.fixture(autouse=True)
def _fresh_throttles(monkeypatch):
    """The write and warn throttles are module state; put them back between tests."""
    monkeypatch.setattr(preview, "_last_write", float("-inf"))
    monkeypatch.setattr(preview, "_last_warn", float("-inf"))


def _frame(value: int = 255) -> np.ndarray:
    """A flat 1600x900 BGR frame, the shape the controller hands to maybe_write."""
    return np.full((config.SCREEN_H, config.SCREEN_W, 3), value, dtype=np.uint8)


def _preview_path():
    return config.DATA_DIR / preview.PREVIEW_NAME


def test_maybe_write_writes_a_half_size_jpeg() -> None:
    assert preview.maybe_write(_frame()) is True
    path = _preview_path()
    assert path.read_bytes().startswith(JPEG_SIGNATURE)
    frame = cv2.imread(str(path))
    assert frame is not None
    assert frame.shape == (450, 800, 3)
    assert not path.with_name(path.name + ".tmp").exists()


def test_maybe_write_leaves_the_captured_frame_alone() -> None:
    screen = _frame(128)
    before = screen.copy()
    assert preview.maybe_write(screen) is True
    assert np.array_equal(screen, before)


def test_a_second_call_inside_the_interval_writes_nothing() -> None:
    assert preview.maybe_write(_frame(255), now=100.0) is True
    first = _preview_path().read_bytes()
    assert preview.maybe_write(_frame(0), now=100.0 + preview.PREVIEW_INTERVAL_S / 2) is False
    assert _preview_path().read_bytes() == first
    assert preview.maybe_write(_frame(0), now=100.0 + preview.PREVIEW_INTERVAL_S) is True
    assert _preview_path().read_bytes() != first


def test_encode_raises_when_cv2_cannot_encode(monkeypatch) -> None:
    monkeypatch.setattr(preview.cv2, "imencode", lambda *a, **kw: (False, None))
    with pytest.raises(ValueError):
        preview.encode(_frame())


def test_a_failed_encode_is_swallowed(monkeypatch) -> None:
    monkeypatch.setattr(preview.cv2, "imencode", lambda *a, **kw: (False, None))
    assert preview.maybe_write(_frame()) is False
    assert not _preview_path().exists()


def test_a_raising_encode_is_swallowed(monkeypatch) -> None:
    def boom(*a, **kw):
        raise cv2.error("imencode blew up")

    monkeypatch.setattr(preview.cv2, "imencode", boom)
    assert preview.maybe_write(_frame()) is False
    assert not _preview_path().exists()


def test_a_data_dir_that_is_a_file_is_swallowed(tmp_path, monkeypatch) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the data dir should be", encoding="utf-8")
    monkeypatch.setattr(config, "DATA_DIR", blocked)
    assert preview.maybe_write(_frame()) is False


def test_failures_are_logged_at_most_once_a_minute(monkeypatch, caplog) -> None:
    monkeypatch.setattr(preview.cv2, "imencode", lambda *a, **kw: (False, None))
    with caplog.at_level(logging.WARNING, logger="brawlfarm.core.preview"):
        assert preview.maybe_write(_frame(), now=100.0) is False
        assert preview.maybe_write(_frame(), now=101.0) is False
        assert len(caplog.records) == 1
        assert preview.maybe_write(_frame(), now=100.0 + preview.WARN_INTERVAL_S) is False
        assert len(caplog.records) == 2


# --- the loop calls it ---------------------------------------------------------


class _DL:
    """The datalog seam: run() writes events and reads the session file's name."""

    def __init__(self):
        self.events = []
        self.session_path = pathlib.Path("session.jsonl")

    def event(self, etype, **fields):
        self.events.append((etype, fields))


class _Rec:
    """The recorder seam: run() polls the flag, observes every frame and closes
    on exit. Off in this test, so all three are no-ops."""

    def poll(self):
        pass

    def observe(self, screen, state, phase):
        return False

    def close(self):
        pass


def _one_iteration_controller(monkeypatch, frame) -> Controller:
    """A bare Controller whose run() does exactly one loop iteration (the
    tests/test_network_stuck.py pattern: __new__ plus the attributes the loop reads, every
    seam stubbed). The phase handler ends the run, so the loop never comes round again."""
    monkeypatch.setattr(controller_mod.adb, "connect", lambda *a, **kw: None)
    monkeypatch.setattr(controller_mod.adb, "emulator_mismatch", lambda: None)
    monkeypatch.setattr(
        controller_mod.adb, "screen_size", lambda: (config.SCREEN_W, config.SCREEN_H)
    )
    monkeypatch.setattr(controller_mod.adb, "screencap", lambda *a, **kw: frame)
    monkeypatch.setattr(controller_mod.states, "classify", lambda screen, phase=None: State.MENU)
    monkeypatch.setattr(config, "LOOP_POLL_INTERVAL", 0)
    monkeypatch.setattr(Controller, "log", lambda self, msg: None)
    monkeypatch.setattr(Controller, "ensure_game_open", lambda self: True)
    monkeypatch.setattr(Controller, "_check_stop_backstop", lambda self: False)
    monkeypatch.setattr(Controller, "check_freeze", lambda self, screen: None)
    monkeypatch.setattr(
        Controller, "phase_returning", lambda self, screen, state: setattr(self, "running", False)
    )

    c = Controller.__new__(Controller)  # skip __init__ (ApiClient/DataLog/adb)
    c.dl = _DL()
    c.phase = "returning"
    c._prev_phase = "returning"  # equal, so the loop takes no debug shot
    c.running = True
    c.max_games = None
    c.max_minutes = None
    c._adb_errors = 0
    c._unknown_streak = 0
    c._loop_i = 0
    c.recorder = _Rec()
    return c


def test_the_loop_hands_the_captured_frame_to_maybe_write(monkeypatch) -> None:
    frame = _frame(64)
    seen: list[object] = []
    monkeypatch.setattr(preview, "maybe_write", lambda screen: seen.append(screen))
    _one_iteration_controller(monkeypatch, frame).run()
    assert len(seen) == 1
    assert seen[0] is frame  # the frame it already has, never a second capture
