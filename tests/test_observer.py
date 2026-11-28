"""Observe mode: the loop records labeled frames off a fake adb that cannot tap, keeps
record.flag for exactly as long as it runs, refuses a display that is not 1600x900, and
stops on stop.flag. No real device, no real screencap, no tap function anywhere."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from brawlfarm.core import adb, config, observer


def _frame() -> np.ndarray:
    return np.zeros((config.SCREEN_H, config.SCREEN_W, 3), np.uint8)


@pytest.fixture()
def data(tmp_path: Path, monkeypatch) -> Path:
    """An instance folder under a throwaway home, with adb faked out entirely."""
    inst = tmp_path / "instances" / "alpha"
    inst.mkdir(parents=True)
    monkeypatch.setattr(config, "HOME_DIR", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", inst)
    monkeypatch.setattr(observer, "POLL_INTERVAL_S", 0.0)
    monkeypatch.setattr(adb, "connect", lambda serial=None: None)
    monkeypatch.setattr(adb, "emulator_mismatch", lambda: None)
    monkeypatch.setattr(adb, "screen_size", lambda: (config.SCREEN_W, config.SCREEN_H))
    monkeypatch.setattr(adb, "screencap", lambda serial=None: _frame())
    return inst


def test_one_pass_records_a_frame_and_leaves_no_flag(data: Path, tmp_path: Path) -> None:
    obs = observer.Observer(max_minutes=0.0)  # the cap is checked after the first frame
    obs.run()
    assert obs.frames == 1
    assert not (data / "record.flag").exists()
    sessions = sorted((tmp_path / "calibration" / "recordings" / "alpha").iterdir())
    assert len(sessions) == 1
    assert (sessions[0] / "labels.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_the_recorded_frames_are_full_size(data: Path, tmp_path: Path) -> None:
    observer.Observer(max_minutes=0.0).run()
    session = sorted((tmp_path / "calibration" / "recordings" / "alpha").iterdir())[0]
    frame = cv2.imread(str(next(session.glob("*.jpg"))))
    assert frame is not None
    assert frame.shape == (config.SCREEN_H, config.SCREEN_W, 3)


def test_the_heartbeat_names_the_mode(data: Path) -> None:
    observer.Observer(max_minutes=0.0).run()
    from brawlfarm.core import status

    heartbeat = status.read_status(data)
    assert heartbeat is not None
    assert heartbeat["mode"] == "observe"
    assert heartbeat["phase"] == "observing"
    assert heartbeat["running"] is False


def test_stop_flag_ends_the_session(data: Path, monkeypatch) -> None:
    seen: list[int] = []

    def screencap(serial=None):
        seen.append(1)
        if len(seen) == 2:  # a leftover flag is cleared at startup, so raise it mid-loop
            (data / "stop.flag").write_text("stop\n", encoding="utf-8")
        return _frame()

    monkeypatch.setattr(adb, "screencap", screencap)
    observer.Observer().run()
    assert len(seen) == 2
    assert not (data / "record.flag").exists()


def test_a_display_that_is_not_1600x900_never_starts(data: Path, monkeypatch) -> None:
    monkeypatch.setattr(adb, "screen_size", lambda: (1280, 720))
    with pytest.raises(SystemExit, match="1280x720"):
        observer.Observer().run()
    assert not (data / "record.flag").exists()


def test_an_exception_still_clears_the_flag(data: Path, monkeypatch) -> None:
    def boom(serial=None):
        raise RuntimeError("device gone")

    monkeypatch.setattr(adb, "screencap", boom)
    with pytest.raises(RuntimeError):
        observer.Observer().run()
    assert not (data / "record.flag").exists()


def test_a_match_frame_opens_a_match_recording_in_the_session(
    data: Path, tmp_path: Path, monkeypatch
) -> None:
    from brawlfarm.core import states
    from brawlfarm.core.states import State
    from brawlfarm.play import matchrec

    opened: list[object] = []

    class FakeSource:
        error = None

        def __init__(self) -> None:
            opened.append(self)

        def start(self) -> None:
            pass

        def latest(self):
            return None, None

        def stop(self) -> None:
            pass

    monkeypatch.setattr(matchrec, "_default_factory", FakeSource)
    monkeypatch.setattr(states, "classify", lambda screen, phase=None: State.IN_MATCH)
    observer.Observer(max_minutes=0.0).run()
    session = sorted((tmp_path / "calibration" / "recordings" / "alpha").iterdir())[0]
    assert len(opened) == 1
    assert (session / "match-1").is_dir()


# --- the rails ------------------------------------------------------------------------
#
# Observe mode's whole safety argument is that no tap function is reachable from it. Both
# halves of that are checked here: the source names no input call, and importing the
# module pulls in nothing that owns one. A future edit that breaks either CANNOT land
# green, which is the point, so treat a failure here as a design question and not a test
# to fix.

NO_INPUT = (
    "adb.tap",
    "adb.swipe",
    "adb.tap_hold",
    "adb.input_text",
    "adb.keyevent",
    "adb.go_home",
    "adb.launch_app",
    "adb.force_stop",
)

FORBIDDEN_MODULES = (
    "brawlfarm.core.controller",
    "brawlfarm.core.brawlers",
    "brawlfarm.core.quests",
    "brawlfarm.core.rewards",
    "brawlfarm.core.settings",
)


def test_the_observer_source_names_no_input_call() -> None:
    source = Path(observer.__file__).read_text(encoding="utf-8")
    named = [call for call in NO_INPUT if call in source]
    assert named == [], f"observer.py must never tap: {named}"


def test_importing_the_observer_pulls_in_nothing_that_taps(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if not k.startswith(("BRAWL_", "DISCORD_"))}
    env["BRAWLFARM_HOME"] = str(tmp_path)
    code = (
        "import sys; import brawlfarm.core.observer; "
        f"bad = [m for m in {FORBIDDEN_MODULES!r} if m in sys.modules]; "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
