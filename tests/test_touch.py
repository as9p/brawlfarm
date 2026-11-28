"""Touch frames for BlueStacks and MuMu, from the two `getevent -p` dumps measured on this
machine (BlueStacks 2026-11-27, MuMu 2026-11-28). BlueStacks frames must stay byte for byte
what the farm has always sent; MuMu frames are type B in the portrait node frame, and the
probe's measured tap (screen 750, 66 landing at node 834, 750) is the anchor."""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from brawlfarm.core import adb, config, touch

FIXTURES = Path(__file__).parent / "fixtures" / "getevent"
MUMU = (FIXTURES / "mumu.txt").read_text(encoding="utf-8")
BLUESTACKS = (FIXTURES / "bluestacks.txt").read_text(encoding="utf-8")


def _decode(blob: bytes) -> list[tuple[int, int, int]]:
    return [struct.unpack("<qqHHi", blob[i : i + 24])[2:] for i in range(0, len(blob), 24)]


def _mumu_profile(rotation: int | None = 1) -> touch.TouchProfile:
    node = touch.pick_touch_node(touch.parse_getevent(MUMU))
    assert node is not None
    return touch.TouchProfile.from_node(node, 1600, 900, rotation)


# --- parsing -----------------------------------------------------------------------------


def test_mumu_dump_yields_the_xiaomi_touchscreen_as_a_type_b_portrait_node() -> None:
    node = touch.pick_touch_node(touch.parse_getevent(MUMU))
    assert node is not None
    assert (node.path, node.name) == ("/dev/input/event4", "Xiaomi Touchscreen")
    assert node.abs[touch.ABS_MT_POSITION_X] == (0, 900)
    assert node.abs[touch.ABS_MT_POSITION_Y] == (0, 1600)
    assert touch.ABS_MT_SLOT in node.abs
    assert "INPUT_PROP_DIRECT" in node.props
    assert touch.BTN_TOUCH not in node.keys  # it declares BTN_MOUSE and BTN_TOOL_FINGER


def test_bluestacks_dump_yields_the_virtual_touch_node_as_type_a() -> None:
    node = touch.pick_touch_node(touch.parse_getevent(BLUESTACKS))
    assert node is not None
    assert (node.path, node.name) == ("/dev/input/event4", "BlueStacks Virtual Touch")
    assert node.abs == {touch.ABS_MT_POSITION_X: (0, 32767), touch.ABS_MT_POSITION_Y: (0, 32767)}


def test_mouse_and_joystick_nodes_are_never_picked() -> None:
    nodes = [n for n in touch.parse_getevent(MUMU) if n.name != "Xiaomi Touchscreen"]
    assert touch.pick_touch_node(nodes) is None
    assert touch.pick_touch_node([]) is None


def test_the_node_names_tell_the_two_emulators_apart() -> None:
    assert touch.emulator_of(touch.parse_getevent(BLUESTACKS)) == "bluestacks"
    assert touch.emulator_of(touch.parse_getevent(MUMU)) == "other"


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("  Viewport INTERNAL: displayId=0, orientation=1, logicalFrame=[0, 0, 1600, 900]", 1),
        ("    SurfaceOrientation: 3\n", 3),
        ("mCurrentRotation=ROTATION_270", 3),
        ("mCurrentRotation=2", 2),
        ("nothing about rotation here", None),
        ("", None),
    ],
)
def test_parse_rotation(text: str, want: int | None) -> None:
    assert touch.parse_rotation(text) == want


def test_an_unread_rotation_is_inferred_from_the_node_aspect() -> None:
    assert _mumu_profile(rotation=None).rotation == 1
    bs = touch.pick_touch_node(touch.parse_getevent(BLUESTACKS))
    assert touch.default_rotation(bs, 1600, 900) == 0


# --- coordinates -------------------------------------------------------------------------


def test_rotation_1_maps_the_measured_probe_tap() -> None:
    prof = _mumu_profile()
    assert prof.to_node(750, 66) == (834, 750)  # tap-mumu-rot-b, 2026-11-28 01:36
    assert prof.to_node(0, 0) == (900, 0)
    assert prof.to_node(1600, 900) == (0, 1600)


def test_rotation_3_is_the_mirror_turn() -> None:
    prof = _mumu_profile(rotation=3)
    assert prof.to_node(750, 66) == (66, 850)
    assert prof.to_node(0, 0) == (0, 1600)


def test_coordinates_are_clamped_to_the_node_range() -> None:
    prof = _mumu_profile()
    assert prof.to_node(-20, 2000) == (0, 0)


# --- frames ------------------------------------------------------------------------------


def _old_bluestacks_frame(x: int, y: int) -> bytes:
    """The pre-MuMu _touch_frame, copied verbatim, as the byte-for-byte reference."""

    def ev(t, c, v):
        return struct.pack("<qqHHi", 0, 0, t, c, v)

    dx = round(int(x) / 1600 * 32767)
    dy = round(int(y) / 900 * 32767)
    return ev(3, 53, dx) + ev(3, 54, dy) + ev(0, 2, 0) + ev(0, 0, 0)


@pytest.mark.parametrize(("x", "y"), [(0, 0), (800, 450), (1599, 899), (1323, 761), (1600, 900)])
def test_bluestacks_frames_are_unchanged(x: int, y: int) -> None:
    prof = touch.bluestacks_profile("/dev/input/event4", 32767, 32767, 1600, 900)
    old_up = struct.pack("<qqHHi", 0, 0, 0, 2, 0) + struct.pack("<qqHHi", 0, 0, 0, 0, 0)
    assert prof.down(x, y) == _old_bluestacks_frame(x, y)
    assert prof.move(x, y) == _old_bluestacks_frame(x, y)
    assert prof.up() == old_up


def test_type_b_down_move_up() -> None:
    prof = _mumu_profile()
    s, tid, px, py, syn = (
        touch.ABS_MT_SLOT,
        touch.ABS_MT_TRACKING_ID,
        touch.ABS_MT_POSITION_X,
        touch.ABS_MT_POSITION_Y,
        touch.SYN_REPORT,
    )
    assert _decode(prof.down(750, 66)) == [
        (3, s, 0),
        (3, tid, 1),
        (3, px, 834),
        (3, py, 750),
        (0, syn, 0),
    ]
    assert _decode(prof.move(800, 450)) == [(3, s, 0), (3, px, 450), (3, py, 800), (0, syn, 0)]
    assert _decode(prof.up()) == [(3, s, 0), (3, tid, -1), (0, syn, 0)]
    assert _decode(prof.down(1, 1))[1] == (3, tid, 2)  # each touch gets a fresh id


def test_type_b_sends_btn_touch_when_the_node_declares_it() -> None:
    prof = _mumu_profile()
    prof.btn_touch = True
    assert (1, touch.BTN_TOUCH, 1) in _decode(prof.down(10, 10))
    assert (1, touch.BTN_TOUCH, 0) in _decode(prof.up())


def test_tracking_id_wraps_before_its_maximum() -> None:
    prof = _mumu_profile()
    prof._next_id = touch.TRACKING_ID_MAX
    assert _decode(prof.down(1, 1))[1][2] == 1


# --- adb wiring --------------------------------------------------------------------------


@pytest.fixture()
def fake_adb(monkeypatch):
    """adb._run answering getevent, dumpsys and wm from canned text; records every call."""
    answers = {"getevent": MUMU, "dumpsys": "Viewport INTERNAL: orientation=1,", "wm": ""}
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        joined = " ".join(args)
        for key, text in answers.items():
            if key in joined:
                return text
        return ""

    monkeypatch.setattr(adb, "_run", fake_run)
    monkeypatch.setattr(config, "ADB_SERIAL", "127.0.0.1:16384")
    monkeypatch.setattr(config, "FAST_INPUT", True)
    monkeypatch.setattr(config, "EMULATOR", "mumu")
    adb.reset_touch_profile()
    yield answers, calls
    adb.reset_touch_profile()


def _octal(blob: bytes) -> str:
    return "".join("\\%03o" % b for b in blob)


def test_mumu_tap_writes_type_b_down_then_up_in_one_shell_call(fake_adb) -> None:
    _, calls = fake_adb
    adb.tap(750, 66)
    writes = [c for c in calls if "printf" in c[-1]]
    assert len(writes) == 1
    script = writes[0][-1]
    prof = _mumu_profile()
    down, up = prof.down(750, 66), prof.up()
    assert script == (
        f"printf '{_octal(down)}' > /dev/input/event4; "
        f"sleep {config.TOUCH_HOLD_S:g}; printf '{_octal(up)}' > /dev/input/event4"
    )
    adb.tap(10, 10)  # the profile is read once, not per tap
    assert sum("getevent" in " ".join(c) for c in calls) == 1


def test_mumu_without_a_touch_node_falls_back_to_input_tap(fake_adb) -> None:
    answers, calls = fake_adb
    answers["getevent"] = 'add device 1: /dev/input/event0\n  name:     "Power Button"\n'
    adb.tap(750, 66)
    adb.tap(751, 66)
    assert calls[-1] == ["-s", "127.0.0.1:16384", "shell", "input", "tap", "751", "66"]
    assert sum("getevent" in " ".join(c) for c in calls) == 1  # not re-read every tap


def test_a_failing_touch_node_read_is_retried_only_after_a_wait(fake_adb, monkeypatch) -> None:
    _, calls = fake_adb
    clock = [1000.0]
    monkeypatch.setattr(adb.time, "monotonic", lambda: clock[0])
    real_run = adb._run

    def failing_getevent(args, **kwargs):
        if "getevent" in args:
            calls.append(list(args))
            raise adb.AdbError("device offline")
        return real_run(args, **kwargs)

    monkeypatch.setattr(adb, "_run", failing_getevent)
    adb.tap(750, 66)
    adb.tap(751, 66)
    assert sum("getevent" in c for c in calls) == 1  # the second tap did not ask again
    assert calls[-1] == ["-s", "127.0.0.1:16384", "shell", "input", "tap", "751", "66"]
    clock[0] += config.TOUCH_PROFILE_RETRY_S + 1
    adb.tap(752, 66)
    assert sum("getevent" in c for c in calls) == 2


def test_a_failed_type_b_tap_lifts_the_finger_before_the_fallback(fake_adb, monkeypatch) -> None:
    _, calls = fake_adb
    adb.tap(1, 1)  # read the profile
    real_run = adb._run
    first = [True]

    def die_mid_tap(args, **kwargs):
        if "sleep" in args[-1] and first[0]:
            first[0] = False
            calls.append(list(args))
            raise adb.AdbError("timed out")
        return real_run(args, **kwargs)

    monkeypatch.setattr(adb, "_run", die_mid_tap)
    calls.clear()
    adb.tap(750, 66)
    up = f"printf '{_octal(_mumu_profile().up())}' > /dev/input/event4"
    assert calls[1] == ["-s", "127.0.0.1:16384", "shell", up]
    assert calls[2] == ["-s", "127.0.0.1:16384", "shell", "input", "tap", "750", "66"]


def test_bluestacks_tap_sends_the_same_command_as_before(fake_adb, monkeypatch) -> None:
    _, calls = fake_adb
    monkeypatch.setattr(config, "EMULATOR", "bluestacks")
    monkeypatch.setattr(config, "ADB_SERIAL", "127.0.0.1:5555")
    adb.tap(800, 450)
    frame = _old_bluestacks_frame(800, 450)
    up = struct.pack("<qqHHi", 0, 0, 0, 2, 0) + struct.pack("<qqHHi", 0, 0, 0, 0, 0)
    assert calls == [
        ["-s", "127.0.0.1:5555", "shell", f"printf '{_octal(frame + up)}' > /dev/input/event4"]
    ]


def test_mumu_swipe_streams_moves_and_lifts(fake_adb, monkeypatch) -> None:
    _, calls = fake_adb
    monkeypatch.setattr(adb.time, "sleep", lambda s: None)
    adb.swipe(100, 100, 400, 100, duration_ms=60)
    writes = [c[-1] for c in calls if "printf" in c[-1]]
    assert len(writes) == 4  # down, two moves, up
    assert all(w.endswith("> /dev/input/event4") for w in writes)
    assert _octal(_mumu_profile().up()) in writes[-1]


def test_connect_forgets_the_profile(fake_adb, monkeypatch) -> None:
    answers, calls = fake_adb
    answers["devices"] = "List of devices attached\n127.0.0.1:16384\tdevice\n"
    adb.tap(1, 1)
    adb.connect()
    adb.tap(1, 1)
    assert sum("getevent" in " ".join(c) for c in calls) == 2


def test_rotation_is_read_again_after_its_ttl(fake_adb, monkeypatch) -> None:
    answers, calls = fake_adb
    clock = [1000.0]
    monkeypatch.setattr(adb.time, "monotonic", lambda: clock[0])
    adb.tap(1, 1)
    answers["dumpsys"] = "SurfaceOrientation: 3"
    clock[0] += config.TOUCH_ROTATION_TTL_S + 1
    adb.tap(1, 1)
    assert adb._touch_profile.rotation == 3


@pytest.mark.parametrize(
    ("emulator", "dump", "wrong"),
    [
        ("bluestacks", BLUESTACKS, False),
        ("bluestacks", MUMU, True),  # MuMu answered on 5555
        ("mumu", MUMU, False),
        ("mumu", BLUESTACKS, True),
        ("bluestacks", "", False),  # unreadable: not called a mismatch
    ],
)
def test_emulator_mismatch(fake_adb, monkeypatch, emulator, dump, wrong) -> None:
    answers, _ = fake_adb
    answers["getevent"] = dump
    monkeypatch.setattr(config, "EMULATOR", emulator)
    assert (adb.emulator_mismatch() is not None) is wrong


def test_screen_size_turns_a_portrait_display_by_its_frame(fake_adb, monkeypatch) -> None:
    answers, _ = fake_adb
    answers["wm"] = "Physical size: 900x1600\n"
    monkeypatch.setattr(adb, "screencap", lambda serial=None: np.zeros((900, 1600, 3), np.uint8))
    assert adb.screen_size() == (1600, 900)
    answers["wm"] = "Physical size: 1600x900\n"
    assert adb.screen_size() == (1600, 900)
    # A BlueStacks instance left portrait is reported as it is, so the startup check fails.
    monkeypatch.setattr(config, "EMULATOR", "bluestacks")
    answers["wm"] = "Physical size: 900x1600\n"
    assert adb.screen_size() == (900, 1600)


def test_mumu_defaults_to_port_16384() -> None:
    code = "from brawlfarm.core import config; print(config.ADB_SERIAL, config.EMULATOR)"
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("BRAWL_")}
    for emulator, want in (("mumu", "127.0.0.1:16384 mumu"), ("nonsense", "127.0.0.1:5555")):
        out = subprocess.run(
            [sys.executable, "-c", code],
            env={**env, "BRAWL_EMULATOR": emulator},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert out.startswith(want)
