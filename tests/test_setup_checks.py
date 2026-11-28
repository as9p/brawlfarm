"""The panel's own adb wrapper: argv lists only, a serial built from a validated integer
port, a screencap that refuses anything that is not a PNG, and the 1600 x 900 / DPI 240
display check with the sentence that tells the user where to change it."""

from __future__ import annotations

import struct
import subprocess

import pytest

from brawlfarm.core import config
from brawlfarm.setup import checks


def _done(
    argv: list[str], rc: int = 0, out: bytes = b"", err: bytes = b""
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(argv, rc, out, err)


class RecordingRunner:
    """A checks.Runner: answers by the first key found in the joined argv, records every
    call, and never starts a process."""

    def __init__(self, answers: dict[str, subprocess.CompletedProcess]) -> None:
        self.answers = answers
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], timeout_s: float) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        joined = " ".join(argv)
        for key, done in self.answers.items():
            if key in joined:
                return done
        return _done(argv, rc=1, err=b"unexpected argv")


def _png(width: int, height: int) -> bytes:
    """A PNG signature plus an IHDR chunk: all png_size and the guard ever look at."""
    ihdr = struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00"
    return checks.PNG_SIGNATURE + struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr


def test_run_adb_passes_an_argv_list_with_the_binary_first() -> None:
    runner = RecordingRunner({"devices": _done(["adb", "devices"], out=b"List of devices\n")})
    done = checks.run_adb("C:/tools/adb.exe", ["devices"], runner=runner)
    assert runner.calls == [["C:/tools/adb.exe", "devices"]]
    assert done.returncode == 0


def test_serial_is_built_from_an_integer_port_only() -> None:
    assert checks.serial(5555) == "127.0.0.1:5555"
    assert checks.serial("5565") == "127.0.0.1:5565"
    with pytest.raises(ValueError):
        checks.serial("5555 && shutdown")


def test_screencap_returns_the_png_bytes() -> None:
    frame = _png(1600, 900)
    runner = RecordingRunner(
        {
            "connect": _done(["adb", "connect"], out=b"connected to 127.0.0.1:5555\n"),
            "screencap": _done(["adb", "screencap"], out=frame),
        }
    )
    assert checks.screencap_png("C:/tools/adb.exe", 5555, runner=runner) == frame
    assert runner.calls[1] == [
        "C:/tools/adb.exe",
        "-s",
        "127.0.0.1:5555",
        "exec-out",
        "screencap",
        "-p",
    ]


def test_screencap_raises_when_adb_fails() -> None:
    runner = RecordingRunner(
        {
            "connect": _done(["adb", "connect"], out=b"connected\n"),
            "screencap": _done(["adb", "screencap"], rc=1, err=b"error: device offline\n"),
        }
    )
    with pytest.raises(checks.AdbUnavailable) as exc:
        checks.screencap_png("C:/tools/adb.exe", 5555, runner=runner)
    assert "device offline" in str(exc.value)


def test_screencap_raises_when_the_output_is_not_a_png() -> None:
    runner = RecordingRunner(
        {
            "connect": _done(["adb", "connect"], out=b"connected\n"),
            "screencap": _done(["adb", "screencap"], out=b"error: closed\n"),
        }
    )
    with pytest.raises(checks.AdbUnavailable) as exc:
        checks.screencap_png("C:/tools/adb.exe", 5555, runner=runner)
    assert "did not return a PNG" in str(exc.value)


def test_png_size_reads_the_ihdr_chunk() -> None:
    assert checks.png_size(_png(1600, 900)) == (1600, 900)
    with pytest.raises(checks.AdbUnavailable):
        checks.png_size(b"not a png at all")


def test_parse_wm_size_prefers_the_override_line() -> None:
    assert checks.parse_wm_size("Physical size: 1600x900\n") == (1600, 900)
    assert checks.parse_wm_size("Physical size: 1920x1080\nOverride size: 1600x900\n") == (
        1600,
        900,
    )
    assert checks.parse_wm_size("") is None
    assert checks.parse_wm_size("error: device offline") is None


def test_parse_wm_density_prefers_the_override_line() -> None:
    assert checks.parse_wm_density("Physical density: 240\n") == 240
    assert checks.parse_wm_density("Physical density: 320\nOverride density: 240\n") == 240
    assert checks.parse_wm_density("nothing here") is None


def test_display_check_passes_at_the_resolution_the_controller_asserts() -> None:
    runner = RecordingRunner(
        {
            "connect": _done(["adb", "connect"], out=b"connected\n"),
            "wm size": _done(["adb", "wm", "size"], out=b"Physical size: 1600x900\n"),
            "wm density": _done(["adb", "wm", "density"], out=b"Physical density: 240\n"),
        }
    )
    result = checks.display_check("C:/tools/adb.exe", 5555, runner=runner)
    assert result.ok is True
    assert (result.width, result.height, result.dpi) == (
        config.SCREEN_W,
        config.SCREEN_H,
        config.SCREEN_DPI,
    )
    assert result.detail == "1600 x 900 at DPI 240"
    assert result.hint == ""


def _portrait_runner(frame: bytes) -> RecordingRunner:
    """MuMu: wm size answers the natural portrait size, the frame comes out landscape."""
    return RecordingRunner(
        {
            "connect": _done(["adb", "connect"], out=b"connected\n"),
            "wm size": _done(["adb", "wm", "size"], out=b"Physical size: 900x1600\n"),
            "wm density": _done(["adb", "wm", "density"], out=b"Physical density: 240\n"),
            "screencap": _done(["adb", "screencap"], out=frame),
        }
    )


def test_display_check_passes_a_portrait_display_turned_to_landscape() -> None:
    result = checks.display_check(
        "C:/tools/adb.exe", 16384, runner=_portrait_runner(_png(1600, 900))
    )
    assert result.ok is True
    assert (result.width, result.height, result.dpi) == (1600, 900, 240)
    assert "900 x 1600" in result.detail
    assert result.hint == ""


def test_display_check_fails_a_portrait_display_left_portrait() -> None:
    result = checks.display_check(
        "C:/tools/adb.exe", 16384, runner=_portrait_runner(_png(900, 1600))
    )
    assert result.ok is False
    assert (result.width, result.height) == (900, 1600)
    assert result.hint == checks.DISPLAY_HINT


def test_display_check_fails_a_portrait_display_whose_frame_cannot_be_read() -> None:
    result = checks.display_check(
        "C:/tools/adb.exe", 16384, runner=_portrait_runner(b"error: closed")
    )
    assert result.ok is False
    assert "did not return a PNG" in result.detail


def test_display_check_explains_a_mismatch_and_where_to_fix_it() -> None:
    runner = RecordingRunner(
        {
            "connect": _done(["adb", "connect"], out=b"connected\n"),
            "wm size": _done(["adb", "wm", "size"], out=b"Physical size: 1280x720\n"),
            "wm density": _done(["adb", "wm", "density"], out=b"Physical density: 320\n"),
        }
    )
    result = checks.display_check("C:/tools/adb.exe", 5555, runner=runner)
    assert result.ok is False
    assert (result.width, result.height, result.dpi) == (1280, 720, 320)
    assert "1600 x 900 at DPI 240" in result.detail
    assert result.hint == checks.DISPLAY_HINT
    assert "Settings, Display" in result.hint
