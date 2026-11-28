"""ADB calls the setup wizard and the screenshot route need.

Every call is an argv list with shell=False, and every serial is built as
f"127.0.0.1:{int(port)}" from a validated integer port (spec section 9: no user-supplied
string reaches a shell). The subprocess runner is injectable so tests never spawn adb.
"""

from __future__ import annotations

import re
import struct
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from brawlfarm.core import config

# The eight bytes every PNG starts with; adb prints its errors on stdout, so this is how
# we tell a frame from an error message.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

DISPLAY_HINT = (
    "Set the display to 1600 x 900 and pixel density 240 in BlueStacks: Settings, Display, "
    "then restart the instance. In MuMu Player, set the same size and DPI in its display "
    "settings."
)

# `wm size` / `wm density` answer with a Physical line and, when the user set a custom
# resolution, an Override line as well.
_SIZE_RE = re.compile(r"^(Physical|Override) size:\s*(\d+)x(\d+)\s*$", re.MULTILINE)
_DENSITY_RE = re.compile(r"^(Physical|Override) density:\s*(\d+)\s*$", re.MULTILINE)

Runner = Callable[[list[str], float], subprocess.CompletedProcess]


class AdbUnavailable(RuntimeError):
    """adb could not answer for this instance. The message is one line, for the panel."""


def serial(port: int) -> str:
    """The adb serial for a local BlueStacks instance. int() is the validation: a string
    from the user can never widen this into a second argument."""
    return f"127.0.0.1:{int(port)}"


def decode(raw: bytes | str | None) -> str:
    """adb output as text. It is captured as bytes because screencap needs a binary pipe."""
    if raw is None:
        return ""
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)


def first_line(raw: bytes | str | None) -> str:
    """The first non-empty line of an adb message, for a one-line panel detail."""
    for line in decode(raw).splitlines():
        if line.strip():
            return line.strip()
    return ""


def _default_runner(argv: list[str], timeout_s: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        capture_output=True,
        timeout=timeout_s,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def run_adb(
    adb_path: str,
    args: list[str],
    *,
    timeout_s: float = 15.0,
    runner: Runner | None = None,
) -> subprocess.CompletedProcess:
    """Run `<adb_path> <args...>`. An argv list, never a command string."""
    return (runner or _default_runner)([str(adb_path), *args], timeout_s)


def screencap_png(
    adb_path: str, port: int, *, timeout_s: float = 15.0, runner: Runner | None = None
) -> bytes:
    """One PNG frame from the instance, straight out of `adb exec-out screencap -p`.

    Raises AdbUnavailable with a one-line reason when adb errors, returns nothing, or
    returns something that is not a PNG (a closed instance answers with an error on
    stdout, which would otherwise reach the browser as a broken image).
    """
    dev = serial(port)
    try:
        run_adb(adb_path, ["connect", dev], timeout_s=timeout_s, runner=runner)
        done = run_adb(
            adb_path,
            ["-s", dev, "exec-out", "screencap", "-p"],
            timeout_s=timeout_s,
            runner=runner,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdbUnavailable(f"adb timed out taking a screenshot of {dev}") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise AdbUnavailable(f"adb failed for {dev}: {exc}") from exc
    if done.returncode != 0:
        raise AdbUnavailable(f"adb screencap failed for {dev}: {first_line(done.stderr)}")
    data = done.stdout or b""
    if not data:
        raise AdbUnavailable(f"adb screencap returned nothing for {dev}")
    if not data.startswith(PNG_SIGNATURE):
        raise AdbUnavailable(f"adb screencap did not return a PNG for {dev}")
    return data


def png_size(data: bytes) -> tuple[int, int]:
    """Width and height from the PNG's IHDR chunk (bytes 16 to 24, big-endian)."""
    if len(data) < 24 or not data.startswith(PNG_SIGNATURE):
        raise AdbUnavailable("not a PNG")
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def parse_wm_size(text: str) -> tuple[int, int] | None:
    """`adb shell wm size`. An Override line wins: that is what the instance is actually
    running at, and it is what BlueStacks writes when the user picks a custom size."""
    found: dict[str, tuple[int, int]] = {}
    for kind, w, h in _SIZE_RE.findall(text or ""):
        found[kind] = (int(w), int(h))
    return found.get("Override") or found.get("Physical")


def parse_wm_density(text: str) -> int | None:
    """`adb shell wm density`, same Override-wins rule as parse_wm_size."""
    found: dict[str, int] = {}
    for kind, value in _DENSITY_RE.findall(text or ""):
        found[kind] = int(value)
    density = found.get("Override", found.get("Physical"))
    return density


@dataclass(frozen=True)
class DisplayCheck:
    """The wizard's Display step: what adb reported, and what to do when it is wrong."""

    ok: bool
    width: int | None
    height: int | None
    dpi: int | None
    detail: str
    hint: str


def display_check(adb_path: str, port: int, *, runner: Runner | None = None) -> DisplayCheck:
    """Is this instance at 1600 x 900 with DPI 240?

    The controller asserts the resolution at startup and exits otherwise (spec section 9),
    so the wizard checks it first and names the exact BlueStacks screen to change it on.
    """
    dev = serial(port)
    try:
        run_adb(adb_path, ["connect", dev], runner=runner)
        size_out = run_adb(adb_path, ["-s", dev, "shell", "wm", "size"], runner=runner)
        density_out = run_adb(adb_path, ["-s", dev, "shell", "wm", "density"], runner=runner)
    except subprocess.TimeoutExpired:
        return DisplayCheck(
            False, None, None, None, f"adb timed out talking to {dev}", DISPLAY_HINT
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return DisplayCheck(False, None, None, None, f"adb failed for {dev}: {exc}", DISPLAY_HINT)
    size = parse_wm_size(decode(size_out.stdout))
    dpi = parse_wm_density(decode(density_out.stdout))
    if size is None:
        detail = first_line(size_out.stderr) or first_line(size_out.stdout) or "no answer"
        return DisplayCheck(
            False,
            None,
            None,
            dpi,
            f"adb could not read the display of {dev}: {detail}",
            DISPLAY_HINT,
        )
    width, height = size
    if width == config.SCREEN_W and height == config.SCREEN_H and dpi == config.SCREEN_DPI:
        return DisplayCheck(True, width, height, dpi, f"{width} x {height} at DPI {dpi}", "")
    if (width, height) == (config.SCREEN_H, config.SCREEN_W) and dpi == config.SCREEN_DPI:
        # MuMu: portrait by nature, the game turned to landscape. The frame decides.
        try:
            frame = png_size(screencap_png(adb_path, port, runner=runner))
        except AdbUnavailable as exc:
            return DisplayCheck(
                False, width, height, dpi, f"{width} x {height}, and {exc}", DISPLAY_HINT
            )
        if frame == (config.SCREEN_W, config.SCREEN_H):
            fw, fh = frame
            detail = f"{fw} x {fh} at DPI {dpi} (a {width} x {height} display, turned)"
            return DisplayCheck(True, fw, fh, dpi, detail, "")
        width, height = frame
    detail = (
        f"{width} x {height} at DPI {dpi if dpi is not None else '?'}; "
        f"the farm needs {config.SCREEN_W} x {config.SCREEN_H} at DPI {config.SCREEN_DPI}"
    )
    return DisplayCheck(False, width, height, dpi, detail, DISPLAY_HINT)
