"""
Thin wrapper around BlueStacks' bundled adb (HD-Adb.exe).

This is the bot's hands and eyes:
  - connect()      -> make sure adb is talking to the BlueStacks instance
  - screencap()    -> grab the current screen as an OpenCV image (numpy array)
  - tap()          -> tap a single point
  - swipe()        -> swipe / drag between two points (also used for the joystick)

We shell out to adb with subprocess. Everything is addressed to one specific
device via the serial (e.g. "127.0.0.1:5555") so it keeps working even if you
have multiple BlueStacks instances open.
"""

from __future__ import annotations

import re
import struct
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from brawlfarm.core import config


class AdbError(RuntimeError):
    """Raised when an adb command fails."""


# stderr fragments that mean "transient adb hiccup, worth a retry" (vs a real failure).
_TRANSIENT = (
    "offline",
    "closed",
    "device not found",
    "protocol fault",
    "connection reset",
    "no devices",
    "timeout",
)


def _run(args: list[str], *, binary: bool = False, timeout: float = 20.0, retries: int = 2):
    """Run `HD-Adb.exe <args>` and return its output.

    binary=True returns raw bytes (used for screenshots); otherwise decoded text.
    Retries transient failures (a command that times out, or a transient adb-server
    state) with backoff — under multi-instance contention the shared adb server can
    briefly stall, and a single slow command must NOT crash a long farm run. Raises
    AdbError only after exhausting retries (or immediately on a clearly non-transient
    failure).
    """
    cmd = [config.ADB_PATH, *args]
    last_err = None
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last_err = AdbError(f"adb {' '.join(args)} timed out after {timeout}s")
            time.sleep(0.5 * (attempt + 1))
            continue
        if result.returncode == 0:
            return result.stdout if binary else result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        last_err = AdbError(f"adb {' '.join(args)} failed: {stderr or 'unknown error'}")
        if any(s in stderr.lower() for s in _TRANSIENT):
            time.sleep(0.5 * (attempt + 1))
            continue
        raise last_err  # non-transient -> fail immediately
    raise last_err  # exhausted retries on a transient failure


# --- helpers for the play stream (brawlfarm/play/stream.py) -------------------------
# Read-only plumbing: a file onto the device, a port forward, a long-running device
# program. None of these sends input; every argument is a literal or an integer port.


def push(local: Path, remote: str) -> None:
    """Copy one local file to a device path. ``remote`` is chosen by the caller from a
    constant, never from user input."""
    _run(["-s", config.ADB_SERIAL, "push", str(local), remote], timeout=60.0)


def forward(local_port: int, remote: str) -> None:
    """Forward a host TCP port to a device socket (``localabstract:<name>`` or ``tcp:<n>``)."""
    _run(["-s", config.ADB_SERIAL, "forward", f"tcp:{int(local_port)}", remote])


def forward_remove(local_port: int) -> None:
    """Drop the forward for one host port. Raises AdbError when there is none."""
    _run(["-s", config.ADB_SERIAL, "forward", "--remove", f"tcp:{int(local_port)}"])


def shell_process(args: list[str]) -> subprocess.Popen:
    """Start ``adb shell <args>`` and hand back the running process; the caller owns its
    lifetime and must kill it. For device programs that run until stopped, unlike ``_run``,
    which waits for a command to finish."""
    return subprocess.Popen(
        [config.ADB_PATH, "-s", config.ADB_SERIAL, "shell", *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def connect(serial: str | None = None) -> None:
    """Connect adb to a BlueStacks instance (default: the configured one).

    Safe to call repeatedly. Raises AdbError if the device never shows up
    (usually means BlueStacks isn't running, or ADB is off in its settings).

    ``serial`` (r10) lets a multi-instance caller (the control panel's screenshot) target
    another emulator WITHOUT mutating the config.ADB_* globals the worker's own
    capture path reads — omitted, behavior is byte-identical to before.
    """
    serial = serial or config.ADB_SERIAL
    _run(["connect", serial])

    # Verify the device is actually listed and authorized.
    devices = _run(["devices"])
    for line in devices.splitlines():
        if line.startswith(serial):
            state = line.split("\t")[-1].strip() if "\t" in line else ""
            if state == "device":
                return
            raise AdbError(f"Device {serial} is in state '{state}', not 'device'.")
    raise AdbError(
        f"Could not connect to {serial}. Is BlueStacks running, and is "
        f"ADB enabled (Settings -> Advanced -> Android Debug Bridge)?"
    )


# RAW capture self-disable: if the device streams a raw format we can't parse (header
# says it isn't RGBA_8888), retrying every loop would just double the capture cost — so
# we flip this and stay on the PNG path for the rest of the session. Transient failures
# (short read / adb hiccup) do NOT set it; those fall back for one capture only.
_raw_cap_unsupported = False


def _decode_raw_screencap(buf: bytes) -> np.ndarray | None:
    """Decode the output of `exec-out screencap` (no -p): a 16-byte header
    (w, h, format, dataspace as <IIII — 16 bytes on Android 9+, where AOSP added the
    dataspace field) followed by the raw framebuffer. format 1 = RGBA_8888 (what this
    BlueStacks reports). Returns None if the buffer doesn't parse — caller falls back
    to the PNG path."""
    global _raw_cap_unsupported
    if len(buf) < 16:
        return None  # short read — transient, don't disable
    w, h, fmt, _dataspace = struct.unpack("<IIII", buf[:16])
    if fmt != 1 or not (0 < w <= 8192 and 0 < h <= 8192):
        _raw_cap_unsupported = True  # structurally unusable -> stop trying raw
        return None
    if len(buf) < 16 + w * h * 4:
        return None  # truncated transfer — transient
    arr = np.frombuffer(buf, np.uint8, w * h * 4, 16).reshape(h, w, 4)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)


def screencap(serial: str | None = None) -> np.ndarray:
    """Capture the current screen as a BGR image (OpenCV format).

    Fast path (config.RAW_CAP, default on): `exec-out screencap` streams the RAW
    framebuffer — larger transfer, but skips the on-device PNG encode AND the host
    cv2.imdecode (~200 ms faster per capture, measured; see the legacy research note
    "performance optimization", item 4, not ported). Verified byte-identical pixels to
    the PNG path (PNG is lossless). Falls back silently to the PNG path on any
    failure, mirroring the FAST_INPUT pattern.

    PNG path: `exec-out screencap -p` streams a PNG over stdout without the
    newline-translation corruption that the old `shell screencap` had on Windows.

    ``serial`` (r10): explicit device override for multi-instance callers (/ss) —
    omitted, the configured device is captured exactly as before.
    """
    serial = serial or config.ADB_SERIAL
    if config.RAW_CAP and not _raw_cap_unsupported:
        try:
            raw = _run(["-s", serial, "exec-out", "screencap"], binary=True)
        except AdbError:
            raw = b""  # transient adb failure -> try the PNG path this iteration
        if raw:
            img = _decode_raw_screencap(raw)
            if img is not None:
                return img
    png_bytes = _run(["-s", serial, "exec-out", "screencap", "-p"], binary=True)
    if not png_bytes:
        raise AdbError("screencap returned no data.")
    img = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise AdbError("Failed to decode screenshot PNG.")
    return img


# --- Fast input (raw /dev/input writes) --------------------------------------
# `adb shell input tap/swipe` spawns a JVM per call (~700ms + CPU churn -> timeouts under
# load). Instead we write the raw touch events straight to the device node in one shell
# call (~70ms, no JVM). The node is a type-A multitouch, position-only device, so a touch
# frame is: ABS_MT_POSITION_X, ABS_MT_POSITION_Y, SYN_MT_REPORT, SYN_REPORT; lifting the
# finger is an empty MT frame: SYN_MT_REPORT, SYN_REPORT. See config.FAST_INPUT.
_EV_ABS, _EV_SYN = 3, 0
_ABS_MT_X, _ABS_MT_Y = 53, 54
_SYN_REPORT, _SYN_MT_REPORT = 0, 2


def _ev(t: int, c: int, v: int) -> bytes:
    # 64-bit struct input_event: time(8+8), type(H), code(H), value(i) = 24 bytes, LE.
    return struct.pack("<qqHHi", 0, 0, t, c, v)


def _touch_frame(x: int, y: int) -> bytes:
    dx = round(int(x) / config.SCREEN_W * config.TOUCH_MAX_X)
    dy = round(int(y) / config.SCREEN_H * config.TOUCH_MAX_Y)
    return (
        _ev(_EV_ABS, _ABS_MT_X, dx)
        + _ev(_EV_ABS, _ABS_MT_Y, dy)
        + _ev(_EV_SYN, _SYN_MT_REPORT, 0)
        + _ev(_EV_SYN, _SYN_REPORT, 0)
    )


_TOUCH_UP = _ev(_EV_SYN, _SYN_MT_REPORT, 0) + _ev(_EV_SYN, _SYN_REPORT, 0)


def _write_touch(blob: bytes) -> None:
    """Write raw input_event bytes to the touch node in one shell call (octal-escaped
    through printf, so only ASCII crosses adb — no binary-over-shell issues)."""
    esc = "".join("\\%03o" % b for b in blob)
    _run(["-s", config.ADB_SERIAL, "shell", f"printf '{esc}' > {config.TOUCH_DEVICE}"])


def tap(x: int, y: int) -> None:
    """Tap a single screen coordinate (pixels). Uses the fast raw-write path when
    config.FAST_INPUT is on, falling back to `adb shell input tap` on any failure."""
    if config.FAST_INPUT:
        try:
            _write_touch(_touch_frame(x, y) + _TOUCH_UP)
            return
        except AdbError:
            pass  # fall back to the slow-but-sure path
    _run(["-s", config.ADB_SERIAL, "shell", "input", "tap", str(int(x)), str(int(y))])


def _slow_swipe(x1, y1, x2, y2, duration_ms) -> None:
    _run(
        [
            "-s",
            config.ADB_SERIAL,
            "shell",
            "input",
            "swipe",
            str(int(x1)),
            str(int(y1)),
            str(int(x2)),
            str(int(y2)),
            str(int(duration_ms)),
        ]
    )


def swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> None:
    """Swipe / drag from (x1,y1) to (x2,y2) over duration_ms ms (also nudges the in-match
    joystick and scrolls menus). Fast path streams touch frames over the duration so the
    game sees a real moving finger; falls back to `adb shell input swipe` on failure."""
    if config.FAST_INPUT:
        try:
            frames = max(2, min(12, int(duration_ms) // 30))
            _write_touch(_touch_frame(x1, y1))  # finger down
            for i in range(1, frames + 1):
                t = i / frames
                _write_touch(_touch_frame(round(x1 + (x2 - x1) * t), round(y1 + (y2 - y1) * t)))
                time.sleep(duration_ms / 1000 / frames)
            _write_touch(_TOUCH_UP)  # finger up
            return
        except AdbError:
            try:
                _write_touch(_TOUCH_UP)  # don't leave the finger stuck down
            except AdbError:
                pass
    _slow_swipe(x1, y1, x2, y2, duration_ms)


def tap_hold(x: int, y: int, duration_ms: int = 4500) -> None:
    """Press and hold a point (a zero-distance swipe). Used to open Star Drops."""
    swipe(x, y, x, y, duration_ms)


def input_text(text: str) -> None:
    """Type ``text`` into the currently-focused field (Supercell ID email / code
    entry — the login form is rendered in-game, so there's no Android EditText
    shortcut). Uses the plain `adb shell input text` path: onboarding types once,
    so the JVM-per-call cost doesn't matter. Spaces become %s per input's syntax.

    SECURITY: `adb shell` re-evaluates its arguments through the DEVICE-side shell,
    so metacharacters (;, $(), backticks, quotes) in user-supplied text would
    execute there — the injection vector in the legacy planning note "multi-user
    security" (not ported). Whitelist-gate instead of escaping; callers still
    validate semantics (email shape, 6-digit code) on top of this."""
    s = str(text)
    if not re.fullmatch(r"[A-Za-z0-9@._%+\- ]+", s):
        raise ValueError(f"input_text refused {s!r}: characters outside the safe set")
    _run(["-s", config.ADB_SERIAL, "shell", "input", "text", s.replace(" ", "%s")])


def keyevent(code: int) -> None:
    """Send an Android key event (e.g. 3 = HOME, 4 = BACK)."""
    _run(["-s", config.ADB_SERIAL, "shell", "input", "keyevent", str(int(code))])


def go_home() -> None:
    """Press the Android HOME button (used in freeze recovery)."""
    keyevent(3)


def launch_app(package: str | None = None) -> None:
    """Launch (or foreground) an app by package name."""
    pkg = package or config.BS_PACKAGE
    _run(
        [
            "-s",
            config.ADB_SERIAL,
            "shell",
            "monkey",
            "-p",
            pkg,
            "-c",
            "android.intent.category.LAUNCHER",
            "1",
        ]
    )


def force_stop(package: str | None = None) -> None:
    """Force-stop an app by package name (used to fully restart on freeze)."""
    pkg = package or config.BS_PACKAGE
    _run(["-s", config.ADB_SERIAL, "shell", "am", "force-stop", pkg])


def current_package() -> str:
    """Return the package name of the foreground (resumed) activity, or "" if it
    can't be read. Lets the bot tell whether the game is already open vs. sitting on
    the BlueStacks launcher home (e.g. right after a fresh emulator boot)."""
    try:
        out = _run(
            [
                "-s",
                config.ADB_SERIAL,
                "shell",
                "dumpsys activity activities | grep mResumedActivity",
            ]
        )
    except AdbError:
        return ""
    # line looks like: "mResumedActivity: ActivityRecord{hash u0 <package>/.Activity t#}"
    m = re.search(r"u0\s+([\w.]+)/", out)
    return m.group(1) if m else ""


# Package names reach the DEVICE shell (adb shell re-parses its arguments), so gate them
# to the characters a real Android package name can hold before building a command.
_PACKAGE_RE = re.compile(r"[A-Za-z0-9_.]+")


def _checked_package(package: str) -> str:
    s = str(package)
    if not _PACKAGE_RE.fullmatch(s):
        raise ValueError(f"refused package name {s!r}: characters outside the safe set")
    return s


def open_store_page(package: str) -> None:
    """Open ``package``'s Play Store page (game auto-update). The URL is single-quoted
    because the device shell would treat ``?`` as a glob character."""
    pkg = _checked_package(package)
    _run(
        [
            "-s",
            config.ADB_SERIAL,
            "shell",
            "am",
            "start",
            "-a",
            "android.intent.action.VIEW",
            "-d",
            f"'market://details?id={pkg}'",
        ]
    )


def package_version(package: str) -> str | None:
    """Installed ``versionName`` of ``package``, or None when adb fails or the package
    is missing. Used to confirm a Play Store update actually landed."""
    pkg = _checked_package(package)
    try:
        out = _run(["-s", config.ADB_SERIAL, "shell", "dumpsys", "package", pkg])
    except AdbError:
        return None
    m = re.search(r"versionName=(\S+)", out)
    return m.group(1) if m else None


def screen_size() -> tuple[int, int]:
    """Return (width, height) of the BlueStacks display in pixels."""
    out = _run(["-s", config.ADB_SERIAL, "shell", "wm", "size"])
    # Output looks like: "Physical size: 1600x900"
    for token in out.replace("\n", " ").split():
        if "x" in token and token.replace("x", "").isdigit():
            w, h = token.split("x")
            return int(w), int(h)
    raise AdbError(f"Could not parse screen size from: {out!r}")
