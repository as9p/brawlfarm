"""OS helpers for restarting BlueStacks: find one instance's HD-Player, size it, read the
free commit charge, close and reopen it, and check that Android finished booting.

Safety rail: HD-Player is only ever found by the exact --instance value on its command
line, and close_player() refuses a PID that is not HD-Player, so a reused PID or a sibling
instance can never be closed by mistake. Only the given PID is signalled, never its tree:
the BlueStacks services are shared with every other instance.

Each helper does one thing and returns. boot_done() never loops and nothing here sleeps
past the grace it was given, because the caller is the supervisor's 60 s tick.
"""

from __future__ import annotations

import ctypes
import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import psutil

from brawlfarm.supervisor.process import Runner, _default_runner, pid_alive

PLAYER_NAME = "HD-Player.exe"
INSTANCE_FLAG = "--instance"
BOOT_PROP = "sys.boot_completed"

_GB = 1024**3  # binary, the same unit as PowerShell's 1GB
_WM_CLOSE = 0x0010
# DETACHED_PROCESS rather than the CREATE_NO_WINDOW launch_worker uses: Windows ignores
# CREATE_NO_WINDOW alongside it, and this child has to keep no tie to the supervisor's
# console. Neither shows a console window.
_DETACHED_FLAGS = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
)


class _MemoryStatusEx(ctypes.Structure):
    """MEMORYSTATUSEX. The "PageFile" fields are misnamed: they are the system commit limit
    and what is left of it, not the size of pagefile.sys."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


if sys.platform == "win32":
    # The same plain ctypes types as process.py, on this module's own library objects so
    # the argtypes set here stay out of its handle.
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
    _kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    _user32.EnumWindows.restype = ctypes.c_int
    _user32.EnumWindows.argtypes = [_WNDENUMPROC, ctypes.c_void_p]
    _user32.GetWindowThreadProcessId.restype = ctypes.c_ulong
    _user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    _user32.IsWindowVisible.restype = ctypes.c_int
    _user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
    _user32.PostMessageW.restype = ctypes.c_int
    _user32.PostMessageW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_ssize_t,
    ]
else:  # pragma: no cover - the supervisor only runs on Windows
    _kernel32 = None
    _user32 = None


def find_player(instance: str) -> int | None:
    """The PID of the HD-Player running exactly this instance, or None. The value after
    --instance is compared whole, so Pie64 never matches Pie64_2. A process that denies
    access or vanishes mid-scan is stepped over."""
    for proc in psutil.process_iter(["name"]):
        if (proc.info.get("name") or "").lower() != PLAYER_NAME.lower():
            continue
        try:
            argv = proc.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            continue
        if (INSTANCE_FLAG, instance) in zip(argv, argv[1:], strict=False):
            return proc.pid
    return None


def private_gb(pid: int) -> float | None:
    """Committed private memory in GB, or None when the process is gone or unreadable.

    psutil's ``private`` is PrivateUsage from GetProcessMemoryInfo. PowerShell's
    PagedMemorySize64 reports the same number (Windows fills PagefileUsage from it), and
    the working set (rss) is the wrong one: it hides whatever was trimmed to disk. The
    pid_alive check covers the moment Windows still lists a PID that has already exited.
    """
    if not pid_alive(pid):
        return None
    try:
        return psutil.Process(pid).memory_info().private / _GB
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return None


def _avail_page_file() -> int | None:
    """Bytes of commit charge left system-wide (ullAvailPageFile), or None on failure."""
    if _kernel32 is None:
        return None
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not _kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullAvailPageFile


def commit_free_gb(read: Callable[[], int | None] = _avail_page_file) -> float:
    """Free commit charge in GB, the same number as Win32_OperatingSystem.FreeVirtualMemory.
    This is the figure that runs out: free physical memory stayed near 5.6 GB while commit
    was exhausted, so psutil.virtual_memory() would never have seen it. Fails OPEN (inf) on
    a failed read, so a broken probe can never restart a healthy emulator."""
    try:
        available = read()
    except Exception:
        return math.inf
    return math.inf if available is None else available / _GB


def _post_wm_close(pid: int) -> int:
    """Ask each visible top-level window of ``pid`` to close; returns how many were asked.
    PostMessage, not SendMessage: it queues and returns, so a hung window cannot block the
    tick. Hidden helper windows are left alone."""
    if _user32 is None:
        return 0
    asked = 0

    def visit(hwnd: int, _lparam: int) -> int:
        nonlocal asked
        owner = ctypes.c_ulong()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if (
            owner.value == pid
            and _user32.IsWindowVisible(hwnd)
            and _user32.PostMessageW(hwnd, _WM_CLOSE, 0, 0)
        ):
            asked += 1
        return 1  # keep enumerating

    _user32.EnumWindows(_WNDENUMPROC(visit), None)
    return asked


def close_player(
    pid: int,
    grace_s: float,
    post_close: Callable[[int], object] = _post_wm_close,
    log: Callable[[str], None] = print,
) -> bool:
    """Ask the window to close, wait up to ``grace_s``, then terminate this one PID.

    True when the process is gone or the terminate was delivered. Terminating does not wait
    for the exit, so the whole call stays within the grace: the caller confirms with
    find_player() on its next tick. False (and logs why) when the PID is not HD-Player or
    could not be signalled. A PID already gone is True: nothing is left to close.
    """
    try:
        proc = psutil.Process(pid)
        if (proc.name() or "").lower() != PLAYER_NAME.lower():
            log(f"close {pid}: refused, process is not {PLAYER_NAME}")
            return False
        post_close(pid)
        try:
            proc.wait(timeout=grace_s)
            return True
        except psutil.TimeoutExpired:
            log(f"close {pid}: still running after {grace_s}s, terminating")
        proc.terminate()  # this Process object refuses a PID that was reused meanwhile
    except psutil.NoSuchProcess:
        return True
    except (psutil.Error, OSError, ValueError, ctypes.ArgumentError) as exc:
        log(f"close {pid}: failed: {exc}")
        return False
    return True


def open_player(
    path: str,
    instance: str,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    log: Callable[[str], None] = print,
) -> int | None:
    """Start HD-Player for ``instance``, detached so it outlives the supervisor, with no
    console window and no handles of ours. Returns the PID, or None (and logs why) when it
    could not start, say a wrong emulator_path."""
    try:
        proc = popen(
            [path, INSTANCE_FLAG, instance],
            cwd=str(Path(path).parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_DETACHED_FLAGS,
        )
    except (OSError, ValueError) as exc:
        log(f"open {instance}: failed: {exc}")
        return None
    return proc.pid


def boot_done(
    adb_path: str, port: int, timeout_s: float = 5.0, runner: Runner = _default_runner
) -> bool:
    """One check: adb connect, then getprop sys.boot_completed. True only on "1". Unlike
    instance_online() this fails CLOSED: a missing adb, a timeout or any other probe error
    answers False, so a restart is never declared finished on a broken probe. Worst case
    is two ``timeout_s`` calls; it never loops, the caller's tick does."""
    if not Path(adb_path).exists():
        return False
    serial = f"127.0.0.1:{int(port)}"
    try:
        runner([adb_path, "connect", serial], timeout_s)
        out = runner([adb_path, "-s", serial, "shell", "getprop", BOOT_PROP], timeout_s)
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    return out.strip() == "1"
