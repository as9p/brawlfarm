"""The emulator helpers: finding, sizing, closing and starting HD-Player, the free commit
figure, and the boot check. Every OS call is faked: nothing here may start, signal or kill
a real process, because a live farm can be running on the same machine."""

from __future__ import annotations

import ctypes
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from brawlfarm.supervisor import emulator as E

GB = 1024**3
PLAYER = "HD-Player.exe"


class FakeProc:
    """Stands in for psutil.Process. kill() and children() fail the test outright: the
    helpers may only ever terminate the one PID they were given."""

    def __init__(
        self,
        pid: int,
        name: str | None = PLAYER,
        argv: list[str] | None = None,
        private: int = 0,
        exits: bool = False,
        cmdline_error: Exception | None = None,
        memory_error: Exception | None = None,
        terminate_error: Exception | None = None,
    ) -> None:
        self.pid = pid
        self.info = {"name": name}
        self.argv = argv or []
        self.private = private
        self.exits = exits
        self.cmdline_error = cmdline_error
        self.memory_error = memory_error
        self.terminate_error = terminate_error
        self.calls: list[tuple] = []

    def name(self) -> str | None:
        return self.info["name"]

    def cmdline(self) -> list[str]:
        if self.cmdline_error:
            raise self.cmdline_error
        return self.argv

    def memory_info(self) -> SimpleNamespace:
        if self.memory_error:
            raise self.memory_error
        return SimpleNamespace(rss=1, wset=2, private=self.private)

    def wait(self, timeout: float) -> int:
        self.calls.append(("wait", timeout))
        if self.exits:
            return 0
        raise psutil.TimeoutExpired(timeout, self.pid)

    def terminate(self) -> None:
        self.calls.append(("terminate",))
        if self.terminate_error:
            raise self.terminate_error

    def kill(self) -> None:
        raise AssertionError("kill() is never the way out")

    def children(self, recursive: bool = False) -> list:
        raise AssertionError("the process tree is never touched")


@pytest.fixture()
def table(monkeypatch: pytest.MonkeyPatch) -> dict[int, FakeProc]:
    """The fake process table: fill it in the test. An unknown PID is a gone process."""
    procs: dict[int, FakeProc] = {}

    def process(pid: int) -> FakeProc:
        if pid not in procs:
            raise psutil.NoSuchProcess(pid)
        return procs[pid]

    monkeypatch.setattr(E.psutil, "Process", process)
    monkeypatch.setattr(E.psutil, "process_iter", lambda attrs=None: list(procs.values()))
    monkeypatch.setattr(E, "pid_alive", lambda pid: pid in procs)
    return procs


def add(table: dict[int, FakeProc], *procs: FakeProc) -> None:
    table.update({p.pid: p for p in procs})


def player(pid: int, instance: str, **kw) -> FakeProc:
    return FakeProc(pid, argv=[f"C:\\x\\{PLAYER}", "--instance", instance], **kw)


# find_player


def test_find_player_matches_the_exact_instance(table) -> None:
    add(table, FakeProc(10, name="explorer.exe"), player(4242, "Pie64"))
    assert E.find_player("Pie64") == 4242


def test_find_player_ignores_flags_after_the_instance(table) -> None:
    argv = [PLAYER, "--instance", "Pie64", "--cmd", "launchApp"]
    add(table, FakeProc(4242, argv=argv))
    assert E.find_player("Pie64") == 4242


def test_find_player_never_matches_a_partial_name(table) -> None:
    add(table, player(1, "Pie64_2"))
    assert E.find_player("Pie64") is None
    add(table, player(2, "Pie64"))
    assert E.find_player("Pie64") == 2
    assert E.find_player("Pie64_2") == 1
    assert E.find_player("Pie") is None
    assert E.find_player("pie64") is None


@pytest.mark.parametrize(
    "argv",
    [
        [PLAYER],
        [PLAYER, "--instance"],
        [PLAYER, "Pie64"],
        [PLAYER, "--cmd", "Pie64"],
        [PLAYER, "--instance=Pie64"],
        [PLAYER, "--instance", "other", "Pie64"],
    ],
)
def test_find_player_needs_the_name_right_after_the_flag(table, argv) -> None:
    add(table, FakeProc(1, argv=argv))
    assert E.find_player("Pie64") is None


@pytest.mark.parametrize("name", ["python.exe", "HD-Player.exe.bak", "HD-Adb.exe", None])
def test_find_player_only_looks_at_hd_player(table, name) -> None:
    add(table, FakeProc(1, name=name, argv=[PLAYER, "--instance", "Pie64"]))
    assert E.find_player("Pie64") is None


@pytest.mark.parametrize(
    "error",
    [psutil.AccessDenied(1), psutil.NoSuchProcess(1), psutil.ZombieProcess(1)],
)
def test_find_player_steps_over_a_process_it_cannot_read(table, error) -> None:
    add(table, FakeProc(1, argv=[], cmdline_error=error), player(2, "Pie64"))
    assert E.find_player("Pie64") == 2


def test_find_player_with_nothing_running(table) -> None:
    assert E.find_player("Pie64") is None


# private_gb


def test_private_gb_reads_private_bytes_not_the_working_set(table) -> None:
    add(table, player(1, "Pie64", private=3 * GB))
    assert E.private_gb(1) == 3.0


def test_private_gb_is_none_for_a_gone_process(table) -> None:
    assert E.private_gb(1) is None


@pytest.mark.parametrize("error", [psutil.AccessDenied(1), psutil.NoSuchProcess(1)])
def test_private_gb_is_none_when_the_read_fails(table, error) -> None:
    add(table, player(1, "Pie64", memory_error=error))
    assert E.private_gb(1) is None


def test_private_gb_is_none_for_a_pid_windows_still_lists_after_exit(table, monkeypatch) -> None:
    add(table, player(1, "Pie64", private=GB))
    monkeypatch.setattr(E, "pid_alive", lambda pid: False)
    assert E.private_gb(1) is None


# commit_free_gb


def test_commit_free_gb_converts_bytes() -> None:
    assert E.commit_free_gb(read=lambda: 5 * GB // 2) == 2.5


def test_commit_free_gb_fails_open_when_the_read_fails() -> None:
    """A broken read must never look like commit exhaustion, or it would recycle a healthy
    emulator."""
    assert E.commit_free_gb(read=lambda: None) == math.inf


@pytest.mark.parametrize("error", [OSError("no"), ValueError("no"), RuntimeError("no")])
def test_commit_free_gb_fails_open_when_the_read_raises(error) -> None:
    def read():
        raise error

    assert E.commit_free_gb(read=read) == math.inf


@pytest.mark.skipif(sys.platform != "win32", reason="kernel32")
def test_avail_page_file_takes_the_commit_field_not_physical(monkeypatch) -> None:
    calls: list[int] = []

    def fake_status(ref) -> int:
        status = ref._obj
        calls.append(status.dwLength)
        status.ullAvailPhys = 111
        status.ullAvailPageFile = 222
        status.ullAvailVirtual = 333
        return 1

    fake = SimpleNamespace(GlobalMemoryStatusEx=fake_status)
    monkeypatch.setattr(E, "_kernel32", fake)
    assert E._avail_page_file() == 222
    assert calls == [64]


@pytest.mark.skipif(sys.platform != "win32", reason="kernel32")
def test_avail_page_file_is_none_when_the_call_fails(monkeypatch) -> None:
    monkeypatch.setattr(E, "_kernel32", SimpleNamespace(GlobalMemoryStatusEx=lambda ref: 0))
    assert E._avail_page_file() is None


@pytest.mark.skipif(sys.platform != "win32", reason="kernel32")
def test_commit_free_gb_reads_a_real_number() -> None:
    assert 0 < E.commit_free_gb() < math.inf


# close_player


def test_close_player_stops_at_the_polite_ask_when_the_window_closes(table) -> None:
    add(table, player(1, "Pie64", exits=True))
    asked: list[int] = []
    assert E.close_player(1, 12.5, post_close=asked.append) is True
    assert asked == [1]
    assert table[1].calls == [("wait", 12.5)]


def test_close_player_terminates_only_the_given_pid_after_the_grace(table) -> None:
    add(table, player(1, "Pie64"), player(2, "Pie64_2"), FakeProc(3, name="BstkSVC.exe"))
    asked: list[int] = []
    assert E.close_player(1, 7.0, post_close=asked.append) is True
    assert asked == [1]
    assert table[1].calls == [("wait", 7.0), ("terminate",)]
    assert table[2].calls == []
    assert table[3].calls == []


def test_close_player_waits_no_longer_than_the_grace(table) -> None:
    add(table, player(1, "Pie64"))
    E.close_player(1, 3.0, post_close=lambda pid: None)
    assert [c[1] for c in table[1].calls if c[0] == "wait"] == [3.0]


def test_close_player_refuses_a_pid_that_is_not_the_player(table) -> None:
    add(table, FakeProc(1, name="notepad.exe"))
    asked: list[int] = []
    lines: list[str] = []
    assert E.close_player(1, 1.0, post_close=asked.append, log=lines.append) is False
    assert asked == []
    assert table[1].calls == []
    assert "refused" in lines[0]


def test_close_player_on_a_gone_pid_has_nothing_to_do(table) -> None:
    asked: list[int] = []
    assert E.close_player(1, 1.0, post_close=asked.append) is True
    assert asked == []


def test_close_player_counts_an_exit_during_terminate_as_closed(table) -> None:
    add(table, player(1, "Pie64", terminate_error=psutil.NoSuchProcess(1)))
    assert E.close_player(1, 0.0, post_close=lambda pid: None) is True


def test_close_player_reports_a_terminate_it_could_not_deliver(table) -> None:
    add(table, player(1, "Pie64", terminate_error=psutil.AccessDenied(1)))
    lines: list[str] = []
    assert E.close_player(1, 0.0, post_close=lambda pid: None, log=lines.append) is False
    assert "failed" in lines[-1]


@pytest.mark.parametrize("error", [OSError("no"), ctypes.ArgumentError("no"), ValueError("no")])
def test_close_player_reports_a_polite_ask_that_raised(table, error) -> None:
    add(table, player(1, "Pie64"))
    lines: list[str] = []

    def post_close(pid):
        raise error

    assert E.close_player(1, 0.0, post_close=post_close, log=lines.append) is False
    assert "failed" in lines[-1]


@pytest.mark.skipif(sys.platform != "win32", reason="user32")
def test_post_wm_close_asks_only_the_visible_windows_of_that_pid(monkeypatch) -> None:
    """Three windows: ours visible, ours hidden, another process's. Faked outright, so no
    real window can ever receive a WM_CLOSE from this test."""
    owners = {101: 42, 102: 42, 103: 7}
    posted: list[tuple] = []

    def get_owner(hwnd, ref) -> int:
        ref._obj.value = owners[hwnd]
        return 1

    def post(hwnd, msg, wparam, lparam) -> int:
        posted.append((hwnd, msg))
        return 1

    def enum(callback, lparam) -> int:
        for hwnd in owners:
            callback(hwnd, lparam)
        return 1

    fake = SimpleNamespace(
        EnumWindows=enum,
        GetWindowThreadProcessId=get_owner,
        IsWindowVisible=lambda hwnd: hwnd != 102,
        PostMessageW=post,
    )
    monkeypatch.setattr(E, "_user32", fake)
    assert E._post_wm_close(42) == 1
    assert posted == [(101, 0x0010)]


# open_player


class FakePopen:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.argv: list[str] | None = None
        self.kwargs: dict = {}

    def __call__(self, argv: list[str], **kwargs) -> SimpleNamespace:
        if self.error:
            raise self.error
        self.argv = argv
        self.kwargs = kwargs
        return SimpleNamespace(pid=555)


def test_open_player_starts_it_with_the_instance_flag() -> None:
    popen = FakePopen()
    path = r"C:\Program Files\BlueStacks_nxt\HD-Player.exe"
    assert E.open_player(path, "Pie64", popen=popen) == 555
    assert popen.argv == [path, "--instance", "Pie64"]
    assert popen.kwargs.get("shell", False) is False
    assert popen.kwargs["cwd"] == str(Path(path).parent)


def test_open_player_detaches_so_it_outlives_the_supervisor() -> None:
    popen = FakePopen()
    E.open_player("HD-Player.exe", "Pie64", popen=popen)
    kw = popen.kwargs
    assert kw["stdin"] == kw["stdout"] == kw["stderr"] == subprocess.DEVNULL
    if sys.platform == "win32":
        assert kw["creationflags"] & subprocess.DETACHED_PROCESS
        assert kw["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP


def test_open_player_reports_a_start_that_failed() -> None:
    lines: list[str] = []
    popen = FakePopen(FileNotFoundError("no such file"))
    assert E.open_player("missing.exe", "Pie64", popen=popen, log=lines.append) is None
    assert "failed" in lines[0]


# boot_done


@pytest.fixture()
def adb(tmp_path: Path) -> str:
    path = tmp_path / "adb.exe"
    path.write_bytes(b"")
    return str(path)


def scripted(answer: str) -> tuple[list[list[str]], object]:
    seen: list[list[str]] = []

    def runner(cmd: list[str], timeout_s: float) -> str:
        seen.append(cmd)
        return answer if "getprop" in cmd else "connected"

    return seen, runner


def test_boot_done_is_one_connect_and_one_getprop(adb) -> None:
    seen, runner = scripted("1\r\n")
    assert E.boot_done(adb, 5555, runner=runner) is True
    assert seen == [
        [adb, "connect", "127.0.0.1:5555"],
        [adb, "-s", "127.0.0.1:5555", "shell", "getprop", "sys.boot_completed"],
    ]


@pytest.mark.parametrize("answer", ["", "0", "\n", "10", "error: device offline"])
def test_boot_done_is_false_until_the_prop_says_one(adb, answer) -> None:
    seen, runner = scripted(answer)
    assert E.boot_done(adb, 5555, runner=runner) is False
    assert len(seen) == 2


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired("adb", 5),
        subprocess.SubprocessError("adb"),
        OSError("adb"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"),
    ],
)
def test_boot_done_is_false_on_adb_failure(adb, error) -> None:
    def runner(cmd: list[str], timeout_s: float) -> str:
        raise error

    assert E.boot_done(adb, 5555, runner=runner) is False


def test_boot_done_is_false_when_adb_is_missing(tmp_path) -> None:
    seen, runner = scripted("1")
    assert E.boot_done(str(tmp_path / "nope.exe"), 5555, runner=runner) is False
    assert seen == []


def test_boot_done_passes_a_short_timeout(adb) -> None:
    timeouts: list[float] = []

    def runner(cmd: list[str], timeout_s: float) -> str:
        timeouts.append(timeout_s)
        return "1"

    E.boot_done(adb, 5555, runner=runner)
    assert timeouts and max(timeouts) <= 10
