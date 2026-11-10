"""Fakes and builders for the API tests: a supervisor whose clock, adb probe, launcher,
liveness check, killer, sleep and events refresh are all in memory, an AppSettings
builder, and a TestClient whose lifespan has already run.

Mirrors the World fake in tests/test_supervisor_loop.py. Nothing here starts a process,
opens a socket or talks to adb; player tags are invented from the game's alphabet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from brawlfarm import settings as S
from brawlfarm.api.app import create_app
from brawlfarm.core import scheduler
from brawlfarm.supervisor import Supervisor

T0 = datetime(2026, 9, 10, 12, 0, 0)
DEFAULT_PORTS = (5555, 5565, 5575)
LOOPBACK = ("127.0.0.1", 50000)  # what TestClient reports as request.client
PANEL_URL = "http://127.0.0.1:8765"  # the Host header the panel really gets


@dataclass
class FakeProc:
    """Stands in for subprocess.Popen: the supervisor only ever reads .pid and .poll()."""

    pid: int
    exited: bool = False

    def poll(self):
        return 0 if self.exited else None


@dataclass
class FakeWorld:
    """The supervisor's injectable seams. Set `online[port] = False` to make the adb probe
    miss, put a pid in `alive` to make it look like a running worker."""

    now: datetime = T0
    online: dict[int, bool] = field(default_factory=dict)
    alive: set[int] = field(default_factory=set)
    launches: list[tuple[list[str], dict[str, str], str]] = field(default_factory=list)
    kills: list[int] = field(default_factory=list)
    procs: list[FakeProc] = field(default_factory=list)
    refreshes: int = 0
    next_pid: int = 100

    def clock(self) -> datetime:
        return self.now

    def probe(self, adb_path, port, timeout_s=15.0) -> bool:
        return self.online.get(port, True)

    def launcher(self, args, env, log_dir, name, module=None) -> FakeProc:
        self.launches.append((list(args), dict(env), name))
        proc = FakeProc(self.next_pid)
        self.next_pid += 1
        self.alive.add(proc.pid)
        self.procs.append(proc)
        return proc

    def is_alive(self, pid) -> bool:
        return pid in self.alive

    def kill(self, pid, log=print) -> bool:
        if pid in self.alive:
            self.alive.discard(pid)
            self.kills.append(pid)
            return True
        return False

    def refresh(self, now=None) -> int:
        self.refreshes += 1
        return 0


def build_settings(
    names: tuple[str, ...] = ("alpha",),
    ports: tuple[int, ...] | None = None,
    **overrides: object,
) -> S.AppSettings:
    """AppSettings with one instance per name; ports default to 5555, 5565, 5575 in order.

    Overrides are dotted section paths, e.g. build_settings(**{"app.port": 9000}) or
    build_settings(**{"connection.adb_path": str(fake_adb)}).
    """
    if ports is None:
        ports = DEFAULT_PORTS[: len(names)]
    if len(ports) != len(names):
        raise ValueError("ports must line up with names")
    s = S.AppSettings(
        instances=[S.InstanceSettings(name=n, adb_port=p) for n, p in zip(names, ports)]
    )
    for dotted, value in overrides.items():
        section, _, field_name = dotted.partition(".")
        if not field_name:
            raise ValueError(f"override {dotted!r} must be section.field")
        setattr(getattr(s, section), field_name, value)
    return s


def make_client(
    tmp_path: Path,
    names: tuple[str, ...] = ("alpha",),
    *,
    world: FakeWorld | None = None,
    **settings_overrides: object,
) -> tuple[TestClient, Supervisor, Path]:
    """A TestClient with the lifespan already entered, plus its supervisor and home dir.

    The caller closes it -- `client.__exit__(None, None, None)` runs the lifespan shutdown,
    which `client.close()` does not -- so every API test module wraps this in a fixture:

        @pytest.fixture()
        def api(tmp_path):
            client, sup, home = make_client(tmp_path, ("alpha", "bravo"))
            try:
                yield client, sup, home
            finally:
                client.__exit__(None, None, None)

    The schedule is switched off for every instance so the tick's desired state is always
    "run" and never depends on the wall clock (the same trick tests/test_supervisor_loop.py
    uses). Turn it back on inside a test with scheduler.set_enabled([name], True).
    """
    world = world or FakeWorld()
    settings = build_settings(names, **settings_overrides)
    home = Path(tmp_path)
    S.save(settings, home)
    sup = Supervisor(
        settings,
        home,
        clock=world.clock,
        probe=world.probe,
        launcher=world.launcher,
        alive=world.is_alive,
        killer=world.kill,
        sleep=lambda _s: None,
        events_refresh=world.refresh,
    )
    scheduler.set_enabled(list(names), False)
    client = TestClient(create_app(sup, home), base_url=PANEL_URL, client=LOOPBACK)
    client.__enter__()  # runs the lifespan, so the startup tick has filled sup.views()
    return client, sup, home
