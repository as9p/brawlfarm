"""The supervisor tick against a temp home with fake process plumbing: launches one
worker per tick with the settings env, stops gracefully then kills after 110 s, honours
boot grace and offline backoff, and exposes start / stop / stop-now / restart / retry."""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from brawlfarm import settings as S
from brawlfarm.core import config, scheduler, status
from brawlfarm.supervisor import Health, InstanceState, Supervisor
from brawlfarm.supervisor import loop as L
from brawlfarm.supervisor.state import InstanceView

T0 = datetime(2026, 9, 10, 12, 0, 0)


@dataclass
class FakeProc:
    pid: int
    exited: bool = False

    def poll(self):
        return 0 if self.exited else None


@dataclass
class World:
    now: datetime = T0
    online: dict[int, bool] = field(default_factory=dict)
    alive: set[int] = field(default_factory=set)
    refuse: set[int] = field(default_factory=set)
    kill_attempts: list[int] = field(default_factory=list)
    launches: list[tuple[list[str], dict[str, str], str]] = field(default_factory=list)
    kills: list[int] = field(default_factory=list)
    procs: list[FakeProc] = field(default_factory=list)
    refreshes: int = 0
    pings: int = 0
    next_pid: int = 100
    probe_delay_s: float = 0.0
    players: dict[str, int] = field(default_factory=dict)  # instance -> HD-Player PID
    player_gb: float = 1.0
    commit_gb: float = 100.0
    close_sticks: bool = False  # the close is delivered and HD-Player stays anyway
    close_ok: bool = True
    open_ok: bool = True
    booted: bool = True
    closes: list[tuple[int, float]] = field(default_factory=list)
    opens: list[tuple[str, str]] = field(default_factory=list)
    boot_checks: list[int] = field(default_factory=list)

    def clock(self):
        return self.now

    def probe(self, adb_path, port, timeout_s=15.0):
        if self.probe_delay_s:  # widen the real probe's subprocess window for race tests
            time.sleep(self.probe_delay_s)
        return self.online.get(port, True)

    def launcher(self, args, env, log_dir, name, module=None):
        self.launches.append((list(args), dict(env), name))
        proc = FakeProc(self.next_pid)
        self.next_pid += 1
        self.alive.add(proc.pid)
        self.procs.append(proc)
        return proc

    def is_alive(self, pid):
        return pid in self.alive

    def kill(self, pid, log=print):
        self.kill_attempts.append(pid)
        if pid in self.refuse:  # e.g. AccessDenied: the process stays alive
            return False
        if pid in self.alive:
            self.alive.discard(pid)
            self.kills.append(pid)
            return True
        return False

    def refresh(self, now=None):
        self.refreshes += 1
        return 0

    def ping(self) -> bool:
        self.pings += 1
        return True

    def find_player(self, instance):
        return self.players.get(instance)

    def private_gb(self, pid):
        return self.player_gb

    def commit_free_gb(self):
        return self.commit_gb

    def close_player(self, pid, grace_s, log=print):
        self.closes.append((pid, grace_s))
        if self.close_ok and not self.close_sticks:
            self.players = {n: p for n, p in self.players.items() if p != pid}
        return self.close_ok

    def open_player(self, path, instance, log=print):
        self.opens.append((path, instance))
        if not self.open_ok:
            return None
        self.players[instance] = 9000 + len(self.opens)
        return self.players[instance]

    def boot_done(self, adb_path, port):
        self.boot_checks.append(port)
        return self.booted


_PORTS = {"Pie64": 5555, "Rome64": 5565}
_TAGS = {"Pie64": "#2P0YLQ9", "Rome64": ""}


def _settings(names=("Pie64", "Rome64")) -> S.AppSettings:
    s = S.AppSettings(
        instances=[
            S.InstanceSettings(name=n, adb_port=_PORTS[n], player_tag=_TAGS[n]) for n in names
        ]
    )
    s.connection.brawl_api_token = "tok"  # the events refresh only runs with a token
    return s


def make_sup(tmp_path: Path, world: World, names=("Pie64", "Rome64")) -> Supervisor:
    s = _settings(names)
    S.save(s, tmp_path)
    sup = Supervisor(
        s,
        tmp_path,
        clock=world.clock,
        probe=world.probe,
        launcher=world.launcher,
        alive=world.is_alive,
        killer=world.kill,
        sleep=lambda _s: None,
        events_refresh=world.refresh,
        pinger=world.ping,
        find_player=world.find_player,
        private_gb=world.private_gb,
        commit_free_gb=world.commit_free_gb,
        close_player=world.close_player,
        open_player=world.open_player,
        boot_done=world.boot_done,
    )
    # Schedule off = legacy always-run, so tests control "desired" through overrides.
    scheduler.set_enabled(list(names), False)
    return sup


@pytest.fixture()
def world() -> World:
    return World()


@pytest.fixture()
def sup(tmp_path: Path, world: World) -> Supervisor:
    return make_sup(tmp_path, world)


def _heartbeat(tmp_path: Path, name: str, pid: int, now: datetime, age_s: float = 5, **fields):
    d = S.instance_dir(tmp_path, name)
    d.mkdir(parents=True, exist_ok=True)
    ts = (now - timedelta(seconds=age_s)).strftime("%Y-%m-%dT%H:%M:%S")
    status.write_status(d, {"pid": pid, "phase": "playing", "games_played": 3, **fields})
    # write_status stamps "now"; rewrite ts to the age we want
    p = d / "status.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["ts"] = ts
    p.write_text(json.dumps(data), encoding="utf-8")


def _view(views, name):
    return next(v for v in views if v.name == name)


def test_init_points_the_core_at_home(sup: Supervisor, tmp_path: Path) -> None:
    from brawlfarm.core import config

    assert config.HOME_DIR == tmp_path.resolve()
    assert set(config.INSTANCES) == {"Pie64", "Rome64"}
    assert config.INSTANCES["Pie64"]["data"] == "instances/Pie64"


def test_first_tick_launches_one_worker_and_defers_the_rest(sup, world, tmp_path) -> None:
    views = sup.tick()
    assert len(world.launches) == 1
    args, env, name = world.launches[0]
    assert name == "Pie64"
    assert args == ["--select-brawler", "--dnd"]
    assert env["BRAWL_ADB_PORT"] == "5555"
    assert env["BRAWL_DATA_DIR"] == "instances/Pie64"
    assert env["BRAWLFARM_HOME"] == str(tmp_path.resolve())
    assert "SystemRoot" in env or "PATH" in env  # inherits the parent environment
    assert _view(views, "Pie64").state == InstanceState.STARTING
    assert _view(views, "Rome64").state == InstanceState.STARTING
    assert _view(views, "Rome64").note == "Waiting for its turn to start"
    assert world.refreshes == 1
    world.now += timedelta(seconds=60)
    sup.tick()
    assert [n for _, _, n in world.launches] == ["Pie64", "Rome64"]


def test_boot_grace_prevents_double_launch(sup, world) -> None:
    sup.tick()
    world.now += timedelta(seconds=60)
    sup.tick()
    world.now += timedelta(seconds=60)  # 120 s after Pie64's launch, still no heartbeat
    views = sup.tick()
    assert [n for _, _, n in world.launches] == ["Pie64", "Rome64"]
    assert _view(views, "Pie64").state == InstanceState.STARTING
    world.now += timedelta(seconds=120)  # 240 s: grace over, still dead -> relaunch
    sup.tick()
    assert [n for _, _, n in world.launches] == ["Pie64", "Rome64", "Pie64"]


def test_healthy_worker_is_farming_and_left_alone(sup, world, tmp_path) -> None:
    world.alive.add(4242)
    _heartbeat(tmp_path, "Pie64", 4242, world.now, farm_brawler="Shelly")
    views = sup.tick()
    v = _view(views, "Pie64")
    assert v.state == InstanceState.FARMING
    assert v.health == Health.HEALTHY
    assert v.pid == 4242
    assert v.games_played == 3
    assert v.farm_brawler == "Shelly"
    assert [n for _, _, n in world.launches] == ["Rome64"]


def test_stale_worker_is_killed_then_relaunched(sup, world, tmp_path) -> None:
    world.alive.add(4242)
    _heartbeat(tmp_path, "Pie64", 4242, world.now, age_s=500)
    sup.tick()
    assert world.kills == [4242]
    assert [n for _, _, n in world.launches] == ["Pie64"]


def test_failed_kill_never_double_launches(sup, world, tmp_path) -> None:
    world.alive.add(4242)
    world.refuse.add(4242)  # the killer reports failure and the process stays alive
    _heartbeat(tmp_path, "Pie64", 4242, world.now, age_s=500)
    views = sup.tick()
    assert world.kill_attempts == [4242]
    assert [n for _, _, n in world.launches] == ["Rome64"]  # Pie64 was NOT relaunched
    v = _view(views, "Pie64")
    assert v.pid == 4242
    assert "could not be stopped" in v.note


def test_malformed_pid_does_not_strand_the_fleet(sup, world, tmp_path) -> None:
    d = S.instance_dir(tmp_path, "Pie64")
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text('{"ts": "2026-09-10T11:59:55", "pid": "n/a"}', encoding="utf-8")
    views = sup.tick()
    assert _view(views, "Pie64").pid is None
    assert [n for _, _, n in world.launches] == ["Pie64"]  # treated as dead and launched
    assert _view(views, "Rome64").state == InstanceState.STARTING  # Rome64 still processed


def test_hung_worker_is_killed_before_relaunch(tmp_path, world) -> None:
    sup = make_sup(tmp_path, world, ("Pie64",))
    sup.tick()  # launches a worker that never writes status.json
    first = world.procs[0]
    world.now += timedelta(seconds=240)  # boot grace over, still no heartbeat, still running
    views = sup.tick()
    assert world.kills == [first.pid]
    assert [n for _, _, n in world.launches] == ["Pie64", "Pie64"]
    assert _view(views, "Pie64").state == InstanceState.STARTING


def test_graceful_stop_then_kill_after_escalation(sup, world, tmp_path) -> None:
    world.alive.add(4242)
    _heartbeat(tmp_path, "Pie64", 4242, world.now)
    sup.stop("Pie64")
    flag = S.instance_dir(tmp_path, "Pie64") / "stop.flag"
    assert flag.exists()
    views = sup.tick()
    v = _view(views, "Pie64")
    assert v.state == InstanceState.STOPPING
    assert v.until == world.now + timedelta(seconds=110)
    assert world.kills == []
    world.now += timedelta(seconds=60)
    _heartbeat(tmp_path, "Pie64", 4242, world.now)
    sup.tick()
    assert world.kills == []
    world.now += timedelta(seconds=60)
    _heartbeat(tmp_path, "Pie64", 4242, world.now)
    views = sup.tick()
    assert world.kills == [4242]
    assert _view(views, "Pie64").state == InstanceState.STOPPED  # killed this tick
    assert _view(views, "Pie64").pid is None
    world.now += timedelta(seconds=60)
    views = sup.tick()
    assert _view(views, "Pie64").state == InstanceState.STOPPED
    assert [n for _, _, n in world.launches] == ["Rome64"]  # never relaunched while stopped


def test_undo_stop_is_start(sup, world, tmp_path) -> None:
    world.alive.add(4242)
    _heartbeat(tmp_path, "Pie64", 4242, world.now)
    sup.stop("Pie64")
    sup.start("Pie64")
    assert not (S.instance_dir(tmp_path, "Pie64") / "stop.flag").exists()
    views = sup.tick()
    assert _view(views, "Pie64").state == InstanceState.FARMING
    assert world.kills == []


def test_stop_now_only_while_pending(sup, world, tmp_path) -> None:
    world.alive.add(4242)
    _heartbeat(tmp_path, "Pie64", 4242, world.now)
    assert sup.stop_now("Pie64") is False
    assert world.kills == []
    sup.stop("Pie64")
    assert sup.stop_now("Pie64") is True
    assert world.kills == [4242]


def test_restart_stops_then_relaunches(sup, world, tmp_path) -> None:
    world.alive.add(4242)
    _heartbeat(tmp_path, "Pie64", 4242, world.now)
    sup.tick()
    sup.restart("Pie64")
    flag = S.instance_dir(tmp_path, "Pie64") / "stop.flag"
    assert flag.exists()
    views = sup.tick()
    assert _view(views, "Pie64").state == InstanceState.STOPPING
    world.alive.discard(4242)  # the worker exited at the menu
    world.now += timedelta(seconds=60)
    views = sup.tick()
    assert [n for _, _, n in world.launches] == ["Rome64", "Pie64"]
    assert _view(views, "Pie64").state == InstanceState.STARTING


def test_offline_backoff_and_alert(tmp_path, world, monkeypatch) -> None:
    sup = make_sup(tmp_path, world, ("Pie64",))  # one instance, so nothing else launches
    alerts: list[tuple[str, dict]] = []
    monkeypatch.setattr(L.notify, "maybe_alert", lambda kind, fields: alerts.append((kind, fields)))
    world.online[5555] = False
    views = sup.tick()
    v = _view(views, "Pie64")
    assert v.state == InstanceState.OFFLINE
    assert v.until == world.now + timedelta(minutes=2)
    assert v.note == "BlueStacks window not found. Retrying in 2 min."
    assert world.launches == []
    world.now += timedelta(minutes=1)
    sup.tick()  # still inside the 2 min window: no probe, no launch
    assert world.launches == []
    world.now += timedelta(minutes=1)
    sup.tick()  # miss 2 -> 4 min
    world.now += timedelta(minutes=4)
    sup.tick()  # miss 3 -> 8 min, alert
    assert alerts == [("offline", {"instance": "Pie64", "misses": 3})]
    assert world.launches == []
    sup.retry_now("Pie64")
    world.online[5555] = True
    views = sup.tick()
    assert [n for _, _, n in world.launches] == ["Pie64"]
    assert _view(views, "Pie64").state == InstanceState.STARTING


def test_events_refresh_only_with_a_token(tmp_path, world) -> None:
    sup = make_sup(tmp_path, world, ("Pie64",))
    sup.settings.connection.brawl_api_token = ""
    sup.apply_settings(sup.settings)
    sup.tick()
    assert world.refreshes == 0


def test_scheduled_break_and_run_for_hours(sup, world, tmp_path) -> None:
    scheduler.write_override("Pie64", "stop", world.now + timedelta(hours=2))
    views = sup.tick()
    assert _view(views, "Pie64").state == InstanceState.STOPPED
    scheduler.set_enabled(["Pie64"], True)  # caps come from the scheduler, so turn it on
    sup.start("Pie64", hours=1.5)
    views = sup.tick()
    assert _view(views, "Pie64").state == InstanceState.STARTING
    assert world.launches[-1][0] == ["--select-brawler", "--dnd", "--max-minutes", "90"]


def test_state_change_callbacks(sup, world, tmp_path) -> None:
    seen: list[tuple[str, str]] = []
    sup.subscribe(lambda v: seen.append((v.name, v.state)))
    sup.tick()
    world.now += timedelta(seconds=60)
    sup.tick()
    assert seen == [("Pie64", "starting"), ("Rome64", "starting")]
    world.alive.add(7)
    _heartbeat(tmp_path, "Pie64", 7, world.now)
    sup.tick()
    assert seen[-1] == ("Pie64", "farming")


def test_desired_stop_from_scheduler_writes_flag_without_user_action(sup, world, tmp_path) -> None:
    world.alive.add(4242)
    _heartbeat(tmp_path, "Pie64", 4242, world.now)
    scheduler.write_override("Pie64", "stop", world.now + timedelta(hours=1))
    views = sup.tick()
    assert (S.instance_dir(tmp_path, "Pie64") / "stop.flag").exists()
    assert _view(views, "Pie64").state == InstanceState.STOPPING


@pytest.mark.asyncio
async def test_run_forever_ticks_and_shuts_down(sup, world) -> None:
    import asyncio

    sup.interval_s = 0.05
    task = asyncio.create_task(sup.run_forever())
    await asyncio.sleep(0.2)
    sup.poke()
    await asyncio.sleep(0.05)
    sup.request_shutdown()
    await asyncio.wait_for(task, 2)
    assert world.launches  # at least one tick ran


def test_every_completed_tick_pings_healthchecks(sup, world) -> None:
    sup.tick()
    sup.tick()
    assert world.pings == 2


def test_offline_alert_also_reaches_the_panel(tmp_path, world, monkeypatch) -> None:
    sup = make_sup(tmp_path, world, ("Pie64",))
    monkeypatch.setattr(L.notify, "maybe_alert", lambda kind, fields: None)
    seen: list[tuple[str, str, dict]] = []
    sup.subscribe_alerts(lambda name, kind, fields: seen.append((name, kind, fields)))
    world.online[5555] = False
    sup.tick()  # miss 1 -> retry in 2 min
    world.now += timedelta(minutes=2)
    sup.tick()  # miss 2 -> retry in 4 min
    world.now += timedelta(minutes=4)
    sup.tick()  # miss 3 -> alert
    assert seen == [("Pie64", "offline", {"misses": 3})]


def test_two_ticks_at_once_launch_only_one_worker(tmp_path: Path, world: World) -> None:
    """The API's startup tick and the supervisor loop's first tick land in two worker
    threads at the same instant (brawlfarm/api/app.py's lifespan and _serve's task). Two
    ticks running together would each read "nothing is running" from status.json, both
    pass the launch guard and start a second worker for the same instance, which is the
    one-worker-per-instance rail. tick() serialises instead."""
    world.probe_delay_s = 0.05  # the real probe shells out to adb; that is the window
    sup = make_sup(tmp_path, world, ("Pie64",))
    ready = threading.Barrier(2)
    failures: list[Exception] = []

    def ticker() -> None:
        try:
            ready.wait(timeout=5)
            sup.tick()
        except Exception as exc:  # a thread that dies would otherwise just print and pass
            failures.append(exc)

    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=ticker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
    finally:
        sys.setswitchinterval(previous_interval)

    assert failures == []
    assert all(not t.is_alive() for t in threads)  # both ticks finished
    assert len(world.launches) == 1
    assert len(world.alive) == 1


def test_clearing_the_token_clears_the_applied_one(tmp_path, world, monkeypatch) -> None:
    monkeypatch.delenv("BRAWL_API_TOKEN", raising=False)
    sup = make_sup(tmp_path, world, ("Pie64",))
    assert config.API_TOKEN == "tok"  # _settings() sets one, and __init__ applied it

    sup.settings.connection.brawl_api_token = ""
    sup.apply_settings(sup.settings)
    # Blank means blank. Before this, the last non-blank token stuck for the life of the
    # process, so clearing it in Settings changed nothing until a restart.
    assert config.API_TOKEN == ""

    monkeypatch.setenv("BRAWL_API_TOKEN", "from-the-environment")
    sup.apply_settings(sup.settings)
    # The environment override the spec promises is still the fallback, not the leftover.
    assert config.API_TOKEN == "from-the-environment"


def test_observe_override_launches_the_worker_with_observe(sup, world, tmp_path) -> None:
    sup.observe("Pie64", True)
    assert (scheduler.read_override("Pie64") or {})["mode"] == "observe"
    world.now += timedelta(seconds=L.BOOT_GRACE_S + 20)
    sup.tick()
    args = [a for a, _env, n in world.launches if n == "Pie64"][-1]
    assert args == ["--observe"]
    assert "--select-brawler" not in args and "--dnd" not in args


def test_observe_does_not_read_as_a_scheduled_break(sup, world, tmp_path) -> None:
    sup.observe("Pie64", True)
    views = sup.tick()
    view = _view(views, "Pie64")
    assert view.desired == "observe"
    assert view.state == InstanceState.STARTING


def test_observe_off_is_the_ordinary_stop(sup, world, tmp_path) -> None:
    sup.observe("Pie64", True)
    sup.observe("Pie64", False)
    assert (scheduler.read_override("Pie64") or {})["mode"] == "stop"
    assert (S.instance_dir(tmp_path, "Pie64") / "stop.flag").exists()


def test_observe_stops_a_live_farm_worker_before_relaunching_it(sup, world, tmp_path) -> None:
    world.alive.add(4242)
    _heartbeat(tmp_path, "Pie64", 4242, world.now)  # no "mode" field = a farm worker
    sup.observe("Pie64", True)
    views = sup.tick()
    # The farm worker taps, so it is stopped rather than left running under an observe
    # override; the relaunch is the next tick's job.
    assert (S.instance_dir(tmp_path, "Pie64") / "stop.flag").exists()
    assert _view(views, "Pie64").state == InstanceState.STOPPING
    assert world.kills == []
    world.alive.discard(4242)  # the worker exited at the menu
    world.now += timedelta(seconds=60)
    sup.tick()
    assert [a for a, _env, n in world.launches if n == "Pie64"][-1] == ["--observe"]


def test_a_live_observer_stops_when_observe_is_cleared(sup, world, tmp_path) -> None:
    world.alive.add(4242)
    _heartbeat(tmp_path, "Pie64", 4242, world.now, mode="observe")
    # No override and the schedule off means desired run, and an observer has no session
    # cap of its own, so the tick has to stop it.
    views = sup.tick()
    assert (S.instance_dir(tmp_path, "Pie64") / "stop.flag").exists()
    assert _view(views, "Pie64").state == InstanceState.STOPPING


# --- the emulator guard: restart a leaking HD-Player, one step per tick -----------------

PLAYER = 777
WORKER = 4242


def _guarded(
    tmp_path: Path, world: World, recycle_gb: float = 40.0, floor_gb: float = 0.0
) -> Supervisor:
    """One farming instance with a live worker and a live HD-Player, limits applied."""
    sup = make_sup(tmp_path, world, ("Pie64",))
    sup.settings.behavior.emulator_recycle_gb = recycle_gb
    sup.settings.behavior.commit_floor_gb = floor_gb
    sup.apply_settings(sup.settings)
    world.alive.add(WORKER)
    world.players["Pie64"] = PLAYER
    _heartbeat(tmp_path, "Pie64", WORKER, world.now)
    return sup


def _next(sup: Supervisor, world: World, seconds: float = 60) -> InstanceView:
    world.now += timedelta(seconds=seconds)
    return _view(sup.tick(), "Pie64")


def _to_booting(sup: Supervisor, world: World) -> None:
    """Trigger, the worker exits, close, confirm gone: the next tick opens HD-Player."""
    world.player_gb = 41.0
    sup.tick()
    world.alive.discard(WORKER)
    _next(sup, world)
    _next(sup, world)


def _capture_alerts(sup: Supervisor, monkeypatch) -> tuple[list, list]:
    pushed: list = []
    panel: list = []
    monkeypatch.setattr(L.notify, "maybe_alert", lambda kind, fields: pushed.append((kind, fields)))
    sup.subscribe_alerts(lambda name, kind, fields: panel.append((name, kind, fields)))
    return pushed, panel


def _schedule_files(tmp_path: Path) -> dict[Path, dict]:
    """Both files as data, less the stamp the scheduler's own tick moves every minute."""
    out = {}
    for f in tmp_path.rglob("*.json"):
        if f.name in ("schedule.json", "override.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            for section in data.values():
                if isinstance(section, dict):
                    section.pop("updated_at", None)
            out[f] = data
    return out


def test_guard_is_off_by_default(tmp_path, world) -> None:
    sup = _guarded(tmp_path, world, recycle_gb=0.0, floor_gb=0.0)
    world.player_gb = 500.0
    world.commit_gb = 0.1
    view = _view(sup.tick(), "Pie64")
    assert view.state == InstanceState.FARMING
    assert view.desired_reason == "disabled"  # the schedule's own reason, nothing of ours
    assert not (S.instance_dir(tmp_path, "Pie64") / "stop.flag").exists()
    assert world.closes == []


def test_guard_needs_a_player_to_restart(tmp_path, world) -> None:
    sup = _guarded(tmp_path, world)
    world.player_gb = 41.0
    world.players.clear()
    assert _view(sup.tick(), "Pie64").state == InstanceState.FARMING
    assert not (S.instance_dir(tmp_path, "Pie64") / "stop.flag").exists()


def test_guard_ignores_a_failed_commit_read(tmp_path, world) -> None:
    sup = _guarded(tmp_path, world, recycle_gb=0.0, floor_gb=12.0)
    world.player_gb = 41.0
    world.commit_gb = float("inf")  # what commit_free_gb answers when the read fails
    assert _view(sup.tick(), "Pie64").state == InstanceState.FARMING


def test_memory_trigger_runs_the_full_restart_and_launches_once(tmp_path, world) -> None:
    sup = _guarded(tmp_path, world)
    sup.tick()  # the scheduler's own first tick writes schedule.json
    before = _schedule_files(tmp_path)
    assert before
    world.player_gb = 40.0  # at the limit counts

    view = _view(sup.tick(), "Pie64")  # 1. stopping: the graceful path, the match finishes
    assert view.state == InstanceState.STOPPING
    assert view.desired_reason == "emulator_recycle"
    assert (S.instance_dir(tmp_path, "Pie64") / "stop.flag").exists()
    assert world.kills == [] and world.closes == []

    _heartbeat(tmp_path, "Pie64", WORKER, world.now)
    assert _next(sup, world).state == InstanceState.STOPPING  # still in its match
    assert world.closes == []

    world.alive.discard(WORKER)
    view = _next(sup, world)  # 2. closing
    assert view.state == InstanceState.SCHEDULED_BREAK
    assert view.desired_reason == "emulator_recycle"
    assert world.closes == [(PLAYER, 20.0)]
    assert world.opens == []

    view = _next(sup, world)  # gone is confirmed on a later tick, and that is the whole tick
    assert world.opens == []

    view = _next(sup, world)  # 3. booting: open, and no boot check on the same tick
    assert world.opens == [(sup.settings.behavior.emulator_path, "Pie64")]
    assert world.boot_checks == []

    world.booted = False
    view = _next(sup, world)
    assert world.boot_checks == [5555]
    assert view.state == InstanceState.SCHEDULED_BREAK

    world.booted = True
    _next(sup, world)  # 4. settling starts here
    assert world.boot_checks == [5555, 5555]
    view = _next(sup, world)
    assert view.desired_reason == "emulator_recycle"
    assert world.launches == []

    view = _next(sup, world)  # 120 s after boot: 5. the normal path launches
    assert [n for _, _, n in world.launches] == ["Pie64"]
    assert view.state == InstanceState.STARTING
    assert view.desired_reason == "disabled"  # the schedule's own reason, nothing of ours
    assert not (S.instance_dir(tmp_path, "Pie64") / "stop.flag").exists()

    _next(sup, world)
    assert len(world.launches) == 1
    assert len(world.closes) == 1
    assert _schedule_files(tmp_path) == before  # never the schedule, never the override
    assert scheduler.read_override("Pie64") is None


def test_stubborn_worker_is_killed_by_the_existing_escalation(tmp_path, world) -> None:
    sup = _guarded(tmp_path, world)
    world.player_gb = 41.0
    sup.tick()
    _heartbeat(tmp_path, "Pie64", WORKER, world.now)
    _next(sup, world, L.STOP_ESCALATE_S)
    assert world.kills == [WORKER]
    assert world.closes == []  # the kill was this tick's step
    _next(sup, world)
    assert world.closes == [(PLAYER, 20.0)]


def test_closed_emulator_is_not_an_offline_fault(tmp_path, world, monkeypatch) -> None:
    sup = _guarded(tmp_path, world)
    pushed, panel = _capture_alerts(sup, monkeypatch)
    _to_booting(sup, world)
    world.online[5555] = False  # adb cannot see a closed or booting emulator
    world.booted = False
    for _ in range(4):
        view = _next(sup, world)
        assert view.state == InstanceState.SCHEDULED_BREAK
        assert sup._backoff.until("Pie64", world.now) is None
    assert sup._backoff.misses("Pie64") == 0
    assert pushed == [] and panel == []


def test_commit_trigger_needs_hd_player_to_hold_the_memory(tmp_path, world) -> None:
    sup = _guarded(tmp_path, world, recycle_gb=0.0, floor_gb=12.0)
    world.commit_gb = 11.9
    world.player_gb = L.MIN_RECYCLE_GB - 0.1  # something else holds the commit
    view = _view(sup.tick(), "Pie64")
    assert view.state == InstanceState.FARMING
    assert not (S.instance_dir(tmp_path, "Pie64") / "stop.flag").exists()

    world.player_gb = L.MIN_RECYCLE_GB
    _heartbeat(tmp_path, "Pie64", WORKER, world.now)
    assert _next(sup, world).desired_reason == "emulator_recycle"


def test_commit_at_the_floor_does_not_trigger(tmp_path, world) -> None:
    sup = _guarded(tmp_path, world, recycle_gb=0.0, floor_gb=12.0)
    world.commit_gb = 12.0
    world.player_gb = 41.0
    assert _view(sup.tick(), "Pie64").state == InstanceState.FARMING


def test_no_second_restart_inside_thirty_minutes(tmp_path, world) -> None:
    sup = _guarded(tmp_path, world)
    started = world.now
    _to_booting(sup, world)
    for _ in range(5):
        _next(sup, world)
    assert len(world.launches) == 1

    # The leak is back at once and the new worker is up, but the window is from the start.
    world.alive.add(5000)
    world.now = started + timedelta(seconds=L.MIN_INTERVAL_S - 1)
    _heartbeat(tmp_path, "Pie64", 5000, world.now)
    assert _view(sup.tick(), "Pie64").state == InstanceState.FARMING

    world.now = started + timedelta(seconds=L.MIN_INTERVAL_S)
    _heartbeat(tmp_path, "Pie64", 5000, world.now)
    assert _view(sup.tick(), "Pie64").desired_reason == "emulator_recycle"


def test_boot_timeout_alerts_once_and_stays_stopped_until_start(
    tmp_path, world, monkeypatch
) -> None:
    sup = _guarded(tmp_path, world)
    pushed, panel = _capture_alerts(sup, monkeypatch)
    world.booted = False
    _to_booting(sup, world)
    _next(sup, world)  # open

    for _ in range(5):  # 300 s after the open: still waiting
        assert _next(sup, world).desired_reason == "emulator_recycle"
    assert pushed == []

    view = _next(sup, world)  # 360 s
    assert view.state == InstanceState.STOPPED
    assert view.desired_reason == "emulator_recycle_failed"
    assert view.pid is None
    assert view.note != ""
    assert len(pushed) == 1 and len(panel) == 1
    assert pushed[0][0] == "offline"
    assert panel[0][:2] == ("Pie64", "offline")

    world.booted = True  # no retry, even with the emulator up and adb answering
    checks = len(world.boot_checks)
    for _ in range(5):
        view = _next(sup, world)
        assert view.desired_reason == "emulator_recycle_failed"
    assert len(world.boot_checks) == checks
    assert len(world.opens) == 1 and world.launches == []
    assert len(pushed) == 1 and len(panel) == 1

    sup.start("Pie64")
    view = _next(sup, world)
    assert [n for _, _, n in world.launches] == ["Pie64"]
    assert view.desired_reason == "disabled"  # the schedule's own reason, nothing of ours


def test_failed_open_goes_straight_to_failed(tmp_path, world, monkeypatch) -> None:
    sup = _guarded(tmp_path, world)
    pushed, _panel = _capture_alerts(sup, monkeypatch)
    world.open_ok = False
    _to_booting(sup, world)
    view = _next(sup, world)
    assert view.desired_reason == "emulator_recycle_failed"
    assert len(pushed) == 1
    _next(sup, world)
    assert len(world.opens) == 1
    assert len(pushed) == 1


def test_close_never_confirmed_fails_after_three_ticks(tmp_path, world, monkeypatch) -> None:
    sup = _guarded(tmp_path, world)
    pushed, _panel = _capture_alerts(sup, monkeypatch)
    world.close_sticks = True
    world.player_gb = 41.0
    sup.tick()
    world.alive.discard(WORKER)
    _next(sup, world)  # the close
    assert _next(sup, world).desired_reason == "emulator_recycle"
    assert _next(sup, world).desired_reason == "emulator_recycle"
    assert _next(sup, world).desired_reason == "emulator_recycle_failed"
    assert len(pushed) == 1
    assert world.opens == []
    assert len(world.closes) == 1


def test_refused_close_fails_without_waiting(tmp_path, world, monkeypatch) -> None:
    sup = _guarded(tmp_path, world)
    pushed, _panel = _capture_alerts(sup, monkeypatch)
    world.close_ok = False
    world.player_gb = 41.0
    sup.tick()
    world.alive.discard(WORKER)
    assert _next(sup, world).desired_reason == "emulator_recycle_failed"
    assert len(pushed) == 1
    assert world.opens == []


def test_owner_stop_while_stopping_abandons_the_restart(tmp_path, world) -> None:
    sup = _guarded(tmp_path, world)
    world.player_gb = 41.0
    sup.tick()
    sup.stop("Pie64")
    world.alive.discard(WORKER)
    for _ in range(8):
        view = _next(sup, world)
        assert view.state == InstanceState.STOPPED
        assert view.desired_reason == "override_stop"
    assert world.closes == [] and world.opens == [] and world.launches == []


def test_owner_stop_past_stopping_finishes_the_boot_then_stays_stopped(tmp_path, world) -> None:
    sup = _guarded(tmp_path, world)
    _to_booting(sup, world)
    sup.stop("Pie64")
    for _ in range(8):
        view = _next(sup, world)
        assert view.state == InstanceState.STOPPED
        assert view.desired_reason == "override_stop"
    assert len(world.opens) == 1  # not left closed halfway through
    assert world.launches == []

    sup.start("Pie64")
    _next(sup, world)
    assert [n for _, _, n in world.launches] == ["Pie64"]


def test_guard_settings_apply_through_apply_settings(tmp_path, world) -> None:
    sup = _guarded(tmp_path, world, recycle_gb=0.0, floor_gb=0.0)
    world.player_gb = 41.0
    assert _view(sup.tick(), "Pie64").state == InstanceState.FARMING

    sup.settings.behavior.emulator_recycle_gb = 40.0
    sup.apply_settings(sup.settings)
    _heartbeat(tmp_path, "Pie64", WORKER, world.now)
    assert _next(sup, world).desired_reason == "emulator_recycle"


def test_guard_changes_nothing_while_desired_is_stop(tmp_path, world) -> None:
    sup = _guarded(tmp_path, world)
    world.player_gb = 41.0
    sup.stop("Pie64")
    world.alive.discard(WORKER)
    for _ in range(3):
        assert _next(sup, world).desired_reason == "override_stop"
    assert world.closes == []


def test_guard_never_restarts_under_a_dead_worker(tmp_path, world) -> None:
    sup = _guarded(tmp_path, world)
    world.player_gb = 41.0
    world.alive.discard(WORKER)
    view = _view(sup.tick(), "Pie64")
    assert view.desired_reason == "disabled"  # the schedule's own reason, nothing of ours
    assert [n for _, _, n in world.launches] == ["Pie64"]  # the ordinary relaunch
    assert world.closes == []


def test_guard_leaves_an_observer_alone(tmp_path, world) -> None:
    sup = _guarded(tmp_path, world)
    world.player_gb = 41.0
    scheduler.write_override("Pie64", "observe", world.now + timedelta(days=1))
    _heartbeat(tmp_path, "Pie64", WORKER, world.now, mode="observe")
    assert _view(sup.tick(), "Pie64").state == InstanceState.FARMING
    assert world.closes == []


def test_restart_respects_the_one_launch_per_tick_rule(tmp_path, world) -> None:
    sup = make_sup(tmp_path, world)
    sup.settings.behavior.emulator_recycle_gb = 40.0
    sup.apply_settings(sup.settings)
    world.alive |= {1111, 2222}
    world.players["Rome64"] = PLAYER  # Pie64 has no player, so only Rome64 restarts
    world.player_gb = 41.0
    _heartbeat(tmp_path, "Pie64", 1111, world.now)
    _heartbeat(tmp_path, "Rome64", 2222, world.now)
    sup.tick()
    world.alive.discard(2222)
    for _ in range(5):  # close, confirm, open, boot, settle: the next tick launches
        world.now += timedelta(seconds=60)
        _heartbeat(tmp_path, "Pie64", 1111, world.now)
        sup.tick()
    assert world.launches == []

    world.alive.discard(1111)  # Pie64 goes first this tick and takes the launch
    world.now += timedelta(seconds=60)
    views = sup.tick()
    assert [n for _, _, n in world.launches] == ["Pie64"]
    assert _view(views, "Rome64").note == "Waiting for its turn to start"
    world.now += timedelta(seconds=60)
    sup.tick()
    assert [n for _, _, n in world.launches] == ["Pie64", "Rome64"]
