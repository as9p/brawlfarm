"""The supervisor tick and its controls (spec section 6), ported from the legacy
PowerShell watchdog.

One tick: scheduler tick for every instance; then per instance read status.json,
classify the heartbeat, read the scheduler's desired state, and act: desired stop and
alive -> write stop.flag, kill by PID after STOP_ESCALATE_S; desired run and unhealthy
-> probe adb (offline backoff), kill a stale PID, launch (one launch per tick).
Everything the panel shows is derived here once per tick into InstanceView objects.

The emulator guard runs first in each instance's tick: when HD-Player has leaked past its
limits it holds the instance on the stop path and restarts HD-Player, one step per tick.
recycle.json in the instance folder carries a restart over a supervisor restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from brawlfarm import settings as S
from brawlfarm.core import config, events, notify, scheduler, status
from brawlfarm.core.jsonio import atomic_write_json
from brawlfarm.supervisor import emulator, process
from brawlfarm.supervisor.backoff import OfflineBackoff
from brawlfarm.supervisor.state import (
    RECYCLE_FAILED_REASON,
    Health,
    InstanceState,
    InstanceView,
    classify,
    derive_state,
    heartbeat_age_s,
)

log = logging.getLogger("brawlfarm.supervisor")

TICK_S = 60.0
BOOT_GRACE_S = 180.0  # after a launch, do not relaunch before the worker can heartbeat
STOP_ESCALATE_S = 110.0  # graceful stop.flag first, kill by PID after this
ALERT_AFTER_MISSES = 3
STOP_OVERRIDE_DAYS = 365  # "stopped until you start it again"
STOP_FLAG = "stop.flag"
MIN_RECYCLE_GB = 8.0  # a commit trigger needs HD-Player itself to hold this much
MIN_INTERVAL_S = 1800.0  # one emulator restart per instance, from the last one's start
CLOSE_GRACE_S = 20.0  # for the window to close before HD-Player is terminated
CLOSE_CONFIRM_TICKS = 3  # ticks HD-Player may still be listed after the close
BOOT_TIMEOUT_S = 360.0  # from the open to sys.boot_completed
SETTLE_S = 120.0  # after boot, before the worker: the stagger rule
RECYCLE_REASON = "emulator_recycle"
RECYCLE_FILE = "recycle.json"  # the restart in flight; gone when it ends
RECYCLE_LAST_FILE = "recycle_last.json"  # the MIN_INTERVAL_S clock; outlives the job


class _Phase(StrEnum):
    STOPPING = "stopping"
    CLOSING = "closing"
    BOOTING = "booting"
    SETTLING = "settling"
    FAILED = "failed"


_PHASE_NOTES = {
    _Phase.STOPPING: "Restarting the emulator after this match",
    _Phase.CLOSING: "Restarting the emulator: closing it",
    _Phase.BOOTING: "Restarting the emulator: waiting for it to boot",
    _Phase.SETTLING: "Restarting the emulator: letting it settle",
    _Phase.FAILED: "Emulator restart failed; stopped until you start it again",
}


@dataclass
class _Recycle:
    """One instance's emulator restart. The tick advances it a step at a time and never
    sleeps; RECYCLE_FILE carries phase and since over a supervisor restart. The other fields
    are not saved: _recycle_resume works them out again."""

    phase: _Phase
    since: datetime  # when this phase began; in BOOTING, when HD-Player was opened
    still_listed: int = 0  # ticks find_player has still seen HD-Player since the close
    opened: bool = False
    resumed: bool = False  # loaded from RECYCLE_FILE and not stepped yet


class Supervisor:
    def __init__(
        self,
        settings: S.AppSettings,
        home: Path,
        *,
        clock: Callable[[], datetime] = datetime.now,
        probe: Callable[..., bool] = process.instance_online,
        launcher: Callable[..., object] = process.launch_worker,
        alive: Callable[[int], bool] = process.pid_alive,
        killer: Callable[..., bool] = process.kill_worker,
        sleep: Callable[[float], None] = time.sleep,
        events_refresh: Callable[..., int] = events.refresh,
        pinger: Callable[[], bool] = notify.ping_healthchecks,
        interval_s: float = TICK_S,
        find_player: Callable[[str], int | None] = emulator.find_player,
        private_gb: Callable[[int], float | None] = emulator.private_gb,
        commit_free_gb: Callable[[], float] = emulator.commit_free_gb,
        close_player: Callable[..., bool] = emulator.close_player,
        open_player: Callable[..., int | None] = emulator.open_player,
        boot_done: Callable[..., bool] = emulator.boot_done,
    ) -> None:
        self.settings = settings
        self.home = Path(home).resolve()
        self.interval_s = interval_s
        self._clock, self._probe, self._launch = clock, probe, launcher
        self._alive, self._kill, self._sleep, self._refresh = alive, killer, sleep, events_refresh
        self._ping = pinger
        self._find_player, self._private_gb = find_player, private_gb
        self._commit_free_gb = commit_free_gb
        self._close_player, self._open_player, self._boot_done = (
            close_player,
            open_player,
            boot_done,
        )
        self._recycling: dict[str, _Recycle] = {}
        self._recycled_at: dict[str, datetime] = {}
        self._commit_warned = False
        self._backoff = OfflineBackoff()
        self._stop_asked: dict[str, datetime] = {}
        self._launched_at: dict[str, datetime] = {}
        self._procs: dict[str, object] = {}
        self._views: dict[str, InstanceView] = {}
        self._listeners: list[Callable[[InstanceView], None]] = []
        self._alert_listeners: list[Callable[[str, str, dict], None]] = []
        self._tick_lock = threading.Lock()
        self._poke: asyncio.Event | None = None
        self._shutdown = False
        self.apply_settings(settings)
        for inst in settings.instances:
            self._recycle_load(inst.name)

    # --- wiring -----------------------------------------------------------------------

    def apply_settings(self, settings: S.AppSettings) -> None:
        """Point the core at the home directory and the instance table; safe to call again
        after the user edits settings (phase 5)."""
        self.settings = settings
        config.set_home(self.home)
        config.set_instances(S.instances_table(settings))
        # Blank means blank: a token cleared in Settings has to clear the applied one, not
        # leave the last non-blank value in place for the life of the process. The
        # environment stays the developer override the spec promises.
        config.API_TOKEN = settings.connection.brawl_api_token or os.environ.get(
            "BRAWL_API_TOKEN", ""
        )
        scheduler.set_default_enabled(settings.scheduler.default_enabled)
        self._recycle_gb = settings.behavior.emulator_recycle_gb
        self._commit_floor_gb = settings.behavior.commit_floor_gb
        self._emulator_path = settings.behavior.emulator_path
        n = settings.notifications
        notify.configure(
            webhook_url=n.webhook_url or None,  # blank defers to the environment
            ntfy_server=n.ntfy_server or None,
            ntfy_topic=n.ntfy_topic or None,
            events=list(n.events),
            healthchecks_url=n.healthchecks_url or None,
        )

    def views(self) -> list[InstanceView]:
        return [self._views[i.name] for i in self.settings.instances if i.name in self._views]

    def subscribe(self, cb: Callable[[InstanceView], None]) -> None:
        self._listeners.append(cb)

    def subscribe_alerts(self, cb: Callable[[str, str, dict], None]) -> None:
        """Called with (instance name, alert kind, fields) for alerts the supervisor raises
        itself. The panel's alert store registers here; the push notifier is separate and
        keeps its own cooldown."""
        self._alert_listeners.append(cb)

    def _fire_alert(self, name: str, kind: str, fields: dict) -> None:
        for cb in self._alert_listeners:
            try:
                cb(name, kind, fields)
            except Exception as exc:
                log.debug("alert listener failed: %s", exc)

    def _dir(self, name: str) -> Path:
        return S.instance_dir(self.home, name)

    def _flag(self, name: str) -> Path:
        return self._dir(name) / STOP_FLAG

    # --- the tick ---------------------------------------------------------------------

    def tick(self) -> list[InstanceView]:
        """One tick at a time. The API's startup tick and the supervisor loop's first
        tick both land here from worker threads at process start, and two of them running
        together would each read "nothing is running" from status.json, both pass the launch
        guard and start a second worker for the same instance. Blocking, not try-acquire:
        the second tick then simply runs after the first and sees the booting state.
        """
        with self._tick_lock:
            now = self._clock()
            rc = scheduler.tick(now)
            if rc != 0:
                log.warning("scheduler tick failed (rc=%s); fail open to always-run", rc)
            if self.settings.connection.brawl_api_token:  # the rotation fetch needs the API
                try:
                    self._refresh(now)
                except Exception as exc:  # best effort, never blocks the tick
                    log.debug("events refresh failed: %s", exc)
            self._recycle_prune()
            launched_this_tick = False
            out: list[InstanceView] = []
            for inst in self.settings.instances:
                try:
                    view, launched = self._tick_instance(inst, now, launched_this_tick)
                except Exception:  # one broken instance must never strand the others
                    log.exception("%s: tick failed", inst.name)
                    view, launched = self._views.get(inst.name) or _error_view(inst), False
                launched_this_tick = launched_this_tick or launched
                out.append(view)
                previous = self._views.get(inst.name)
                self._views[inst.name] = view
                if previous is None or previous.state != view.state:
                    for cb in self._listeners:
                        try:
                            cb(view)
                        except Exception as exc:
                            log.debug("listener failed: %s", exc)
            log.info("tick: %s", ", ".join(f"{v.name}={v.state}" for v in out) or "no instances")
            # Only a tick that got this far pings: a wedged supervisor stops pinging, which is
            # exactly what the healthchecks alarm is for. Never allowed to raise.
            try:
                self._ping()
            except Exception as exc:
                log.debug("healthchecks ping failed: %s", exc)
            return out

    def _tick_instance(
        self, inst: S.InstanceSettings, now: datetime, launched_this_tick: bool
    ) -> tuple[InstanceView, bool]:
        name = inst.name
        st = status.read_status(self._dir(name))
        pid = _pid_of(st)
        alive = pid is not None and self._alive(pid)
        health = classify(st, alive, now)
        desired = scheduler.sched_desired(name, now)
        want = (desired or {}).get("state", "run")
        reason = (desired or {}).get("reason")
        until = _parse_until((desired or {}).get("until"))
        note = ""
        launched = False
        # A restart in flight is a stop as far as this tick goes: the stop branch below does
        # the graceful stop and its escalation, and it is also what keeps the probe away, so
        # the closed emulator is neither offline nor a backoff miss, and nothing launches.
        # A stop that was already wanted keeps its own reason and outlives the restart.
        job = self._recycle_step(inst, now, want, alive)
        if job is not None:
            note = _PHASE_NOTES[job.phase]
            if job.phase == _Phase.FAILED or want != "stop":
                reason = RECYCLE_FAILED_REASON if job.phase == _Phase.FAILED else RECYCLE_REASON
                want, until = "stop", None
        # One worker per instance means the live one has to be the right kind: the farm
        # controller writes no "mode", the observer writes "observe". A live worker whose
        # kind no longer matches what is wanted is stopped like any other stop, and the
        # next tick relaunches it with the right worker_args. Without this a healthy farm
        # worker keeps tapping under an observe override, and a healthy observer (which has
        # no session cap) keeps watching after the override clears.
        wrong_kind = (
            alive
            and want != "stop"
            and (st or {}).get("mode", "farm") != ("observe" if want == "observe" else "farm")
        )

        if want == "stop" or wrong_kind:
            self._backoff.clear(name)
            if alive:
                self._request_stop(name, now)
                if self._maybe_escalate(name, pid, now):
                    alive, health = False, Health.DEAD
            else:
                self._stop_asked.pop(name, None)
        else:
            if health == Health.HEALTHY:
                self._backoff.clear(name)
                if self._maybe_escalate(name, pid, now):  # a restart in flight
                    alive, health = False, Health.DEAD
            elif self._booting(name, now):
                note = "Starting; waiting for the first heartbeat"
            elif (retry := self._backoff.until(name, now)) is not None:
                note = _offline_note(retry, now)
            elif launched_this_tick:
                note = "Waiting for its turn to start"
            elif not self._probe(self.settings.connection.adb_path, inst.adb_port):
                retry = self._backoff.record_miss(name, now)
                misses = self._backoff.misses(name)
                log.warning("%s: adb probe failed (miss %d); retry at %s", name, misses, retry)
                if misses == ALERT_AFTER_MISSES:
                    notify.maybe_alert("offline", {"instance": name, "misses": misses})
                    self._fire_alert(name, "offline", {"misses": misses})
                note = _offline_note(retry, now)
            else:
                self._backoff.clear(name)
                stopped = True
                if alive and pid is not None:
                    log.warning("%s: stale heartbeat, killing pid %s before relaunch", name, pid)
                    stopped = self._ensure_gone(name, pid)
                    self._sleep(0.5)
                    if stopped:
                        alive, health = False, Health.DEAD
                if stopped and self._kill_hung_launch(name):
                    self._launch_worker(inst, desired, now)
                    launched = True
                    note = "Starting"
                else:  # a worker that would not die is never relaunched over
                    note = "Worker could not be stopped; will retry next tick"

        stop_deadline = (
            self._stop_asked[name] + timedelta(seconds=STOP_ESCALATE_S)
            if name in self._stop_asked and alive
            else None
        )
        state, shown_until = derive_state(
            health=health,
            desired=want,
            desired_reason=reason,
            desired_until=until,
            stop_pending=stop_deadline,
            booting=self._booting(name, now) or launched,
            offline_until=self._backoff.until(name, now),
            status=st,
        )
        if state == InstanceState.STOPPING:
            note = "Stopping after this match"
        view = InstanceView(
            name=name,
            adb_port=inst.adb_port,
            state=state,
            health=health,
            pid=pid if alive else None,
            heartbeat_age_s=heartbeat_age_s(st, now),
            phase=(st or {}).get("phase"),
            desired=want,
            desired_reason=reason,
            until=shown_until,
            games_played=_int_or_none((st or {}).get("games_played")),
            farm_brawler=(st or {}).get("farm_brawler") or None,
            note=note,
        )
        return view, launched

    # --- helpers ----------------------------------------------------------------------

    def _booting(self, name: str, now: datetime) -> bool:
        # The window is inclusive of its last instant: TICK_S divides BOOT_GRACE_S, so a
        # tick can land exactly on it and the worker still deserves that tick to report.
        started = self._launched_at.get(name)
        if started is None or (now - started).total_seconds() > BOOT_GRACE_S:
            return False
        proc = self._procs.get(name)
        poll = getattr(proc, "poll", None)
        return poll is None or poll() is None

    def _ensure_gone(self, name: str, pid: int) -> bool:
        """Kill and confirm. True when the killer succeeded or the PID is gone anyway; False
        (logged) when the worker is still there, which blocks every relaunch that would put a
        second worker on the instance."""
        if self._kill(pid, log=log.warning) or not self._alive(pid):
            return True
        log.error("%s: pid %s could not be stopped; not relaunching", name, pid)
        return False

    def _kill_hung_launch(self, name: str) -> bool:
        """One worker per instance: a worker we launched that never wrote status.json has no
        PID in the file to kill, so kill it by the PID of our own launch record before the
        relaunch (still kill-by-PID, still guarded by kill_worker's command-line check).
        False when it would not die, so the caller does not launch a second one."""
        proc = self._procs.get(name)
        poll = getattr(proc, "poll", None)
        pid = getattr(proc, "pid", None)
        if not pid or (poll is not None and poll() is not None):
            return True  # nothing of ours is still running
        log.warning(
            "%s: launched pid %s never wrote status.json, killing before relaunch", name, pid
        )
        if not self._ensure_gone(name, pid):
            return False
        self._sleep(0.5)
        return True

    def _request_stop(self, name: str, now: datetime) -> None:
        if name in self._stop_asked:
            return
        flag = self._flag(name)
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text(f"stop requested {now:%Y-%m-%dT%H:%M:%S}\n", encoding="utf-8")
        self._stop_asked[name] = now
        log.info("%s: stop requested (stop.flag written)", name)

    def _maybe_escalate(self, name: str, pid: int | None, now: datetime) -> bool:
        """Kill by PID once a requested stop has waited STOP_ESCALATE_S. True when killed."""
        asked = self._stop_asked.get(name)
        if asked is None or pid is None:
            return False
        if (now - asked).total_seconds() < STOP_ESCALATE_S:
            return False
        log.warning("%s: stop not honoured in %ds, killing pid %s", name, STOP_ESCALATE_S, pid)
        if not self._ensure_gone(name, pid):
            return False  # the deadline stays set, so the next tick escalates again
        self._stop_asked.pop(name, None)
        return True

    def _launch_worker(self, inst: S.InstanceSettings, desired: dict | None, now: datetime) -> None:
        max_minutes = (desired or {}).get("max_minutes")
        observe = (desired or {}).get("state") == "observe"
        args = S.worker_args(
            self.settings, float(max_minutes) if max_minutes else None, observe=observe
        )
        env = {**os.environ, **S.worker_env(self.settings, inst, self.home)}
        self._flag(inst.name).unlink(missing_ok=True)
        self._stop_asked.pop(inst.name, None)
        proc = self._launch(args, env, self.home / "logs", inst.name)
        self._procs[inst.name] = proc
        self._launched_at[inst.name] = now
        log.info(
            "%s: launched worker pid %s (%s)", inst.name, getattr(proc, "pid", "?"), " ".join(args)
        )

    # --- the emulator guard -----------------------------------------------------------

    def _recycle_step(
        self, inst: S.InstanceSettings, now: datetime, want: str, alive: bool
    ) -> _Recycle | None:
        """Advance this instance's emulator restart by at most one step. Returns the restart
        still in flight, or None when there is none and the tick runs as normal.

        Never raises. A step that raises fails the restart with its one alert, and the tick's
        stop branch still does the graceful stop. With no restart in flight nothing has been
        touched, so a raising trigger is logged and the instance farms on."""
        try:
            return self._recycle_advance(inst, now, want, alive)
        except Exception as exc:
            job = self._recycling.get(inst.name)
            self._log_recycle(inst.name, f"step raised {exc!r}", logging.ERROR, exc)
            if job is not None and job.phase != _Phase.FAILED:
                self._recycle_failed(inst.name, job, job.phase.value)
            return job

    def _recycle_advance(
        self, inst: S.InstanceSettings, now: datetime, want: str, alive: bool
    ) -> _Recycle | None:
        name = inst.name
        job = self._recycling.get(name)
        if job is None:
            return self._recycle_trigger(name, now) if want == "run" and alive else None
        if job.resumed and not self._recycle_resume(name, job, now, want):
            return None
        if job.phase == _Phase.STOPPING:
            if want == "stop":  # nothing is closed yet, so the wanted stop just carries on
                self._recycle_end(name)
                self._log_recycle(name, "abandoned, a stop was asked for")
                return None
            if not alive:
                self._recycle_close(name, job, now)
        elif job.phase == _Phase.CLOSING:
            if self._find_player(name) is None:
                job.phase, job.since = _Phase.BOOTING, now
                self._recycle_save(name, job)
                self._log_recycle(name, "closed")
            else:
                job.still_listed += 1
                if job.still_listed >= CLOSE_CONFIRM_TICKS:
                    self._recycle_failed(name, job, "closing")
        elif job.phase == _Phase.BOOTING:
            self._recycle_boot(inst, job, now)
        elif job.phase == _Phase.SETTLING and (now - job.since).total_seconds() >= SETTLE_S:
            self._recycle_end(name)
            self._log_recycle(name, "done")
            return None
        return job

    def _recycle_trigger(self, name: str, now: datetime) -> _Recycle | None:
        """Start a restart when a limit is crossed. Off unless a limit is set; a commit
        shortage only counts when HD-Player holds enough that a restart would give it back,
        and a failed commit read is inf, which crosses nothing."""
        if self._recycle_gb <= 0 and self._commit_floor_gb <= 0:
            return None
        last = self._recycled_at.get(name)
        if last is not None and (now - last).total_seconds() < MIN_INTERVAL_S:
            return None
        player = self._find_player(name)
        private = None if player is None else self._private_gb(player)
        if private is None:
            return None
        by_memory = 0 < self._recycle_gb <= private
        by_commit = self._commit_short() and private >= MIN_RECYCLE_GB
        if not (by_memory or by_commit):
            return None
        job = self._recycling[name] = _Recycle(_Phase.STOPPING, now)
        self._recycled_at[name] = now
        self._recycle_write(name, RECYCLE_LAST_FILE, {"last_restart": now.isoformat()})
        self._recycle_save(name, job)
        self._log_recycle(name, f"stopping ({'memory' if by_memory else 'commit'} limit)")
        return job

    def _commit_short(self) -> bool:
        """True when free commit is under the floor. A failed read is inf, which never is, and
        that leaves this limit off: said once, and once more if it breaks again later."""
        if self._commit_floor_gb <= 0:
            return False
        free = self._commit_free_gb()
        failed = math.isinf(free)
        if failed and not self._commit_warned:
            log.warning("commit charge read failed; commit_floor_gb is off until it works again")
        self._commit_warned = failed
        return free < self._commit_floor_gb

    def _recycle_close(self, name: str, job: _Recycle, now: datetime) -> None:
        """The worker has exited. close_player() does not wait out a terminate, so the phase
        moves on to confirm the close on later ticks. Already gone is just confirmed."""
        self._log_recycle(name, "closing")
        player = self._find_player(name)
        if player is None or self._close_player(player, grace_s=CLOSE_GRACE_S, log=log.warning):
            job.phase, job.since = _Phase.CLOSING, now
            self._recycle_save(name, job)
        else:
            self._recycle_failed(name, job, "closing")

    def _recycle_boot(self, inst: S.InstanceSettings, job: _Recycle, now: datetime) -> None:
        name = inst.name
        if not job.opened:
            if self._open_player(self._emulator_path, name, log=log.warning) is None:
                self._recycle_failed(name, job, "opening")
                return
            job.opened, job.since = True, now
            self._log_recycle(name, "booting")
        elif self._boot_done(self.settings.connection.adb_path, inst.adb_port):
            job.phase, job.since = _Phase.SETTLING, now
            self._recycle_save(name, job)
            self._log_recycle(name, "settling")
        elif (now - job.since).total_seconds() >= BOOT_TIMEOUT_S:
            self._recycle_failed(name, job, "booting")

    def _recycle_failed(self, name: str, job: _Recycle, step: str) -> None:
        """The one alert for this restart. "offline" because that is the kind the default
        notification events and every saved list carry; the fields say why, and the reason
        gives it its own title (notify.alert_title). The phase changes first and nothing here
        raises, so the caller can never land here a second time for the same failure."""
        job.phase = _Phase.FAILED
        self._recycle_save(name, job)
        fields = {"reason": RECYCLE_FAILED_REASON, "step": step}
        try:
            notify.maybe_alert("offline", {"instance": name, **fields})
        except Exception as exc:  # the panel's copy below still goes out
            log.debug("failed restart alert: %s", exc)
        self._fire_alert(name, "offline", fields)
        self._log_recycle(name, f"failed while {step}", logging.ERROR)

    def _log_recycle(
        self, name: str, step: str, level: int = logging.INFO, exc: Exception | None = None
    ) -> None:
        """Never raises: the readings are only colour on the line, and they call the same
        helpers whose failure may be the very thing being logged."""
        try:
            player = self._find_player(name)
            private = None if player is None else self._private_gb(player)
            shown = "n/a" if private is None else f"{private:.1f} GB"
            detail = f"HD-Player private {shown}, commit free {self._commit_free_gb():.1f} GB"
        except Exception as read_exc:
            detail = f"readings failed: {read_exc!r}"
        try:
            log.log(level, "%s: emulator recycle %s (%s)", name, step, detail, exc_info=exc)
        except Exception:
            pass

    # --- the emulator guard's files ---------------------------------------------------

    def _recycle_write(self, name: str, filename: str, data: dict) -> None:
        """Best effort, like status.json: a failed write costs the resume, never the restart."""
        try:
            atomic_write_json(self._dir(name) / filename, data)
        except OSError as exc:
            log.warning("%s: could not write %s: %s", name, filename, exc)

    def _recycle_unlink(self, name: str, filename: str) -> None:
        try:
            (self._dir(name) / filename).unlink(missing_ok=True)
        except OSError as exc:
            log.warning("%s: could not remove %s: %s", name, filename, exc)

    def _recycle_save(self, name: str, job: _Recycle) -> None:
        data = {"phase": job.phase.value, "since": job.since.isoformat()}
        self._recycle_write(name, RECYCLE_FILE, data)

    def _recycle_end(self, name: str) -> None:
        self._recycling.pop(name, None)
        self._recycle_unlink(name, RECYCLE_FILE)

    def _recycle_read(self, name: str, filename: str, parse: Callable[[dict], Any]) -> Any:
        """``parse`` applied to one saved file. None when the file is missing, or logged and
        None when it cannot be understood: a bad file must never stop the supervisor."""
        try:
            return parse(json.loads((self._dir(name) / filename).read_text(encoding="utf-8")))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("%s: ignoring unreadable %s: %r", name, filename, exc)
            return None

    def _recycle_load(self, name: str) -> None:
        """What the last supervisor left, read once at startup before any tick. A time ahead
        of the clock is taken as now, so a clock set back cannot stretch a limit."""
        now = self._clock()

        def stamp(value: str) -> datetime:
            return min(datetime.fromisoformat(value), now)

        last = self._recycle_read(name, RECYCLE_LAST_FILE, lambda d: stamp(d["last_restart"]))
        if last is not None:
            self._recycled_at[name] = last
        job = self._recycle_read(
            name,
            RECYCLE_FILE,
            lambda d: _Recycle(_Phase(d["phase"]), stamp(d["since"]), resumed=True),
        )
        if job is not None:
            self._recycling[name] = job

    def _recycle_resume(self, name: str, job: _Recycle, now: datetime, want: str) -> bool:
        """The first step after a supervisor restart; False when it dropped the job. STOPPING
        and FAILED carry on as they were. Past that the job is dropped if a stop is wanted or
        it is too old to belong to the same run, say after a power cut: the emulator stays as
        it is and the tick runs as normal, and RECYCLE_LAST_FILE keeps the interval limit.
        Otherwise nobody was watching, so HD-Player itself says where the job stands: not
        listed means closed or dead, and the boot step opens it. Listed in BOOTING means it
        was opened, and it gets a full boot window from here rather than failing for the time
        nobody checked. CLOSING goes on confirming and SETTLING keeps its clock."""
        job.resumed = False
        if job.phase in (_Phase.STOPPING, _Phase.FAILED):
            return True
        stale = (now - job.since).total_seconds() > BOOT_TIMEOUT_S + SETTLE_S
        if want == "stop" or stale:
            self._recycle_end(name)
            why = "a stop is wanted" if want == "stop" else "too old to resume"
            self._log_recycle(name, f"dropped while {job.phase}, {why}", logging.WARNING)
            return False
        self._log_recycle(name, f"resumed while {job.phase}")
        if self._find_player(name) is None:
            job.phase, job.since, job.opened = _Phase.BOOTING, now, False
            self._recycle_save(name, job)
        elif job.phase == _Phase.BOOTING:
            job.since, job.opened = now, True
        return True

    def _recycle_prune(self) -> None:
        """Forget instances no longer in the settings, whether they left just now or while
        the supervisor was down. A job past STOPPING is dropped where it stands: a closed or
        half-booted HD-Player stays that way, because nothing owns it any more."""
        configured = {i.name for i in self.settings.instances}
        on_disk = {
            path.parent.name
            for filename in (RECYCLE_FILE, RECYCLE_LAST_FILE)
            for path in self.home.glob(f"instances/*/{filename}")
        }
        for name in {*self._recycling, *self._recycled_at, *on_disk} - configured:
            if name in self._recycling or (self._dir(name) / RECYCLE_FILE).exists():
                log.warning(
                    "%s: left the settings during an emulator restart; dropped it, "
                    "HD-Player stays as it is",
                    name,
                )
            self._recycle_end(name)
            self._recycled_at.pop(name, None)
            self._recycle_unlink(name, RECYCLE_LAST_FILE)

    # --- controls (called from the event loop thread; phase 3 wires them to the API) ---

    def start(self, name: str, hours: float | None = None) -> None:
        """Start (or undo a pending stop): clear the stop override, or run for N hours."""
        now = self._clock()
        self.settings.instance(name)
        if hours:
            scheduler.write_override(name, "run", now + timedelta(hours=hours))
        else:
            scheduler.clear_override(name)
        self._flag(name).unlink(missing_ok=True)
        self._stop_asked.pop(name, None)
        self._backoff.clear(name)
        job = self._recycling.get(name)
        if job is not None and job.phase == _Phase.FAILED:
            self._recycle_end(name)
        self.poke()

    def stop(self, name: str) -> None:
        """Graceful: schedule override to stop, stop.flag now; the tick kills after 110 s."""
        now = self._clock()
        self.settings.instance(name)
        scheduler.write_override(name, "stop", now + timedelta(days=STOP_OVERRIDE_DAYS))
        self._request_stop(name, now)
        self.poke()

    def observe(self, name: str, on: bool) -> None:
        """Observe mode on: an observe override, so the next tick launches the worker with
        --observe. Off: the ordinary graceful stop, because an observer that is no longer
        wanted is just a worker to stop."""
        now = self._clock()
        self.settings.instance(name)
        if not on:
            self.stop(name)
            return
        scheduler.write_override(name, "observe", now + timedelta(days=STOP_OVERRIDE_DAYS))
        self._flag(name).unlink(missing_ok=True)
        self._stop_asked.pop(name, None)
        self._backoff.clear(name)
        self.poke()

    def stop_now(self, name: str) -> bool:
        """Kill by PID immediately. Only while a graceful stop is pending; else no-op."""
        if name not in self._stop_asked:
            return False
        st = status.read_status(self._dir(name))
        pid = _pid_of(st)
        if pid is None:
            return False
        ok = bool(self._kill(pid, log=log.warning))
        self._stop_asked.pop(name, None)
        self.poke()
        return ok

    def restart(self, name: str) -> None:
        """Stop gracefully without touching the schedule; the next tick relaunches."""
        now = self._clock()
        self.settings.instance(name)
        st = status.read_status(self._dir(name))
        pid = _pid_of(st)
        if pid is not None and self._alive(pid):
            self._request_stop(name, now)
        self.poke()

    def retry_now(self, name: str) -> None:
        self._backoff.clear(name)
        self.poke()

    def poke(self) -> None:
        if self._poke is not None:
            self._poke.set()

    # --- the loop ---------------------------------------------------------------------

    def request_shutdown(self) -> None:
        self._shutdown = True
        self.poke()

    async def run_forever(self) -> None:
        """Tick, then sleep interval_s or until poked. Workers keep running when this
        returns; the next start reattaches through status.json PIDs."""
        self._poke = asyncio.Event()
        self._shutdown = False
        while not self._shutdown:
            try:
                await asyncio.to_thread(self.tick)
            except Exception:
                log.exception("tick failed")
            try:
                await asyncio.wait_for(self._poke.wait(), timeout=self.interval_s)
            except TimeoutError:
                pass
            self._poke.clear()
        log.info("supervisor loop stopped; workers keep running")


def _parse_until(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _int_or_none(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _pid_of(st: dict | None) -> int | None:
    """The worker PID from status.json. A missing, junk or non-positive value is None: a
    hand-edited or half-written file must never abort the tick."""
    pid = _int_or_none((st or {}).get("pid"))
    return pid if pid and pid > 0 else None


def _error_view(inst: S.InstanceSettings) -> InstanceView:
    """The card for an instance whose tick raised before it produced one, and which has no
    previous view to fall back on."""
    return InstanceView(
        name=inst.name,
        adb_port=inst.adb_port,
        state=InstanceState.STOPPED,
        health=Health.DEAD,
        pid=None,
        heartbeat_age_s=None,
        phase=None,
        desired="run",
        desired_reason=None,
        until=None,
        games_played=None,
        farm_brawler=None,
        note="Supervisor error; see supervisor.log",
    )


def _offline_note(retry: datetime, now: datetime) -> str:
    minutes = max(1, round((retry - now).total_seconds() / 60))
    return f"BlueStacks window not found. Retrying in {minutes} min."
