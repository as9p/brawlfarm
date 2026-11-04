"""recover() checks the Play Store for a game update once per run, on the second
recovery. Offline: Controller.__new__ plus minimal attributes (the test_network_stuck.py
pattern), with try_update replaced by a fake."""

from __future__ import annotations

import time

from brawlfarm.core import controller as controller_mod
from brawlfarm.core import game_update
from brawlfarm.core.controller import Controller
from brawlfarm.core.game_update import Outcome, UpdateResult


class _DL:
    def __init__(self):
        self.events = []

    def event(self, etype, **fields):
        self.events.append((etype, fields))


def _ctrl(monkeypatch, outcome: Outcome):
    c = Controller.__new__(Controller)  # skip __init__ (ApiClient/DataLog/adb)
    c.dl = _DL()
    c.phase = "returning"
    c.phase_started = time.monotonic()
    c.games_played = 0
    c.recovery_attempts = 0
    c._dismiss_tried = True  # skip the soft dismiss ladder
    c._update_checked = False
    c._stop_flag_seen = False
    c._soft_stop_since = None
    c.max_games = None
    c.max_minutes = None
    c.start = time.monotonic()
    c.running = True
    c.last_change = 0.0
    c.calls = []
    c.stopped = []
    c.update_calls = 0

    def fake_try_update(fx):
        c.update_calls += 1
        return outcome

    monkeypatch.setattr(controller_mod.game_update, "try_update", fake_try_update)
    monkeypatch.setattr(Controller, "log", lambda self, msg: None)
    monkeypatch.setattr(Controller, "event_shot", lambda self, *a, **k: None)
    monkeypatch.setattr(Controller, "set_phase", lambda self, name: None)
    monkeypatch.setattr(Controller, "stop", lambda self, reason: self.stopped.append(reason))
    monkeypatch.setattr(controller_mod.adb, "screencap", lambda: None)
    monkeypatch.setattr(controller_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(controller_mod.adb, "go_home", lambda: c.calls.append("home"))
    monkeypatch.setattr(controller_mod.adb, "force_stop", lambda: c.calls.append("force_stop"))
    monkeypatch.setattr(controller_mod.adb, "launch_app", lambda: c.calls.append("launch"))
    return c


def _kinds(c):
    return [e for e, _ in c.dl.events]


def test_first_recovery_does_not_check_the_store(monkeypatch):
    c = _ctrl(monkeypatch, Outcome(UpdateResult.NO_UPDATE, "no_update_button"))
    c.recover("stuck_returning")
    assert c.update_calls == 0
    assert c.calls == ["home", "force_stop", "launch"]


def test_second_recovery_checks_once_then_relaunches(monkeypatch):
    c = _ctrl(monkeypatch, Outcome(UpdateResult.NO_UPDATE, "no_update_button", old="1.0"))
    c.recovery_attempts = 1
    c.recover("stuck_returning")
    assert c.update_calls == 1
    assert "update_check" in _kinds(c)
    assert c.calls == ["home", "force_stop", "launch"]
    assert c.stopped == []
    c.recover("stuck_returning")  # attempt 3: no second store visit this run
    assert c.update_calls == 1


def test_updated_resets_recovery_budget_and_relaunches(monkeypatch):
    c = _ctrl(monkeypatch, Outcome(UpdateResult.UPDATED, "installed", old="1.0", new="2.0"))
    c.recovery_attempts = 1
    c.recover("stuck_returning")
    assert c.recovery_attempts == 0
    assert ("game_updated", {"old": "1.0", "new": "2.0"}) in c.dl.events
    assert c.calls == ["home", "force_stop", "launch"]


def test_failed_update_stops_the_run_without_relaunch(monkeypatch):
    c = _ctrl(monkeypatch, Outcome(UpdateResult.FAILED, "timeout", old="1.0"))
    c.recovery_attempts = 1
    c.recover("stuck_returning")
    assert c.stopped == ["update_failed"]
    assert ("update_failed", {"reason": "timeout"}) in c.dl.events
    assert c.calls == ["home"]  # left the store, no force-stop or relaunch


def test_try_update_crash_counts_as_failed(monkeypatch):
    c = _ctrl(monkeypatch, Outcome(UpdateResult.NO_UPDATE, "x"))

    def boom(fx):
        raise RuntimeError("ocr model missing")

    monkeypatch.setattr(controller_mod.game_update, "try_update", boom)
    c.recovery_attempts = 1
    c.recover("stuck_returning")
    assert c.stopped == ["update_failed"]
    assert ("update_failed", {"reason": "error: RuntimeError"}) in c.dl.events
    assert c.calls == ["home"]  # the store is not left in front after a crash


def test_effects_point_at_the_game_package(monkeypatch):
    c = _ctrl(monkeypatch, Outcome(UpdateResult.NO_UPDATE, "x"))
    opened, versions = [], []
    monkeypatch.setattr(controller_mod.adb, "open_store_page", lambda pkg: opened.append(pkg))
    monkeypatch.setattr(
        controller_mod.adb, "package_version", lambda pkg: versions.append(pkg) or "1.0"
    )
    fx = c._game_update_effects()
    assert isinstance(fx, game_update.Effects)
    fx.open_store()
    assert fx.version() == "1.0"
    assert opened == versions == [controller_mod.config.BS_PACKAGE]
