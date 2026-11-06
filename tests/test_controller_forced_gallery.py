"""Forced shooting gallery and the recover() ceremony budget (live farm bug, 2026-11-05).

A launch opened a fullscreen gallery whose EXIT sign is dead. phase_returning tapped its
safe point, pressed BACK and ran recover("stuck_returning") about every 2 minutes, while
about 10 taps on the two cannons cleared the screen. These tests pin the fix:

  1. The detector needs both of its color gates.
  2. phase_returning alternates the cannons, one tap per pass, up to GALLERY_MAX_ROUNDS,
     then falls through to the existing drop/BACK/recover ladder.
  3. recover() gives a relaunched game a fresh ceremony budget, and MAX_RECOVERY still
     ends the loop.

All offline: Controller.__new__ + minimal attrs (the test_network_stuck.py pattern).
"""

from __future__ import annotations

import time

import numpy as np

from brawlfarm.core import config, states
from brawlfarm.core import controller as controller_mod
from brawlfarm.core.controller import (
    BACK_TRIES,
    DROP_TAPS,
    MAX_RECOVERY,
    RETURN_PATIENCE,
    Controller,
)
from brawlfarm.core.states import State

SCREEN = np.zeros((900, 1600, 3), dtype=np.uint8)


class _DL:
    def __init__(self):
        self.events = []

    def event(self, etype, **fields):
        self.events.append((etype, fields))


def _ctrl(monkeypatch):
    c = Controller.__new__(Controller)  # skip __init__ (ApiClient/DataLog/adb)
    c.dl = _DL()
    c.phase = "returning"
    c.phase_started = time.monotonic()
    c.games_played = 0
    c.recovery_attempts = 0
    c.unknown_clears = 0
    c._ceremony_count = 0
    c._gallery_rounds = 0
    c._dismiss_tried = True  # skip the soft dismiss ladder
    c._update_checked = True  # skip the Play Store check
    c._stop_flag_seen = False
    c._soft_stop_since = None
    c.max_games = None
    c.max_minutes = None
    c.start = time.monotonic()
    c.running = True
    c.last_change = 0.0
    c.taps = []
    c.keys = []
    c.stopped = []
    monkeypatch.setattr(
        Controller, "tap", lambda self, point, kind=None: self.taps.append((point, kind))
    )
    monkeypatch.setattr(Controller, "log", lambda self, msg: None)
    monkeypatch.setattr(Controller, "event_shot", lambda self, *a, **k: None)
    monkeypatch.setattr(Controller, "set_phase", lambda self, name: None)
    monkeypatch.setattr(Controller, "stop", lambda self, reason: self.stopped.append(reason))
    monkeypatch.setattr(Controller, "_handle_skin_popup", lambda self, s: False)
    monkeypatch.setattr(Controller, "_handle_ceremony", lambda self, s: False)
    monkeypatch.setattr(controller_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(controller_mod.adb, "keyevent", lambda code: c.keys.append(code))
    monkeypatch.setattr(controller_mod.adb, "screencap", lambda: None)
    monkeypatch.setattr(controller_mod.adb, "go_home", lambda: None)
    monkeypatch.setattr(controller_mod.adb, "force_stop", lambda: None)
    monkeypatch.setattr(controller_mod.adb, "launch_app", lambda: None)
    return c


def _gallery_up(monkeypatch):
    monkeypatch.setattr(controller_mod.states, "is_forced_gallery", lambda s: True)


def _frame(*, wall, sign):
    frame = SCREEN.copy()
    if wall:
        frame[:80, :] = (30, 40, 255)  # BGR, the bright red wall
    if sign:
        frame[90:150, 290:410] = (80, 230, 250)  # BGR, the gold EXIT plate
    return frame


def test_detector_needs_both_gates():
    assert states.is_forced_gallery(_frame(wall=True, sign=True))
    assert not states.is_forced_gallery(_frame(wall=True, sign=False))
    assert not states.is_forced_gallery(_frame(wall=False, sign=True))
    assert not states.is_forced_gallery(SCREEN)


def test_alternates_both_cannons_one_tap_per_pass(monkeypatch):
    c = _ctrl(monkeypatch)
    _gallery_up(monkeypatch)
    c.unknown_clears = 5
    for _ in range(4):
        c.phase_returning(SCREEN, State.UNKNOWN)
    left = (config.GALLERY_CANNON_LEFT, "gallery_cannon")
    right = (config.GALLERY_CANNON_RIGHT, "gallery_cannon")
    assert c.taps == [left, right, left, right]
    assert ("forced_gallery", {"round": 4}) in c.dl.events
    assert c.unknown_clears == 0


def test_cap_stops_the_taps_and_falls_through(monkeypatch):
    c = _ctrl(monkeypatch)
    _gallery_up(monkeypatch)
    recovered = []
    monkeypatch.setattr(Controller, "recover", lambda self, reason: recovered.append(reason))
    for _ in range(config.GALLERY_MAX_ROUNDS):
        c.phase_returning(SCREEN, State.UNKNOWN)
    assert len(c.taps) == config.GALLERY_MAX_ROUNDS
    c.unknown_clears = RETURN_PATIENCE + DROP_TAPS
    c.phase_returning(SCREEN, State.UNKNOWN)
    assert c.keys == [4]  # the existing BACK step, not another cannon tap
    c.unknown_clears = RETURN_PATIENCE + DROP_TAPS + BACK_TRIES
    c.phase_returning(SCREEN, State.UNKNOWN)
    assert recovered == ["stuck_returning"]
    assert len(c.taps) == config.GALLERY_MAX_ROUNDS


def test_recover_resets_the_ceremony_count_and_max_recovery_still_ends_it(monkeypatch):
    c = _ctrl(monkeypatch)
    c._ceremony_count = config.CEREMONY_MAX_SCREENS
    c.recover("stuck_returning")
    assert c._ceremony_count == 0
    assert c.stopped == []
    for _ in range(MAX_RECOVERY):
        c._ceremony_count = config.CEREMONY_MAX_SCREENS
        c.recover("stuck_returning")
    assert c.stopped == ["too_many_recoveries:stuck_returning"]
