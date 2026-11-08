"""Quiet bush (config.BUSH_QUIET): hide hysteresis, gas smoothing, the out-and-back
keepalive, held fire while hidden and the found window. adb.swipe, the bush detector
and the HUD reader are stubbed, so no input is sent and no frame is read."""

import math

import pytest

from brawlfarm.core import config
from brawlfarm.core import controller as controller_mod
from brawlfarm.core import match_vision as mv
from brawlfarm.core.controller import Controller
from brawlfarm.core.states import State
from brawlfarm.play import reader

SX, SY = config.BUSH_SELF_POS
NEAR = mv.Bush(SX + 40, SY, 500)  # inside BUSH_AT_TARGET_PX
OTHER = mv.Bush(SX - 60, SY, 500)  # also near self, but a different bush
FAR = mv.Bush(SX + 400, SY, 500)
SCREEN = object()


class _DL:
    def __init__(self):
        self.events = []

    def event(self, etype, **fields):
        self.events.append((etype, fields))

    def kinds(self):
        return [e for e, _ in self.events]


def _ctrl(monkeypatch, clusters, health=()):
    """A controller with quiet bush on. ``clusters`` is a list of per-tick cluster lists
    (the last repeats); ``health`` scripts the reader's self_health_fraction the same way."""
    monkeypatch.setattr(config, "BUSH_HIDE", True)
    monkeypatch.setattr(config, "BUSH_QUIET", True)
    monkeypatch.setattr(controller_mod.random, "uniform", lambda low, high: low)
    c = Controller.__new__(Controller)
    c.dl = _DL()
    c.logs = []
    c.log = c.logs.append
    c._gas_edges = set()
    c._relocations = 0
    c._heading = 0.0
    c._bush_logged = None
    c._last_move_heading = c._last_move_target = None
    c._reset_quiet_bush()
    c.swipes = []
    monkeypatch.setattr(controller_mod.adb, "swipe", lambda *a: c.swipes.append(a))
    ticks = list(clusters)
    monkeypatch.setattr(
        mv, "bush_clusters", lambda _f: ticks.pop(0) if len(ticks) > 1 else ticks[0]
    )
    reads = list(health) or [None]
    monkeypatch.setattr(
        reader,
        "read",
        lambda _f: reader.Reading(
            self_health_fraction=reads.pop(0) if len(reads) > 1 else reads[0]
        ),
    )
    return c


def _closing(monkeypatch, c):
    """Gas reads 'closing' on two edges from now on."""
    monkeypatch.setattr(config, "GAS_AWARE", True)
    c._gas_edges = {"top", "left"}


# --- hide hysteresis ----------------------------------------------------------------


def test_enters_hidden_within_the_old_radius_and_logs_it(monkeypatch) -> None:
    c = _ctrl(monkeypatch, [[NEAR]])
    assert c._quiet_bush_move(SCREEN, None, 10.0) == "bush_quiet_hold"
    assert c._quiet_target == (NEAR.cx, NEAR.cy)
    assert ("bush_hide", {"target": [NEAR.cx, NEAR.cy], "quiet": True}) in c.dl.events
    assert c.swipes == []  # holding: the keepalive is not due on the entry tick


def test_hidden_survives_one_missed_read_and_leaves_after_n(monkeypatch) -> None:
    misses = [[]] * config.BUSH_QUIET_MISS_TICKS
    c = _ctrl(monkeypatch, [[NEAR], [], [NEAR], *misses])
    assert c._quiet_bush_move(SCREEN, None, 10.0) == "bush_quiet_hold"
    assert c._quiet_bush_move(SCREEN, None, 11.0) == "bush_quiet_hold"  # one miss: still hidden
    assert c._quiet_bush_move(SCREEN, None, 12.0) == "bush_quiet_hold"  # back: miss count resets
    for i in range(config.BUSH_QUIET_MISS_TICKS - 1):
        assert c._quiet_bush_move(SCREEN, None, 13.0 + i) == "bush_quiet_hold"
    assert c._quiet_bush_move(SCREEN, None, 20.0) == "wander"
    assert c._quiet_target is None
    leave = [f for e, f in c.dl.events if e == "bush_leave"]
    assert leave == [{"reason": "lost", "held_s": 10.0, "quiet": True}]


def test_stays_hidden_past_the_enter_radius_and_leaves_beyond_the_leave_radius(
    monkeypatch,
) -> None:
    mid = mv.Bush(SX + 130, SY, 500)  # outside 90, inside 150: no churn
    gone = mv.Bush(SX + 200, SY, 500)  # beyond 150 (and within tracking of mid)
    c = _ctrl(monkeypatch, [[NEAR], [mv.Bush(SX + 100, SY, 500)], [mid], [gone]])
    assert c._quiet_bush_move(SCREEN, None, 1.0) == "bush_quiet_hold"
    assert c._quiet_bush_move(SCREEN, None, 2.0) == "bush_quiet_hold"
    assert c._quiet_bush_move(SCREEN, None, 3.0) == "bush_quiet_hold"
    assert c._quiet_bush_move(SCREEN, None, 4.0) == "hide"  # left, now approaching it again
    assert [f["reason"] for e, f in c.dl.events if e == "bush_leave"] == ["drift"]


def test_does_not_switch_target_while_hidden(monkeypatch) -> None:
    closer = mv.Bush(SX + 5, SY, 500)
    c = _ctrl(monkeypatch, [[NEAR], [NEAR, closer]])
    c._quiet_bush_move(SCREEN, None, 1.0)
    c._quiet_bush_move(SCREEN, None, 2.0)
    assert c._quiet_target == (NEAR.cx, NEAR.cy)


# --- gas smoothing ------------------------------------------------------------------


def test_gas_relocates_only_after_k_ticks_then_honours_the_cooldown(monkeypatch) -> None:
    c = _ctrl(monkeypatch, [[NEAR, OTHER]])
    assert c._quiet_bush_move(SCREEN, None, 1.0) == "bush_quiet_hold"
    _closing(monkeypatch, c)
    t = 2.0
    for _ in range(config.BUSH_QUIET_GAS_TICKS - 1):
        assert c._quiet_bush_move(SCREEN, None, t) == "bush_quiet_hold"  # not confirmed yet
        t += 1.0
    assert c._quiet_bush_move(SCREEN, None, t) == "relocate"
    assert c._relocations == 1
    assert [f["reason"] for e, f in c.dl.events if e == "bush_leave"] == ["gas"]
    assert c.dl.kinds().count("gas_relocate") == 1
    # Still closing, but inside the cooldown: no second relocate.
    assert c._quiet_bush_move(SCREEN, None, t + 1.0) != "relocate"
    assert c._quiet_bush_move(SCREEN, None, t + config.BUSH_QUIET_RELOCATE_COOLDOWN - 0.1) != (
        "relocate"
    )
    assert c._quiet_bush_move(SCREEN, None, t + config.BUSH_QUIET_RELOCATE_COOLDOWN) == "relocate"
    assert c._relocations == 2


def test_a_flicker_resets_the_gas_count(monkeypatch) -> None:
    c = _ctrl(monkeypatch, [[NEAR, OTHER]])
    for i in range(config.BUSH_QUIET_GAS_TICKS - 1):
        _closing(monkeypatch, c)
        c._quiet_bush_move(SCREEN, None, float(i))
    c._gas_edges = set()  # one clear read
    c._quiet_bush_move(SCREEN, None, 10.0)
    _closing(monkeypatch, c)
    assert c._quiet_bush_move(SCREEN, None, 11.0) != "relocate"


def test_the_relocation_cap_still_holds(monkeypatch) -> None:
    c = _ctrl(monkeypatch, [[NEAR, OTHER]])
    _closing(monkeypatch, c)
    c._relocations = config.BUSH_MAX_RELOCATIONS
    c._quiet_gas_streak = config.BUSH_QUIET_GAS_TICKS
    assert c._quiet_bush_move(SCREEN, None, 100.0) == "wander"
    assert "gas_relocate" not in c.dl.kinds()


# --- keepalive ----------------------------------------------------------------------


def test_keepalive_alternates_out_and_back_and_returns(monkeypatch) -> None:
    c = _ctrl(monkeypatch, [[NEAR]])
    c._quiet_bush_move(SCREEN, None, 0.0)  # enter; first keepalive due BUSH_JITTER_MIN later
    t = 0.0
    while len(c.swipes) < 4:
        t += config.MOVE_INTERVAL_MIN
        assert c._quiet_bush_move(SCREEN, None, t) == "bush_quiet_hold"
        assert t <= 4 * config.BUSH_JITTER_MAX_INTERVAL
    ox, oy = config.MOVE_ORIGIN
    for x1, y1, x2, y2, dur in c.swipes:
        assert (x1, y1) == (ox, oy)
        assert math.hypot(x2 - ox, y2 - oy) == pytest.approx(config.BUSH_QUIET_JITTER_PX, abs=1)
        assert dur == config.BUSH_QUIET_JITTER_MS
    out, back = c.swipes[0], c.swipes[1]
    assert (out[2] - ox + back[2] - ox, out[3] - oy + back[3] - oy) == (0, 0)
    assert c.swipes[2] == out and c.swipes[3] == back


def test_keepalive_is_sent_within_the_jitter_interval(monkeypatch) -> None:
    c = _ctrl(monkeypatch, [[NEAR]])
    monkeypatch.setattr(controller_mod.random, "uniform", lambda low, high: high)  # 8 s gap
    c._quiet_bush_move(SCREEN, None, 0.0)
    c._quiet_bush_move(SCREEN, None, config.BUSH_JITTER_MAX_INTERVAL)
    assert len(c.swipes) == 1


# --- held fire and the found window -------------------------------------------------


def _playing(monkeypatch, c, clock):
    monkeypatch.setattr(config, "GAS_AWARE", False)
    monkeypatch.setattr(config, "INGAME_MODAL_CHECK_EVERY", 1000)
    monkeypatch.setattr(config, "ABILITY_BUTTONS_ENABLED", True)
    name = next(iter(config.ABILITY_BUTTONS))
    monkeypatch.setattr(mv, "ready_abilities", lambda _s: {name: 0.9})
    monkeypatch.setattr(controller_mod.time, "monotonic", lambda: clock[0])
    c.phase = "playing"
    c.phase_started = clock[0]
    c._playing_modal_i = 0
    c._spectate_off = True
    c.last_move = c.last_attack = -100.0
    c._move_gap = 0.0
    c._attack_gap = 0.0
    c._ability_last = {}
    c._last_move_kind = None
    c.taps = []
    c.tap = lambda point, kind=None: c.taps.append((point, kind))
    c._jitter = lambda p: p


def test_no_attack_super_or_ability_taps_while_hidden(monkeypatch) -> None:
    clock = [1000.0]
    c = _ctrl(monkeypatch, [[NEAR]], health=[0.8])
    _playing(monkeypatch, c, clock)
    for _ in range(5):
        c.phase_playing(SCREEN, State.IN_MATCH)
        clock[0] += 3.0
        c._move_gap = c._attack_gap = 0.0
    assert c._last_move_kind == "bush_quiet_hold"
    assert c.taps == []


def test_fires_normally_when_not_hidden(monkeypatch) -> None:
    clock = [1000.0]
    c = _ctrl(monkeypatch, [[FAR]])
    _playing(monkeypatch, c, clock)
    c.phase_playing(SCREEN, State.IN_MATCH)
    assert c._last_move_kind == "hide"
    assert (config.ATTACK_POINT, None) in c.taps
    assert (config.SUPER_BUTTON, None) in c.taps
    assert any(kind and kind.startswith("ability_") for _, kind in c.taps)


def test_a_health_drop_opens_a_fire_window_that_ends(monkeypatch) -> None:
    clock = [1000.0]
    c = _ctrl(monkeypatch, [[NEAR]], health=[0.9, None, 0.9, 0.5, 0.5])
    _playing(monkeypatch, c, clock)

    def tick():
        c.taps.clear()
        c._move_gap = c._attack_gap = 0.0
        c._ability_last = {}
        c.phase_playing(SCREEN, State.IN_MATCH)
        clock[0] += 1.0
        return [k for _, k in c.taps]

    assert tick() == []  # enter, baseline 0.9
    assert tick() == []  # None is unknown, not a drop
    assert tick() == []  # 0.9 again
    assert len(tick()) == 3  # 0.5: found -> attack, super, ability
    assert c.dl.kinds().count("bush_found") == 1
    clock[0] += config.BUSH_QUIET_FOUND_S
    assert tick() == []  # window over, health steady: hold fire again
    assert c.dl.kinds().count("bush_found") == 1


# --- switch off ---------------------------------------------------------------------


def test_switch_off_never_runs_the_quiet_tick(monkeypatch) -> None:
    clock = [1000.0]
    c = _ctrl(monkeypatch, [[NEAR]])
    monkeypatch.setattr(config, "BUSH_QUIET", False)
    _playing(monkeypatch, c, clock)
    c._quiet_bush_move = lambda *a: pytest.fail("quiet tick ran with the switch off")
    c._quiet_target = (NEAR.cx, NEAR.cy)  # even a stale hidden state does not hold fire
    c._bush_jitter_at = 0.0
    c.phase_playing(SCREEN, State.IN_MATCH)
    assert c._last_move_kind == "hide"
    assert (config.ATTACK_POINT, None) in c.taps
