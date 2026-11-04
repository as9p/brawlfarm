"""Daily Streak popup and the UNKNOWN-path repeat guard (live farm bug, 2026-11-04).

The bot sat 40 minutes on a day-1 Daily Streak popup that has no CLAIM button,
tapping the fixed CLAIM point 1442 times: the handler tapped blind, and its
`handle_x(); continue` branch skipped every loop guard. These tests pin the fix:

  1. handle_daily_streak verifies before it acts: tap the OCR'd CLAIM word, else
     the popup's own ✕ (local threshold, top-right area only), else one BACK.
  2. Every handler branch on the UNKNOWN path is counted; HANDLER_REPEAT_MAX
     consecutive runs of the same one escalate to recover("<name>_stuck").

All offline: Controller.__new__ + minimal attrs (the test_network_stuck.py pattern).
"""

from __future__ import annotations

import numpy as np

from brawlfarm.core import config
from brawlfarm.core import controller as controller_mod
from brawlfarm.core.controller import Controller
from brawlfarm.core.states import State
from brawlfarm.core.vision import Match

SCREEN = np.zeros((900, 1600, 3), dtype=np.uint8)


class _DL:
    def __init__(self):
        self.events = []

    def event(self, etype, **fields):
        self.events.append((etype, fields))


def _ctrl(monkeypatch):
    c = Controller.__new__(Controller)  # skip __init__ (ApiClient/DataLog/adb)
    c.dl = _DL()
    c.phase = "menu"
    c._repeat_handler = None
    c._repeat_count = 0
    c.taps = []
    c.keys = []
    c.phases = []
    c.recovered = []
    c.handled = []
    monkeypatch.setattr(
        Controller, "tap", lambda self, point, kind=None: self.taps.append((point, kind))
    )
    monkeypatch.setattr(Controller, "set_phase", lambda self, name: self.phases.append(name))
    monkeypatch.setattr(Controller, "recover", lambda self, reason: self.recovered.append(reason))
    monkeypatch.setattr(Controller, "log", lambda self, msg: None)
    monkeypatch.setattr(controller_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(controller_mod.adb, "keyevent", lambda code: c.keys.append(code))
    return c


def _vision(monkeypatch, *, claim=None, close_x=None):
    """Fake OCR and template match. `claim` is the CLAIM word centre (or None);
    `close_x` is the close_x Match find() returns at the streak's local threshold."""

    def find_text(screen, needle, region=None, scale=2.0, exact=False):
        if needle.upper() == "CLAIM" and region == config.DAILY_STREAK_CLAIM_REGION:
            return claim
        return None

    def find(screen, name, threshold=None):
        if name == "close_x" and threshold == config.DAILY_STREAK_CLOSE_X_THRESHOLD:
            return close_x
        return None

    monkeypatch.setattr(controller_mod.vision, "find_text", find_text)
    monkeypatch.setattr(controller_mod.vision, "find", find)


def _x_at(x, y, score=0.82):
    return Match(name="close_x", confidence=score, x=x, y=y, w=60, h=60)


# --- handle_daily_streak: verify then act -------------------------------------


def test_claim_word_present_taps_it(monkeypatch):
    c = _ctrl(monkeypatch)
    _vision(monkeypatch, claim=(1320, 698), close_x=_x_at(1520, 160))
    c.handle_daily_streak(SCREEN)
    assert c.taps == [((1320, 698), "daily_streak_claim")]
    assert c.keys == []
    assert c.phases == ["returning"]
    assert ("daily_streak_claim", {}) in c.dl.events


def test_no_claim_but_close_x_in_region_taps_the_x(monkeypatch):
    c = _ctrl(monkeypatch)
    _vision(monkeypatch, claim=None, close_x=_x_at(1520, 160))
    c.handle_daily_streak(SCREEN)
    assert c.taps == [((1520, 160), "close_x")]
    assert c.keys == []
    assert ("popup_close", {"popup": "daily_streak", "action": "close_x"}) in c.dl.events


def test_close_x_outside_the_region_is_ignored(monkeypatch):
    c = _ctrl(monkeypatch)
    _vision(monkeypatch, claim=None, close_x=_x_at(800, 450))
    c.handle_daily_streak(SCREEN)
    assert c.taps == []
    assert c.keys == [4]  # fell through to BACK


def test_neither_found_presses_back_once(monkeypatch):
    c = _ctrl(monkeypatch)
    _vision(monkeypatch, claim=None, close_x=None)
    c.handle_daily_streak(SCREEN)
    assert c.taps == []
    assert c.keys == [4]
    assert ("popup_close", {"popup": "daily_streak", "action": "back"}) in c.dl.events


def test_never_taps_the_fixed_claim_point_blind(monkeypatch):
    c = _ctrl(monkeypatch)
    _vision(monkeypatch, claim=None, close_x=None)
    c.handle_daily_streak(SCREEN)
    assert config.DAILY_STREAK_CLAIM not in [p for p, _ in c.taps]


def test_close_x_region_covers_the_popup_x():
    x1, y1, x2, y2 = config.DAILY_STREAK_CLOSE_X_REGION
    assert x1 <= 1450 and x2 >= 1600 - 1 and y1 <= 120 and y2 >= 215
    assert config.DAILY_STREAK_CLOSE_X_THRESHOLD == 0.80
    assert config.MATCH_THRESHOLD == 0.85  # the global close_x threshold is untouched


def test_claim_region_surrounds_the_claim_point():
    x1, y1, x2, y2 = config.DAILY_STREAK_CLAIM_REGION
    cx, cy = config.DAILY_STREAK_CLAIM
    assert x1 < cx < x2 and y1 < cy < y2


# --- the UNKNOWN-path repeat guard ---------------------------------------------


def _modals(monkeypatch, c, frames):
    """Feed the guard a sequence of frames, each 'streak', 'invite' or 'none'.
    The handlers are stubbed to record their runs."""
    kind = {"now": None}
    monkeypatch.setattr(controller_mod.states, "is_daily_streak", lambda s: kind["now"] == "streak")
    monkeypatch.setattr(controller_mod.states, "is_team_invite", lambda s: kind["now"] == "invite")
    monkeypatch.setattr(
        Controller, "handle_daily_streak", lambda self, s: self.handled.append("streak")
    )
    monkeypatch.setattr(
        Controller, "handle_team_invite", lambda self, s: self.handled.append("invite")
    )
    results = []
    for f in frames:
        kind["now"] = f
        results.append(c._unknown_modal(SCREEN, State.UNKNOWN))
    return results


def test_repeat_max_is_four():
    assert config.HANDLER_REPEAT_MAX == 4


def test_four_consecutive_streak_iterations_recover(monkeypatch):
    c = _ctrl(monkeypatch)
    res = _modals(monkeypatch, c, ["streak"] * 4)
    assert res == [True] * 4
    assert c.handled == ["streak"] * 3  # the 4th runs recover INSTEAD of the handler
    assert c.recovered == ["daily_streak_stuck"]
    assert c._repeat_count == 0  # reset after escalating


def test_count_restarts_after_escalating(monkeypatch):
    c = _ctrl(monkeypatch)
    _modals(monkeypatch, c, ["streak"] * 7)
    assert c.recovered == ["daily_streak_stuck"]
    assert c.handled == ["streak"] * 6


def test_interleaved_other_branch_resets_the_count(monkeypatch):
    c = _ctrl(monkeypatch)
    res = _modals(monkeypatch, c, ["streak"] * 3 + ["none"] + ["streak"] * 3)
    assert res[3] is False  # no modal -> the loop falls through to the phase path
    assert c.recovered == []
    assert c.handled == ["streak"] * 6


def test_switching_handler_resets_the_count(monkeypatch):
    c = _ctrl(monkeypatch)
    _modals(monkeypatch, c, ["streak"] * 3 + ["invite"] + ["streak"] * 3)
    assert c.recovered == []


def test_non_unknown_state_resets_the_count(monkeypatch):
    c = _ctrl(monkeypatch)
    _modals(monkeypatch, c, ["streak"] * 3)
    assert c._unknown_modal(SCREEN, State.MENU) is False
    _modals(monkeypatch, c, ["streak"] * 3)
    assert c.recovered == []


def test_loop_reset_hook_clears_the_count(monkeypatch):
    # The DISCONNECT and POPUP branches run before the UNKNOWN path and call this.
    c = _ctrl(monkeypatch)
    _modals(monkeypatch, c, ["streak"] * 3)
    c._clear_handler_repeat()
    _modals(monkeypatch, c, ["streak"] * 3)
    assert c.recovered == []


def test_guard_also_covers_team_invite(monkeypatch):
    c = _ctrl(monkeypatch)
    _modals(monkeypatch, c, ["invite"] * 4)
    assert c.handled == ["invite"] * 3
    assert c.recovered == ["team_invite_stuck"]
