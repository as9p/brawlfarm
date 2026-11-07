"""Regression tests for the mid-session brawler reselect (the "CROW 336/100" bug).

Owner's law (discord-v3-refinements.md A1): in ladder mode, if the farmed brawler
is above — or climbs above — the CURRENT ladder step goal, the worker must switch.
The bug: the controller compared against a goal FROZEN at selection time (or the
1000 default after a farmplan hiccup), so when the roster minimum moved (e.g. a
drop unlocked a new brawler at 0, pulling the step down to 0-100) the worker kept
grinding a 336-trophy brawler against a stale goal. The fix re-evaluates the plan
goal against the live roster on every trophy snapshot.

These tests drive Controller._check_farm_brawler_trophies directly (no adb, no
API): the controller object is built without __init__ and farmplan.load_plan is
patched, so the check's inputs are exactly the live player payload + the plan.
"""

from brawlfarm.core import farmplan
from brawlfarm.core.controller import Controller


class _DL:
    """Event-recording stand-in for DataLog (no disk writes)."""

    def __init__(self):
        self.events = []

    def event(self, etype, **fields):
        self.events.append((etype, fields))


def _ctrl(farm_brawler="CROW", farm_goal=1000):
    c = Controller.__new__(Controller)  # skip __init__ (ApiClient/DataLog/adb)
    c._farm_brawler = farm_brawler
    c._farm_goal = farm_goal
    c._reselect_pending = False
    c._rotated = set()  # ladder session rotation state (r6: folded in from optimal)
    c._account_maxed = False  # prestige "nothing under the goal" latch (2026-06-14)
    c.dl = _DL()
    return c


def _player(*pairs):
    return {"brawlers": [{"name": n, "trophies": t} for n, t in pairs]}


def _ladder_plan():
    return {"mode": "ladder", "goal_trophies": 1000}


def test_owner_case_crow_336_over_step_100_switches(monkeypatch):
    """THE bug report: /status showed CROW 336/100 (a new 0-trophy brawler pulled
    the ladder step down to 0-100) and the worker kept farming CROW. The check
    must now use the LIVE step goal (100) and flag the reselect."""
    monkeypatch.setattr(farmplan, "load_plan", lambda: _ladder_plan())
    c = _ctrl(farm_brawler="CROW", farm_goal=400)  # goal frozen at selection time
    c._check_farm_brawler_trophies(_player(("CROW", 336), ("NEW", 0), ("NITA", 500)))
    assert c._reselect_pending is True
    assert c._farm_goal == 100  # the live step goal, not the stale 400


def test_owner_case_with_default_1000_goal_switches(monkeypatch):
    """Same roster, but the selection-time resolve hiccuped so the controller still
    holds the 1000 default — the live re-evaluation must still catch it."""
    monkeypatch.setattr(farmplan, "load_plan", lambda: _ladder_plan())
    c = _ctrl(farm_brawler="CROW", farm_goal=1000)
    c._check_farm_brawler_trophies(_player(("CROW", 336), ("NEW", 0)))
    assert c._reselect_pending is True


def test_ladder_below_live_step_keeps_farming(monkeypatch):
    # farmed brawler IS the minimum and under its tier line: no switch
    monkeypatch.setattr(farmplan, "load_plan", lambda: _ladder_plan())
    c = _ctrl(farm_brawler="CROW", farm_goal=100)
    c._check_farm_brawler_trophies(_player(("CROW", 36), ("NITA", 500)))
    assert c._reselect_pending is False
    assert c._farm_goal == 100


def test_ladder_climbs_above_step_switches(monkeypatch):
    # "or CLIMBS above": CROW passes a lower teammate's tier line mid-session
    monkeypatch.setattr(farmplan, "load_plan", lambda: _ladder_plan())
    c = _ctrl(farm_brawler="CROW", farm_goal=100)
    c._check_farm_brawler_trophies(_player(("CROW", 105), ("NITA", 96)))
    assert c._reselect_pending is True  # NITA (96) is now the step's lowest


def test_goal_refresh_failure_keeps_last_goal(monkeypatch):
    # a plan-read hiccup must not crash the snapshot or zero the goal
    def boom():
        raise OSError("farmplan.json unreadable")

    monkeypatch.setattr(farmplan, "load_plan", boom)
    c = _ctrl(farm_brawler="CROW", farm_goal=400)
    c._check_farm_brawler_trophies(_player(("CROW", 336), ("NEW", 0)))
    assert c._reselect_pending is False  # 336 < the kept 400
    assert c._farm_goal == 400
    assert any(e == "farmplan_error" for e, _ in c.dl.events)


def test_prestige_mode_goal_cross_reselects(monkeypatch):
    # v5: manual mode (and its advance_after_goal queue popper) is gone — the
    # goal-cross check still reselects on the remaining plan-goal mode, prestige.
    plan = {"mode": "prestige", "prestige_start": "highest", "goal_trophies": 1000}
    monkeypatch.setattr(farmplan, "load_plan", lambda: dict(plan))
    c = _ctrl(farm_brawler="CROW", farm_goal=1000)
    c._check_farm_brawler_trophies(_player(("CROW", 1001), ("NITA", 500)))
    assert c._reselect_pending is True


def test_prestige_lowest_goal_cross_reselects_while_others_under_goal(monkeypatch):
    # prestige/lowest (the owner's 2026-06-14 plan): the farmed brawler crosses
    # 1000 but others are still under it -> switch to the next lowest, not maxed.
    plan = {"mode": "prestige", "prestige_start": "lowest", "goal_trophies": 1000}
    monkeypatch.setattr(farmplan, "load_plan", lambda: dict(plan))
    c = _ctrl(farm_brawler="TARA", farm_goal=1000)
    c._check_farm_brawler_trophies(_player(("TARA", 1000), ("SPIKE", 807)))
    assert c._reselect_pending is True
    assert c._account_maxed is False


def test_prestige_all_maxed_holds_without_reselecting(monkeypatch):
    # every brawler >= the goal -> account maxed: hold the current brawler, do NOT
    # reselect (which would fall back to the fooled in-game lowest sort and grind a
    # >1000 brawler). The latch + one event fire; trophies keep being checked.
    plan = {"mode": "prestige", "prestige_start": "lowest", "goal_trophies": 1000}
    monkeypatch.setattr(farmplan, "load_plan", lambda: dict(plan))
    c = _ctrl(farm_brawler="TARA", farm_goal=1000)
    c._check_farm_brawler_trophies(_player(("TARA", 1011), ("SPIKE", 1000)))
    assert c._reselect_pending is False
    assert c._account_maxed is True
    assert any(e == "account_maxed" for e, _ in c.dl.events)


def test_prestige_maxed_fallback_switches_when_not_yet_on_it(monkeypatch):
    # maxed_fallback set + current brawler != FRANK -> switch ONCE: flag the
    # reselect, do NOT latch maxed, fire maxed_fallback_switch (no account_maxed).
    plan = {
        "mode": "prestige",
        "prestige_start": "lowest",
        "goal_trophies": 1000,
        "maxed_fallback": "FRANK",
    }
    monkeypatch.setattr(farmplan, "load_plan", lambda: dict(plan))
    c = _ctrl(farm_brawler="TARA", farm_goal=1000)
    c._check_farm_brawler_trophies(_player(("TARA", 1011), ("FRANK", 1200)))
    assert c._reselect_pending is True
    assert c._account_maxed is False
    assert any(e == "maxed_fallback_switch" for e, _ in c.dl.events)
    assert not any(e == "account_maxed" for e, _ in c.dl.events)


def test_prestige_maxed_fallback_holds_once_already_on_fallback(monkeypatch):
    # already farming FRANK -> hold it (latch + account_maxed), NO switch event, no
    # thrash on subsequent snapshots. (Case-insensitive: stored "frank" vs API "FRANK".)
    plan = {
        "mode": "prestige",
        "prestige_start": "lowest",
        "goal_trophies": 1000,
        "maxed_fallback": "frank",
    }
    monkeypatch.setattr(farmplan, "load_plan", lambda: dict(plan))
    c = _ctrl(farm_brawler="FRANK", farm_goal=1000)
    c._check_farm_brawler_trophies(_player(("FRANK", 1200), ("TARA", 1011)))
    assert c._reselect_pending is False
    assert c._account_maxed is True
    assert not any(e == "maxed_fallback_switch" for e, _ in c.dl.events)
    assert any(e == "account_maxed" for e, _ in c.dl.events)


def test_prestige_maxed_no_fallback_holds_as_today(monkeypatch):
    # no maxed_fallback -> the existing hold behavior, unchanged (asserting no
    # switch event leaks in alongside the latch).
    plan = {"mode": "prestige", "prestige_start": "lowest", "goal_trophies": 1000}
    monkeypatch.setattr(farmplan, "load_plan", lambda: dict(plan))
    c = _ctrl(farm_brawler="TARA", farm_goal=1000)
    c._check_farm_brawler_trophies(_player(("TARA", 1011), ("SPIKE", 1000)))
    assert c._reselect_pending is False
    assert c._account_maxed is True
    assert any(e == "account_maxed" for e, _ in c.dl.events)
    assert not any(e == "maxed_fallback_switch" for e, _ in c.dl.events)


def test_prestige_maxed_fallback_unowned_does_not_thrash(monkeypatch):
    # maxed_fallback names a brawler the account does NOT own (typo / not-yet-owned).
    # The controller must NOT fire the switch — choose_target would return None, the
    # account would hold unchanged, and a switch fired here would re-fire every
    # snapshot forever (~60s reselect-thrash). Gate on ownership: hold, no switch
    # event, no reselect — and stable when repeated.
    plan = {
        "mode": "prestige",
        "prestige_start": "lowest",
        "goal_trophies": 1000,
        "maxed_fallback": "FRNAK",  # typo: not in the roster
    }
    monkeypatch.setattr(farmplan, "load_plan", lambda: dict(plan))
    c = _ctrl(farm_brawler="TARA", farm_goal=1000)
    for _ in range(2):  # must be stable across snapshots, never thrash
        c._check_farm_brawler_trophies(_player(("TARA", 1011), ("SPIKE", 1000)))
        assert c._reselect_pending is False
        assert not any(e == "maxed_fallback_switch" for e, _ in c.dl.events)
    assert c._account_maxed is True
    assert any(e == "account_maxed" for e, _ in c.dl.events)


def test_prestige_maxed_latch_does_not_respam(monkeypatch):
    # once maxed, a second over-goal snapshot must not re-log or set reselect
    plan = {"mode": "prestige", "prestige_start": "lowest", "goal_trophies": 1000}
    monkeypatch.setattr(farmplan, "load_plan", lambda: dict(plan))
    c = _ctrl(farm_brawler="TARA", farm_goal=1000)
    c._account_maxed = True  # already latched
    c._check_farm_brawler_trophies(_player(("TARA", 1011), ("SPIKE", 1000)))
    assert c._reselect_pending is False
    assert not any(e == "account_maxed" for e, _ in c.dl.events)  # no re-log


def test_ladder_all_over_goal_still_reselects(monkeypatch):
    # the maxed-hold is prestige-ONLY: ladder keeps reselecting (it raises the whole
    # roster a tier at a time; the step goal moves with the minimum).
    monkeypatch.setattr(farmplan, "load_plan", lambda: _ladder_plan())
    c = _ctrl(farm_brawler="TARA", farm_goal=1000)
    c._check_farm_brawler_trophies(_player(("TARA", 1011), ("SPIKE", 1000)))
    assert c._reselect_pending is True
    assert c._account_maxed is False


def test_select_api_error_falls_back_to_lowest_not_maxed(monkeypatch):
    # MAJOR (review 2026-06-14): resolve_target raising (transient API error) also
    # yields target=None — it must NOT be misread as "account maxed". The worker
    # falls back to the legacy in-game lowest-trophy selection, as before this change.
    from brawlfarm.core import brawlers

    def boom(api, exclude=()):
        raise RuntimeError("api down")

    monkeypatch.setattr(farmplan, "resolve_target", boom)
    monkeypatch.setattr(
        farmplan,
        "load_plan",
        lambda: {"mode": "prestige", "prestige_start": "lowest", "goal_trophies": 1000},
    )
    called = []
    monkeypatch.setattr(
        brawlers,
        "select_lowest_trophy_brawler",
        lambda log: (called.append(True), "SPIKE")[1],
    )
    c = _ctrl(farm_brawler="TARA", farm_goal=1000)
    c.api = object()
    c.log = lambda *a, **k: None
    c._capture_farm_brawler = lambda: None
    c._do_select_brawler()
    assert called == [True]  # legacy lowest flow ran
    assert c._account_maxed is False  # NOT falsely marked maxed
    assert not any(e == "account_maxed" for e, _ in c.dl.events)


def test_select_prestige_maxed_holds_not_lowest(monkeypatch):
    # The complement: a SUCCESSFUL resolve with nothing under the goal holds — it
    # must NOT run the in-game lowest sort (which would grind a >goal brawler).
    from brawlfarm.core import brawlers

    monkeypatch.setattr(farmplan, "resolve_target", lambda api, exclude=(): (None, 1000, []))
    monkeypatch.setattr(
        farmplan,
        "load_plan",
        lambda: {"mode": "prestige", "prestige_start": "lowest", "goal_trophies": 1000},
    )
    called = []
    monkeypatch.setattr(
        brawlers,
        "select_lowest_trophy_brawler",
        lambda log: (called.append(True), "SPIKE")[1],
    )
    c = _ctrl(farm_brawler="TARA", farm_goal=1000)
    c.api = object()
    c.log = lambda *a, **k: None
    c._capture_farm_brawler = lambda: None
    c._do_select_brawler()
    assert called == []  # held — legacy lowest flow NOT run
    assert c._account_maxed is True
    assert any(e == "account_maxed" for e, _ in c.dl.events)


def test_no_double_fire_once_pending(monkeypatch):
    monkeypatch.setattr(farmplan, "load_plan", lambda: _ladder_plan())
    c = _ctrl(farm_brawler="CROW", farm_goal=100)
    c._reselect_pending = True
    c._check_farm_brawler_trophies(_player(("CROW", 336), ("NEW", 0)))
    assert c._reselect_pending is True  # unchanged, no re-log
    assert c.dl.events == []


class _Api:
    """The roster the API reports: WENDY is the lowest, whatever the game has selected."""

    def get_player(self):
        return {"brawlers": [{"name": "VINCE", "trophies": 300}, {"name": "WENDY", "trophies": 0}]}


def _lowest_select(monkeypatch, select):
    """A controller on VINCE that takes the lowest-trophy path, with the real capture.
    Returns (controller, log lines)."""
    from brawlfarm.core import brawlers

    monkeypatch.setattr(farmplan, "resolve_target", lambda api, exclude=(): (None, 1000, []))
    monkeypatch.setattr(farmplan, "load_plan", lambda: _ladder_plan())
    monkeypatch.setattr(brawlers, "select_lowest_trophy_brawler", select)
    logged = []
    c = _ctrl(farm_brawler="VINCE")
    c._quest_aware = False
    c.api = _Api()
    c.log = logged.append
    return c, logged


def test_failed_select_keeps_the_farm_brawler(monkeypatch):
    # The select never reached the card, so the game is still on VINCE: taking the API's
    # lowest here would watch a brawler nobody is playing.
    c, logged = _lowest_select(monkeypatch, lambda log: None)
    c._do_select_brawler()
    assert c._farm_brawler == "VINCE"
    assert not any("farm brawler =" in line for line in logged)
    assert [e for e, _ in c.dl.events] == ["select_brawler_error"]


def test_select_that_raises_keeps_the_farm_brawler(monkeypatch):
    def boom(log):
        raise RuntimeError("adb down")

    c, logged = _lowest_select(monkeypatch, boom)
    c._do_select_brawler()
    assert c._farm_brawler == "VINCE"
    assert not any("farm brawler =" in line for line in logged)
    assert [e for e, _ in c.dl.events] == ["select_brawler_error"]


def test_successful_select_captures_the_brawler(monkeypatch):
    c, logged = _lowest_select(monkeypatch, lambda log: "WENDY")
    c._do_select_brawler()
    assert c._farm_brawler == "WENDY"
    assert any("farm brawler = WENDY" in line for line in logged)
    assert c.dl.events == [("select_brawler", {"brawler": "WENDY"})]
