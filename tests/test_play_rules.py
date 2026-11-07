"""The shadow rules, on made-up boxes and readings. Every None path is a case here."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from brawlfarm.play import rules
from brawlfarm.play.detect import Box
from brawlfarm.play.reader import Reading
from brawlfarm.play.rules import Context, RuleCall, RuleCalls


def box(cls: str, cx: float, cy: float, score: float = 0.8) -> Box:
    """A 40 x 40 box centred on (cx, cy)."""
    return Box(cls=cls, score=score, x=cx - 20, y=cy - 20, w=40, h=40)


ME = box("self", 800, 450)
ENEMY_RIGHT = box("enemy", 1100, 450)
BUSH_LEFT = box("bush", 600, 450)
BUSH_RIGHT = box("bush", 950, 450)
FIRM = Context(enemy_recent=rules.ENEMY_FRAMES)
BAR = Reading(enemy_health=(1200,))


def evaluate(boxes=(), reading=Reading(), context=Context()) -> RuleCalls:
    return rules.evaluate(reading, boxes, context)


def test_nothing_known_is_three_nones() -> None:
    assert evaluate() == RuleCalls(centre=None, cubes=None, exit=None)


@pytest.mark.parametrize("self_dead", [False, None])
def test_a_live_or_unread_self_keeps_every_call(self_dead) -> None:
    boxes = (ME, box("power_cube", 850, 450), ENEMY_RIGHT, BUSH_LEFT)
    reading = Reading(enemy_health=(1200,), self_dead=self_dead)
    calls = evaluate(boxes, reading, Context(in_centre=True, gas_forced=True, enemy_recent=5))
    assert calls == RuleCalls(
        centre=RuleCall("centre_ok", "gas_forced"),
        cubes=RuleCall("take_cube", "cube_near"),
        exit=RuleCall("engage_ok", "bush_exit"),
    )


def test_a_dead_self_gets_no_call_from_any_rule() -> None:
    boxes = (ME, box("power_cube", 850, 450), ENEMY_RIGHT, BUSH_LEFT)
    reading = Reading(enemy_health=(1200,), self_dead=True)
    calls = evaluate(boxes, reading, Context(in_centre=True, gas_forced=True, enemy_recent=5))
    assert calls == RuleCalls()


# -- rule 2, the centre ------------------------------------------------------------


@pytest.mark.parametrize(
    "context",
    [
        Context(),  # what shadow passes today
        Context(gas_forced=True, ahead_on_cubes=True),  # no map position
        Context(in_centre=True),
        Context(in_centre=True, gas_forced=False),  # ahead on cubes not known
        Context(in_centre=False, ahead_on_cubes=False),  # gas not known
    ],
)
def test_the_centre_rule_needs_its_inputs(context: Context) -> None:
    assert evaluate(context=context).centre is None


@pytest.mark.parametrize(
    ("context", "call"),
    [
        (
            Context(in_centre=True, gas_forced=True),
            RuleCall("centre_ok", "gas_forced"),
        ),
        (
            Context(in_centre=False, gas_forced=False, ahead_on_cubes=True),
            RuleCall("centre_ok", "ahead_on_cubes"),
        ),
        (
            Context(in_centre=True, gas_forced=False, ahead_on_cubes=False),
            RuleCall("leave_centre", "not_ahead"),
        ),
        (
            Context(in_centre=False, gas_forced=False, ahead_on_cubes=False),
            RuleCall("stay_out", "not_ahead"),
        ),
    ],
)
def test_the_centre_rule_with_its_inputs(context: Context, call: RuleCall) -> None:
    assert evaluate(context=context).centre == call


# -- rule 3, cubes and boxes -------------------------------------------------------


def test_the_cubes_rule_needs_a_self_box() -> None:
    assert evaluate([box("power_cube", 800, 450)], Reading(self_ammo=3)).cubes is None


def test_a_nearby_cube_comes_first_and_needs_no_ammo_read() -> None:
    near = box("power_cube", 800 + rules.NEAR_PX, 450)
    crate = box("box", 850, 450)
    assert evaluate([ME, crate, near]).cubes == RuleCall("take_cube", "cube_near")


@pytest.mark.parametrize(
    "boxes",
    [
        [ME],  # nothing to take
        [ME, box("power_cube", 800 + rules.NEAR_PX + 1, 450)],  # a cube, too far off
        [ME, box("box", 800, 450 + rules.NEAR_PX + 1)],  # a box, too far off
    ],
)
def test_the_cubes_rule_is_quiet_with_nothing_nearby(boxes: list[Box]) -> None:
    assert evaluate(boxes, Reading(self_ammo=3)).cubes is None


def test_a_nearby_box_with_unread_ammo_is_no_call() -> None:
    assert evaluate([ME, box("box", 850, 450)], Reading(self_ammo=None)).cubes is None


@pytest.mark.parametrize(
    ("ammo", "call"),
    [
        (0, RuleCall("leave_box", "keep_ammo")),
        (rules.KEEP_AMMO, RuleCall("leave_box", "keep_ammo")),
        (2, RuleCall("hit_box", "ammo_spare")),
        (3, RuleCall("hit_box", "ammo_spare")),
    ],
)
def test_a_nearby_box_never_gets_the_last_ammo(ammo: int, call: RuleCall) -> None:
    assert evaluate([ME, box("box", 850, 450)], Reading(self_ammo=ammo)).cubes == call


def test_the_best_self_box_is_the_one_that_counts() -> None:
    false_self = box("self", 100, 100, score=0.4)
    cube = box("power_cube", 850, 450)
    assert evaluate([false_self, ME, cube]).cubes == RuleCall("take_cube", "cube_near")


# -- rule 5, the exit --------------------------------------------------------------


def test_the_exit_rule_needs_a_self_box() -> None:
    assert evaluate([ENEMY_RIGHT, BUSH_LEFT], BAR, FIRM).exit is None


def test_no_enemy_box_is_no_exit_call_even_with_a_read_bar() -> None:
    assert evaluate([ME, BUSH_LEFT], BAR, FIRM).exit is None


@pytest.mark.parametrize("recent", [0, 1, rules.ENEMY_FRAMES - 1])
def test_one_lone_enemy_box_is_never_treated_as_certain(recent: int) -> None:
    context = Context(enemy_recent=recent, wall_exit=False)
    assert evaluate([ME, ENEMY_RIGHT, BUSH_LEFT], Reading(), context).exit is None


@pytest.mark.parametrize(
    ("reading", "context"),
    [(BAR, Context()), (Reading(), FIRM)],
    ids=["a read enemy bar", "the frames before it"],
)
def test_a_box_is_confirmed_by_a_bar_or_by_recent_frames(reading, context) -> None:
    call = evaluate([ME, ENEMY_RIGHT, BUSH_LEFT], reading, context).exit
    assert call == RuleCall("engage_ok", "bush_exit")


@pytest.mark.parametrize(
    "bush",
    [
        BUSH_RIGHT,  # towards the enemy
        box("bush", 800, 200),  # square to the enemy, not away from it
        box("bush", 800 - rules.EXIT_REACH_PX - 1, 450),  # away, out of reach
    ],
)
def test_a_bush_that_is_no_route_away_says_nothing_without_the_walls(bush: Box) -> None:
    assert evaluate([ME, ENEMY_RIGHT, bush], BAR, FIRM).exit is None


def test_no_bush_at_all_says_nothing_without_the_walls() -> None:
    assert evaluate([ME, ENEMY_RIGHT], BAR, FIRM).exit is None


def test_a_route_must_lead_away_from_every_enemy_box() -> None:
    second = box("enemy", 500, 450)
    assert evaluate([ME, ENEMY_RIGHT, second, BUSH_LEFT], BAR, FIRM).exit is None


@pytest.mark.parametrize(
    ("wall_exit", "call"),
    [(True, RuleCall("engage_ok", "wall_exit")), (False, RuleCall("no_engage", "no_exit"))],
)
def test_the_wall_input_settles_what_the_bushes_cannot(wall_exit: bool, call: RuleCall) -> None:
    context = Context(enemy_recent=rules.ENEMY_FRAMES, wall_exit=wall_exit)
    assert evaluate([ME, ENEMY_RIGHT], BAR, context).exit == call


def test_a_bush_route_does_not_wait_for_the_walls() -> None:
    context = Context(enemy_recent=rules.ENEMY_FRAMES, wall_exit=False)
    assert evaluate([ME, ENEMY_RIGHT, BUSH_LEFT], BAR, context).exit.reason == "bush_exit"


# -- rails -------------------------------------------------------------------------


def test_rules_imports_nothing_at_run_time_that_could_reach_a_device() -> None:
    allowed = {"__future__", "collections.abc", "dataclasses", "typing"}
    typing_only = {"brawlfarm.play.detect", "brawlfarm.play.reader"}
    tree = ast.parse(Path(rules.__file__).read_text(encoding="utf-8"))
    guarded = [node for node in tree.body if isinstance(node, ast.If)]
    assert all(node.test.id == "TYPE_CHECKING" for node in guarded)
    for node in tree.body:
        assert not isinstance(node, ast.Import)
        if isinstance(node, ast.ImportFrom):
            assert node.module in allowed
    for node in (inner for block in guarded for inner in block.body):
        assert node.module in typing_only


def test_importing_rules_loads_no_input_path() -> None:
    watched = ("brawlfarm.core.adb", "brawlfarm.core.controller", "cv2", "numpy")
    code = f"import sys, brawlfarm.play.rules; print([m for m in {watched!r} if m in sys.modules])"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"
