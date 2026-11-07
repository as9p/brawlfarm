"""Shadow rules: what three pro rules would say about one frame. Nothing acts on the answer.

``evaluate(reading, boxes, context)`` turns the HUD ``Reading`` and the detector's boxes into a
``RuleCalls``, one call per rule. A call is a suggested label and a short reason code, or None
when an input is missing, never a guess. Pure: no device, no file, no network, no OpenCV, and
nothing here can send input. The calls land in the shadow rows and change no play. Rules 2, 3 and
5 of docs/superpowers/research/comp-study-showdown.md.

What the detector can do sets the shape of this file. Enemy boxes are about 60% precise with 54%
recall, so one box is a candidate, not an enemy, and a frame with no box is not a clear frame.
The model has a bush class but no wall or gas class, and no input says where the player is on
the map. Those arrive through ``Context``, which is all None in shadow today.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brawlfarm.play.detect import Box
    from brawlfarm.play.reader import Reading

# Frame pixels at 1600 x 900. All three are unmeasured starting values.
NEAR_PX = 300.0  # a power_cube or box this close to self counts as nearby
EXIT_REACH_PX = 350.0  # a bush this close to self is somewhere to retreat to
KEEP_AMMO = 1  # stop hitting a box at this many loaded segments, of three

# An enemy box in this many of the last ENEMY_WINDOW frames, this one included, backs itself up.
# Also unmeasured. A false box that sits still passes this, a read enemy bar is the stronger sign.
ENEMY_WINDOW = 5
ENEMY_FRAMES = 3

Point = tuple[float, float]


@dataclass(frozen=True)
class RuleCall:
    label: str
    reason: str


@dataclass(frozen=True)
class RuleCalls:
    centre: RuleCall | None = None
    cubes: RuleCall | None = None
    exit: RuleCall | None = None


@dataclass(frozen=True)
class Context:
    """What the frame alone cannot say. None means not known, which is every field but one today."""

    enemy_recent: int = 0  # frames of the last ENEMY_WINDOW with an enemy box, this one included
    in_centre: bool | None = None  # needs map position; nothing reads it yet
    ahead_on_cubes: bool | None = None  # needs enemy cube counts; the reader has only our own
    gas_forced: bool | None = None  # needs map position against the gas edge
    wall_exit: bool | None = None  # needs map position against the wall grid


def _point(box: Box) -> Point:
    return box.x + box.w / 2, box.y + box.h / 2


def _self_point(boxes: Sequence[Box]) -> Point | None:
    selves = [box for box in boxes if box.cls == "self"]
    return _point(max(selves, key=lambda box: box.score)) if selves else None


def _distance(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _near(me: Point, boxes: Sequence[Box], cls: str) -> bool:
    return any(box.cls == cls and _distance(me, _point(box)) <= NEAR_PX for box in boxes)


def _enemy_confirmed(reading: Reading, enemies: Sequence[Point], context: Context) -> bool:
    """A box this frame, backed by a read enemy bar or by the frames before it."""
    if not enemies:
        return False
    return bool(reading.enemy_health) or context.enemy_recent >= ENEMY_FRAMES


def _leads_away(me: Point, cover: Point, enemies: Sequence[Point]) -> bool:
    """In reach, and the step towards it is a step away from every enemy box.

    Every box counts, confirmed or not: an extra false box can only cost a call, and a route
    towards a real enemy is the mistake this rule exists to stop.
    """
    if _distance(me, cover) > EXIT_REACH_PX:
        return False
    step = cover[0] - me[0], cover[1] - me[1]
    return all(step[0] * (me[0] - enemy[0]) + step[1] * (me[1] - enemy[1]) > 0 for enemy in enemies)


def _centre_rule(context: Context) -> RuleCall | None:
    """Rule 2: stay out of the centre unless ahead on cubes or forced by gas."""
    if context.in_centre is None:
        return None
    if context.gas_forced:
        return RuleCall("centre_ok", "gas_forced")
    if context.ahead_on_cubes:
        return RuleCall("centre_ok", "ahead_on_cubes")
    if context.gas_forced is None or context.ahead_on_cubes is None:
        return None  # one True above settles it, a False needs both known
    return RuleCall("leave_centre" if context.in_centre else "stay_out", "not_ahead")


def _cubes_rule(reading: Reading, boxes: Sequence[Box]) -> RuleCall | None:
    """Rule 3: a nearby power_cube first, and never the last ammo on a box."""
    me = _self_point(boxes)
    if me is None:
        return None
    if _near(me, boxes, "power_cube"):
        return RuleCall("take_cube", "cube_near")  # walking onto a cube costs no ammo
    if not _near(me, boxes, "box") or reading.self_ammo is None:
        return None
    if reading.self_ammo <= KEEP_AMMO:
        return RuleCall("leave_box", "keep_ammo")
    return RuleCall("hit_box", "ammo_spare")


def _exit_rule(reading: Reading, boxes: Sequence[Box], context: Context) -> RuleCall | None:
    """Rule 5: engage only with a route away from the enemy, by a bush or a wall.

    No bush on the far side says nothing while walls go unseen, so that case needs the wall
    input before it can say no. An unconfirmed enemy is no call at all, either way.
    """
    me = _self_point(boxes)
    enemies = [_point(box) for box in boxes if box.cls == "enemy"]
    if me is None or not _enemy_confirmed(reading, enemies, context):
        return None
    bushes = [_point(box) for box in boxes if box.cls == "bush"]
    if any(_leads_away(me, bush, enemies) for bush in bushes):
        return RuleCall("engage_ok", "bush_exit")
    if context.wall_exit is None:
        return None
    if context.wall_exit:
        return RuleCall("engage_ok", "wall_exit")
    return RuleCall("no_engage", "no_exit")


def evaluate(reading: Reading, boxes: Sequence[Box], context: Context) -> RuleCalls:
    """One frame's calls. A rule whose inputs are missing is None."""
    return RuleCalls(
        centre=_centre_rule(context),
        cubes=_cubes_rule(reading, boxes),
        exit=_exit_rule(reading, boxes, context),
    )
