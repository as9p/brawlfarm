"""A pure reader for the in-match HUD: teams left, self health, ammo, cubes, enemy health, dead.

``read(frame, boxes)`` looks at one 1600 x 900 BGR frame and returns a ``Reading``. A field it
cannot read is None, never a guess. CPU only, numpy and OpenCV only, no device, no network, no
file access. It sends no input and changes no play. docs/superpowers plan 2026-11-05, "Reader
design".

Every pixel constant here was measured on capture-20261104c (scratch work, no frames in the
repository) and sits in this file on purpose: it is a reader of the HUD, not a tap calibration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from brawlfarm.play.detect import Box

# The only frame size the constants below mean anything for.
FRAME_W, FRAME_H = 1600, 900

# Teams left: fixed text top left, white with a black outline, gas does not reach it. The boxes
# are the extents of the white text over 300 in-match frames (x0, y0, x1, y1).
TEAMS_LABEL_BOX = (30, 28, 254, 68)
TEAMS_DIGIT_BOX = (258, 29, 286, 66)
# The text is near pure white; 200 kept the outline out over the same 300 frames.
TEAMS_WHITE = 200
TEAMS_LABEL_SIZE = (56, 10)
TEAMS_DIGIT_SIZE = (10, 13)
TEAMS_LABEL_MIN = 0.92  # in-match 0.956 at the 1st percentile, off-match 0.641 at the 99th
TEAMS_DIGIT_MIN = 0.9  # in-match 0.952 at the 1st percentile, off-match 0.70 at the 99th
TEAMS_DIGIT_MARGIN = 0.05  # same run: best minus second best digit

# Dead: two white labels, fixed on screen wherever the camera is. "Back in:" tops the respawn
# countdown, and "Exit" sits in a button that replaces the controls while the camera watches the
# rest of the match. The boxes are the extents of the text over 254 and 204 frames, plus 2.
BACK_LABEL_BOX = (729, 316, 876, 351)
EXIT_LABEL_BOX = (777, 822, 822, 843)
BACK_LABEL_SIZE = (48, 11)
EXIT_LABEL_SIZE = (22, 11)
# Label on screen 0.947 at the 1st percentile, alive 0.640 at the most. The countdown pops in
# oversized at its first number and the steady text does not start until it settles; that reads
# 0.76 to 0.86 here, so the bar sits below it.
BACK_LABEL_MIN = 0.72
EXIT_LABEL_MIN = 0.85  # button on screen 0.988 at the 1st percentile, alive 0.645 at the most

# Self unit: a bar 90 wide with a row of three ammo segments under it, all offsets from the bar's
# top left. Measured on the 129 frames with a full green bar: the bar component is 85 to 92 wide
# and 16 or 17 tall. Digits overlap its top rows, so the fill is judged on rows 9 to 13.
BAR_W = 90
BODY_DY = 9
BODY_H = 5
# The ammo strip spans rows 22 to 30; 25 to 29 are the rows where an empty segment is dark.
STRIP_DY = 25
STRIP_H = 5
# Three segments, about 28 wide on a 30.5 pitch, from the same 129 frames.
SEGMENT_DX = 2
SEGMENT_PITCH = 30.5
SEGMENT_W = 24
SEGMENTS = 3
# Share of a segment that must be orange to count as loaded, and of the rest that must be the
# dark empty or blue reloading track for the strip to be believed.
SEGMENT_FULL = 0.8
SEGMENT_EMPTY = 0.6
# Where the bar's left edge sits. With no self box: bars over the 300 frames sit in x 709 to 783
# and y 320 to 376 for the 10th to 90th percentile, widened to take the frames either side of
# that. With a box: the bar's centre is level with the box and 83 above it (median).
CENTRE_WINDOW = (705, 800, 290, 420)
BOX_BAR_DY = -83
BOX_SLACK = 35
# The unit must fill this much of both rectangles. 0.8 keeps the frames with a partly hidden
# strip; below it the score was map texture.
UNIT_MIN = 0.8
# A green bar this solid can be the self bar when something hides the strip, if its digits read.
GREEN_SOLID = 0.9
# The strip alone is believed only over a bar this full; any other bar needs its digits to read.
# A solid green block with neither read as a full bar on 48 frames of the capture.
TRUST_FILL = 0.85
# The run of filled columns reads short by about this many: the rounded left end and a pixel or
# two of anchor drift. Fill minus health over maximum health had a median of -0.039 on 92 part
# bars, and the width of a full bar measures 84 to 90 columns.
FILL_SHORT = 3
# Bars of 84 columns carried the same round health as the full ones either side of them.
FULL_FILL = 0.96
# Columns past the fill must be mostly the dark track or the bar is covered, not drained.
TRACK_MIN = 0.6

# Bar colours in OpenCV HSV. Green and yellow and red from the earlier sweep of bar candidates;
# the self bar goes green to yellow to red as it drains.
GREEN_H, GREEN_S, GREEN_V = (35, 85), 100, 150
YELLOW_H, YELLOW_S, YELLOW_V = (20, 34), 100, 170
RED_LOW_H, RED_HIGH_H, RED_S, RED_V = 8, 165, 150, 150
# Loaded ammo. Over the same frames: hue 8 to 22, saturation and value high.
ORANGE_H, ORANGE_S, ORANGE_V = (8, 22), 130, 160
# Empty track: hue 106 to 124 at value 56 to 65. Map floors share the hue at value about 125, so
# the upper bound is what tells them apart.
NAVY_H, NAVY_S, NAVY_V = (95, 135), 45, (40, 100)
# The reload fill creeping along a segment is the same hue, lighter.
RELOAD_V = 175

# Health digits: white on a black outline, 8 to 10 wide, 12 or 13 tall.
DIGIT_WHITE = 170
DIGIT_H = (10, 15)
DIGIT_W = (3, 14)
DIGIT_BAND = (-11, 10)  # rows around the bar's top; the digits start 5 to 10 above it
DIGIT_GAP = 5  # largest gap inside a number
DIGIT_ROW_SLACK = 2  # tops of a number's digits line up within this
DIGIT_COUNT = (3, 5)  # 128 to 11000 on the self bar, 104 to 19360 on others, in the capture
DIGIT_CENTRE_SLACK = 10  # a self number is centred on the bar, full or not
# Canvas every glyph is compared on, in the same proportions as the digit.
GLYPH_H, GLYPH_W = 14, 12
# Over 1791 components: agreement 0.80 at the 1st percentile, margin 0.036 at the 1st.
GLYPH_MIN = 0.8
GLYPH_MARGIN = 0.03

# Cube count: green digits right of the lightning badge, offsets from the bar's top left. Over
# the frames with the badge: digit x 42 to 78, y -64 median, about 19 tall. Full health steps
# 400 with this number: 522 of 526 full bars fit base plus 400 per cube, for the two bases seen.
CUBE_BOX = (35, 90, -78, -38)
CUBE_H = (15, 22)
CUBE_W = (4, 17)
CUBE_MIN_FILL = 0.4
CUBE_GREEN_H, CUBE_GREEN_S, CUBE_GREEN_V = (40, 75), 150, 170
BADGE_BOX = (5, 40, -80, -40)
BADGE_H, BADGE_S, BADGE_V = (18, 40), 100, 150
BADGE_MIN_PIXELS = 40

# Other players: a bar 12 tall and up to about 80 wide, number on top. Teammates are cyan and are
# skipped. Enemies are red, orange or yellow, by team.
OTHER_H = (9, 14)
OTHER_W = (12, 86)
OTHER_BAND = (-13, 7)
OTHER_AREA = (0, FRAME_W, 90, 760)  # below the top row of the HUD, above the bottom row
OTHER_SOLID = 0.6


# Digit glyphs, cluster means of the health numbers on capture-20261104c. The same font is used
# for the cube digits.
_DIGITS = """
...#####.... ...######... ..######.... ..#######... .....#####..
..########.. ..#######... .########... .########... ....######..
.#########.. ..#######... .#########.. .#########.. ....######..
###########. ...######... ..########.. ..########.. ...#######..
####...####. .....####... .....#####.. .....####... ..########..
####...####. .....####... .....####... .....####... ..###.####..
####....###. .....####... .....####... ...#####.... .####.####..
####....###. .....####... ....####.... ...######... .###..####..
####...####. .....####... ...#####.... .....#####.. ###########.
#####..####. .....####... ...####..... ......####.. .##########.
###########. .....####... ..#######... ..########.. .##########.
.#########.. .....####... .#########.. .#########.. .....#####..
..########.. .....####... .#########.. .########... ......####..
...######... .....####... .#########.. ..######.... ......####..

..########.. ....#####... ..########.. ...######... ...######...
..########.. ...######... .#########.. ..########.. ..#######...
..#######... ..#######... .#########.. .#########.. .#########..
..####...... .#####...... ..########.. .####..###.. .#########..
..####...... .####....... .....#####.. .####..###.. .###...####.
..######.... .####....... .....####... ..########.. .####..####.
..#######... .#########.. .....####... ..#######... .##########.
...#######.. .#########.. .....####... ..########.. ..#########.
......####.. .####..###.. ....#####... .####..###.. ...#######..
......####.. .####..####. ....####.... .####..####. ......####..
.....#####.. .####..####. ....####.... .####..####. ......####..
..########.. .#########.. ...#####.... .##########. ..########..
..#######... ..########.. ...####..... .#########.. ..#######...
..######.... ...######... ...####..... ..#######... ..######....
"""

# Teams left digits and the "Teams left:" label, means over 300 frames.
_TEAMS_DIGITS = """
.......... .......... .......... ..........
.#####.... .######... .######... ....####..
.#####.... .#######.. .#######.. ...#####..
.#####.... ..######.. ..######.. ...#####..
...###.... .....###.. .....###.. ..###.##..
...###.... .....###.. ....###... ..###.##..
...###.... ....####.. ..#####... .###..##..
...###.... ...####... ...#####.. .########.
...###.... ...###.... .....###.. .#########
...###.... ..######.. .#######.. .#########
...###.... .########. .#######.. .....###..
...###.... .#######.. .######... .....###..
.......... .......... .......... ..........
"""

_TEAMS_LABEL = """
........................................................
.#####.............................##........###........
.#####.............................##.......##..##......
..###..###...###..########..####...##..###..#######..#..
..###.#####..###..########..###....##.#####.###.###.###.
..###.##.##.#####..##.#####.###....##.##.##.##..###.....
..###.#####.#####..##.#####..###...##.####..##..###.....
..###.###...#####..##.#####.####...##.##....##..###.###.
..###..####.#####..##.#####.###....##..###..##..###.###.
........................................................
"""

# The two dead labels, means over the frames in the boxes' comment.
_BACK_LABEL = """
................................................
.######...............##.........##.............
.######...............###........##.............
.##..##...###.....##..##...#.....##..#..##...##.
.##..##..#####...####.###.##.....##.#######..##.
.#####......##..#####.#####......##..######..##.
.##.###..#####..##....#####......##..###.##.....
.##..######.##..##....#####......##..###.##.....
.##############.#####.##.###.....##..###.##..##.
.######..######..####.##..###....##..###.##..##.
................................................
"""

_EXIT_LABEL = """
......................
.#####........##......
.#####........#...##..
.###.............###..
.###...##.###.##.####.
.####..#####..##..##..
.###....####..##..##..
.###....####..##..##..
.#####.##.##..##..##..
.#####.##.###.##..##..
......................
"""


def _rows(block: str) -> list[str]:
    return block.strip("\n").split("\n")


def _table(block: str, names: str, h: int, w: int) -> dict[str, np.ndarray]:
    """Parse side by side ASCII glyphs, one space between glyphs, a blank line between bands."""
    rows = _rows(block)
    per_band = (len(rows) + 1) // (h + 1)
    per_row = len(names) // per_band
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        band, col = divmod(i, per_row)
        top = band * (h + 1)
        left = col * (w + 1)
        out[name] = np.array([[c == "#" for c in r[left : left + w]] for r in rows[top : top + h]])
    return out


DIGIT_GLYPHS = _table(_DIGITS, "0123456789", GLYPH_H, GLYPH_W)
TEAMS_GLYPHS = _table(_TEAMS_DIGITS, "1234", TEAMS_DIGIT_SIZE[1], TEAMS_DIGIT_SIZE[0])
TEAMS_LABEL = np.array([[c == "#" for c in r] for r in _rows(_TEAMS_LABEL)])
BACK_LABEL = np.array([[c == "#" for c in r] for r in _rows(_BACK_LABEL)])
EXIT_LABEL = np.array([[c == "#" for c in r] for r in _rows(_EXIT_LABEL)])


@dataclass(frozen=True)
class Reading:
    """One frame's HUD. None means not read, and ``enemy_health`` holds only what did read."""

    teams_left: int | None = None
    self_health: int | None = None
    self_health_fraction: float | None = None
    self_ammo: int | None = None
    cube_count: int | None = None
    enemy_health: tuple[int, ...] = ()
    # No steady HUD source exists: no row of teammates was found, only a passing banner.
    teammates_alive: int | None = None
    # True on a dead label, False on a self bar with no label, None with neither: a label shows a
    # moment after the death, so a missing bar alone does not say dead.
    self_dead: bool | None = None


class _Masks(NamedTuple):
    fill: np.ndarray  # green, yellow or red bar fill
    green: np.ndarray
    orange: np.ndarray
    navy: np.ndarray
    track: np.ndarray  # navy or the lighter reload fill


class _Anchor(NamedTuple):
    x: int
    y: int
    strip: bool  # the ammo strip was found under the bar


def _in(channel: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    return (channel >= span[0]) & (channel <= span[1])


def _masks(hsv: np.ndarray) -> _Masks:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = _in(h, GREEN_H) & (s >= GREEN_S) & (v >= GREEN_V)
    yellow = _in(h, YELLOW_H) & (s >= YELLOW_S) & (v >= YELLOW_V)
    red = ((h <= RED_LOW_H) | (h >= RED_HIGH_H)) & (s >= RED_S) & (v >= RED_V)
    orange = _in(h, ORANGE_H) & (s >= ORANGE_S) & (v >= ORANGE_V)
    blue = _in(h, NAVY_H) & (s >= NAVY_S)
    navy = blue & _in(v, NAVY_V)
    track = blue & (v >= NAVY_V[0]) & (v <= RELOAD_V)
    return _Masks(green | yellow | red, green, orange, navy, track)


def _score(mask: np.ndarray, w: int, h: int, dy: int) -> np.ndarray:
    """Share of the w x h rectangle ``dy`` rows below each pixel that is set in ``mask``."""
    blurred = cv2.blur(mask.astype(np.float32), (w, h), borderType=cv2.BORDER_CONSTANT)
    return np.roll(blurred, (-(h // 2) - dy, -(w // 2)), axis=(0, 1))


def _best(score: np.ndarray, window: tuple[int, int, int, int]) -> tuple[float, int, int]:
    x0, x1, y0, y1 = window
    crop = score[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0, 0, 0
    y, x = np.unravel_index(int(np.argmax(crop)), crop.shape)
    return float(crop[y, x]), x0 + int(x), y0 + int(y)


def _anchors(masks: _Masks, boxes: Sequence[Box]) -> list[_Anchor]:
    """Where the self bar may be, best guess first: under the detector's self box when there is
    one, then the fallback around the screen centre. A green bar can stand without the strip, a
    drained bar has only the strip and the dark track to go on."""
    strip = _score(masks.orange | masks.navy, BAR_W, STRIP_H, STRIP_DY)
    body = _score(masks.fill | masks.navy, BAR_W, BODY_H, BODY_DY)
    left = _score(masks.fill, 4, BODY_H, BODY_DY) >= 0.6
    unit = np.minimum(strip, body) * left
    solid = _score(masks.green, BAR_W, BODY_H, BODY_DY)

    windows = []
    selves = [b for b in boxes if b.cls == "self"]
    if selves:
        b = max(selves, key=lambda s: s.score)
        x, y = int(b.x + b.w / 2 - BAR_W / 2), int(b.y + b.h / 2 + BOX_BAR_DY)
        windows.append((x - BOX_SLACK, x + BOX_SLACK, y - BOX_SLACK, y + BOX_SLACK))
    windows.append(CENTRE_WINDOW)

    found: list[_Anchor] = []
    for x0, x1, y0, y1 in windows:
        window = (
            max(x0, 0),
            min(x1, FRAME_W - BAR_W),
            max(y0, 0),
            min(y1, FRAME_H - STRIP_DY - STRIP_H),
        )
        score, x, y = _best(unit, window)
        if score >= UNIT_MIN:
            found.append(_Anchor(x, y, True))
        score, x, y = _best(solid, window)
        if score >= GREEN_SOLID:
            found.append(_Anchor(x, y, False))
    return found


def _fill(masks: _Masks, anchor: _Anchor) -> float | None:
    """The bar's fill as a fraction, from the left end to the first column not filled."""
    x, y = anchor.x, anchor.y
    cols = masks.fill[y + BODY_DY : y + BODY_DY + BODY_H, x : x + BAR_W].mean(axis=0) >= 0.5
    if not cols[:4].any():
        return None
    start = int(np.argmax(cols))
    gaps = np.flatnonzero(~cols[start:])
    run = start + (int(gaps[0]) if gaps.size else BAR_W - start)
    fraction = (run + FILL_SHORT) / BAR_W
    if fraction >= FULL_FILL:
        return 1.0
    rest = masks.track[y + BODY_DY : y + BODY_DY + BODY_H, x + run + 2 : x + BAR_W]
    if rest.size and rest.mean() < TRACK_MIN:
        return None
    return round(fraction, 3)


def _ammo(masks: _Masks, anchor: _Anchor) -> int | None:
    """Loaded segments, left to right. A partly reloaded segment does not count."""
    y0, y1 = anchor.y + STRIP_DY, anchor.y + STRIP_DY + STRIP_H
    loaded = 0
    for i in range(SEGMENTS):
        x0 = anchor.x + SEGMENT_DX + int(SEGMENT_PITCH * i)
        orange = masks.orange[y0:y1, x0 : x0 + SEGMENT_W].mean()
        if orange >= SEGMENT_FULL:
            if loaded != i:
                return None
            loaded += 1
        elif masks.track[y0:y1, x0 : x0 + SEGMENT_W].mean() + orange < SEGMENT_EMPTY:
            return None
    return loaded


def _canvas(blob: np.ndarray) -> np.ndarray:
    """A component scaled to the glyph height and centred, so a narrow 1 stays narrow."""
    h, w = blob.shape
    width = min(GLYPH_W, max(1, round(w * GLYPH_H / h)))
    scaled = cv2.resize(blob.astype(np.uint8) * 255, (width, GLYPH_H), interpolation=cv2.INTER_AREA)
    out = np.zeros((GLYPH_H, GLYPH_W), dtype=bool)
    left = (GLYPH_W - width) // 2
    out[:, left : left + width] = scaled >= 128
    return out


def _glyph(blob: np.ndarray) -> str | None:
    canvas = _canvas(blob)
    scores = sorted(((canvas == g).mean(), d) for d, g in DIGIT_GLYPHS.items())
    (second, _), (best, digit) = scores[-2], scores[-1]
    return digit if best >= GLYPH_MIN and best - second >= GLYPH_MARGIN else None


def _components(
    mask: np.ndarray, heights: tuple[int, int], widths: tuple[int, int]
) -> list[tuple[int, int, int, int, np.ndarray]]:
    """(x, y, w, h, blob) of each component of a plausible size, left to right."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i, :4])
        if heights[0] <= h <= heights[1] and widths[0] <= w <= widths[1]:
            out.append((x, y, w, h, labels[y : y + h, x : x + w] == i))
    out.sort(key=lambda c: c[0])
    return out


def _number(
    frame: np.ndarray, x0: int, x1: int, y0: int, y1: int, centre: float | None
) -> int | None:
    """The white number inside a box. Every digit-sized thing in it has to read as a digit."""
    x0, y0 = max(x0, 0), max(y0, 0)
    white = frame[y0:y1, x0:x1].min(axis=2) >= DIGIT_WHITE
    comps = _components(white, DIGIT_H, DIGIT_W)
    if not DIGIT_COUNT[0] <= len(comps) <= DIGIT_COUNT[1]:
        return None
    for a, b in zip(comps, comps[1:], strict=False):
        if b[0] - (a[0] + a[2]) > DIGIT_GAP or abs(a[1] - b[1]) > DIGIT_ROW_SLACK:
            return None
    if centre is not None:
        middle = x0 + (comps[0][0] + comps[-1][0] + comps[-1][2]) / 2
        if abs(middle - centre) > DIGIT_CENTRE_SLACK:
            return None
    digits = [_glyph(c[4]) for c in comps]
    if None in digits or digits[0] == "0":
        return None
    return int("".join(d for d in digits if d))


def _white_text(
    frame: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int]
) -> np.ndarray:
    x0, y0, x1, y1 = box
    white = (frame[y0:y1, x0:x1].min(axis=2) >= TEAMS_WHITE).astype(np.uint8) * 255
    return cv2.resize(white, size, interpolation=cv2.INTER_AREA) >= 128


def _teams_left(frame: np.ndarray) -> int | None:
    label = _white_text(frame, TEAMS_LABEL_BOX, TEAMS_LABEL_SIZE)
    if (label == TEAMS_LABEL).mean() < TEAMS_LABEL_MIN:
        return None
    digit = _white_text(frame, TEAMS_DIGIT_BOX, TEAMS_DIGIT_SIZE)
    scores = sorted(((digit == g).mean(), d) for d, g in TEAMS_GLYPHS.items())
    (second, _), (best, name) = scores[-2], scores[-1]
    if best < TEAMS_DIGIT_MIN or best - second < TEAMS_DIGIT_MARGIN:
        return None
    return int(name)


def _dead(frame: np.ndarray) -> bool:
    """Either dead label on screen. These are two small fixed crops, so every frame can afford
    them, and a watched teammate's bar in the middle of the screen must not hide them."""
    back = _white_text(frame, BACK_LABEL_BOX, BACK_LABEL_SIZE)
    if (back == BACK_LABEL).mean() >= BACK_LABEL_MIN:
        return True
    leave = _white_text(frame, EXIT_LABEL_BOX, EXIT_LABEL_SIZE)
    return bool((leave == EXIT_LABEL).mean() >= EXIT_LABEL_MIN)


def spectating(frame: np.ndarray) -> bool:
    """The whole team is out: the spectate camera's "Exit" label is up and the respawn
    "Back in:" label is not. "Back in" means a teammate lives and we come back, so it is
    never a reason to leave. The same two small crops as _dead, so the farm can call this
    on every in-match frame."""
    back = _white_text(frame, BACK_LABEL_BOX, BACK_LABEL_SIZE)
    if (back == BACK_LABEL).mean() >= BACK_LABEL_MIN:
        return False
    leave = _white_text(frame, EXIT_LABEL_BOX, EXIT_LABEL_SIZE)
    return bool((leave == EXIT_LABEL).mean() >= EXIT_LABEL_MIN)


def _cube_count(hsv: np.ndarray, anchor: _Anchor) -> int | None:
    """The cube digits, only beside a badge. The badge does hide at zero, but a covered badge
    looks the same, so no badge reads as None and not as zero."""
    x, y = anchor.x, anchor.y
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    x0, x1, y0, y1 = BADGE_BOX
    top, bottom, left, right = max(y + y0, 0), max(y + y1, 0), max(x + x0, 0), x + x1
    badge = _in(h, BADGE_H) & (s >= BADGE_S) & (v >= BADGE_V)
    if badge[top:bottom, left:right].sum() < BADGE_MIN_PIXELS:
        return None
    x0, x1, y0, y1 = CUBE_BOX
    top, bottom, left, right = max(y + y0, 0), max(y + y1, 0), max(x + x0, 0), x + x1
    green = _in(h, CUBE_GREEN_H) & (s >= CUBE_GREEN_S) & (v >= CUBE_GREEN_V)
    comps = _components(green[top:bottom, left:right], CUBE_H, CUBE_W)
    comps = [c for c in comps if c[4].mean() >= CUBE_MIN_FILL]
    if not 1 <= len(comps) <= 2:
        return None
    digits = [_glyph(c[4]) for c in comps]
    if None in digits or digits[0] == "0":
        return None
    return int("".join(d for d in digits if d))


def _self(
    frame: np.ndarray, masks: _Masks, boxes: Sequence[Box]
) -> tuple[_Anchor, int | None, float | None] | None:
    """The first candidate bar that holds up, with its number and fill."""
    for anchor in _anchors(masks, boxes):
        fill = _fill(masks, anchor)
        x0, x1 = anchor.x - 6, anchor.x + BAR_W + 6
        y0, y1 = anchor.y + DIGIT_BAND[0], anchor.y + DIGIT_BAND[1]
        health = _number(frame, x0, x1, y0, y1, anchor.x + BAR_W / 2)
        if health is not None or (anchor.strip and fill is not None and fill >= TRUST_FILL):
            return anchor, health, fill
    return None


def _enemies(frame: np.ndarray, masks: _Masks, skip: _Anchor | None) -> tuple[int, ...]:
    """Numbers on bars that are neither ours nor a teammate's, in frame order."""
    x0, x1, y0, y1 = OTHER_AREA
    colour = (masks.orange | masks.fill) & ~masks.green
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        colour[y0:y1, x0:x1].astype(np.uint8), connectivity=4
    )
    bars = []
    for i in range(1, n):
        x, y, w, h, area = (int(q) for q in stats[i])
        if not (OTHER_H[0] <= h <= OTHER_H[1] and OTHER_W[0] <= w <= OTHER_W[1]):
            continue
        if area < OTHER_SOLID * w * h:
            continue
        x, y = x + x0, y + y0
        if skip is not None and -8 <= x - skip.x <= BAR_W and -8 <= y - skip.y <= 40:
            continue
        bars.append((x, y, w))
    out: list[int] = []
    read_bars: list[tuple[int, int]] = []
    for x, y, w in sorted(bars):
        # Pieces of one split bar reach the same number: same row, and together one bar wide.
        if any(abs(y - ry) < OTHER_H[0] and x + w - rx <= OTHER_W[1] for rx, ry in read_bars):
            continue
        value = _number(
            frame, x - 6, x + OTHER_W[1] + 6, y + OTHER_BAND[0], y + OTHER_BAND[1], None
        )
        if value is not None:
            out.append(value)
            read_bars.append((x, y))
    return tuple(out)


def read(frame: np.ndarray, boxes: Sequence[Box] = ()) -> Reading:
    """Read one BGR frame. ``boxes`` is the detector's output for it, which may be empty."""
    if frame.shape != (FRAME_H, FRAME_W, 3):
        return Reading()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = _masks(hsv)
    teams = _teams_left(frame)
    dead = _dead(frame)
    found = _self(frame, masks, boxes)
    if found is None:
        return Reading(
            teams_left=teams, enemy_health=_enemies(frame, masks, None), self_dead=dead or None
        )
    anchor, health, fill = found
    return Reading(
        teams_left=teams,
        self_health=health,
        self_health_fraction=fill,
        self_ammo=_ammo(masks, anchor) if anchor.strip else None,
        cube_count=_cube_count(hsv, anchor),
        enemy_health=_enemies(frame, masks, anchor),
        self_dead=dead,
    )
