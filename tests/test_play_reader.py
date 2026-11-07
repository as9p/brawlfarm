"""The HUD reader on frames drawn here: the digits come from the reader's own glyph tables, the
bars from its own constants. No real frame is ever in the repository. What matters most is the
other half of each test: what cannot be read is None, never a guess."""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest

from brawlfarm.play import reader
from brawlfarm.play.detect import Box

FLOOR = (120, 40, 90)  # a purple the reader has no mask for
NAVY = (65, 44, 36)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (60, 220, 80)
YELLOW = (40, 210, 235)
RED = (40, 40, 230)
CYAN = (255, 220, 40)
ORANGE = (30, 120, 240)
RELOAD = (150, 110, 90)
BADGE = (30, 220, 240)
CUBE_GREEN = (60, 230, 70)
BAR = (755, 350)


def blank(floor: tuple[int, ...] = FLOOR) -> np.ndarray:
    return np.full((reader.FRAME_H, reader.FRAME_W, 3), floor, dtype=np.uint8)


def paint(frame: np.ndarray, x: int, y: int, mask: np.ndarray, colour: tuple[int, ...]) -> None:
    """``mask`` in ``colour`` with a one pixel black outline."""
    pad = np.pad(mask.astype(np.uint8), 1)
    h, w = pad.shape
    region = frame[y - 1 : y - 1 + h, x - 1 : x - 1 + w]
    region[cv2.dilate(pad, np.ones((3, 3), np.uint8)) > 0] = BLACK
    region[pad > 0] = colour


def glyph(digit: str, height: int) -> np.ndarray:
    g = reader.DIGIT_GLYPHS[digit]
    g = g[:, g.any(axis=0)]
    width = round(g.shape[1] * height / g.shape[0])
    return cv2.resize(g.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0


def text(
    frame: np.ndarray, value: int, centre: int, top: int, height: int, colour: tuple[int, ...]
) -> None:
    masks = [glyph(d, height) for d in str(value)]
    x = centre - (sum(m.shape[1] for m in masks) + 3 * (len(masks) - 1)) // 2
    for m in masks:
        paint(frame, x, top, m, colour)
        x += m.shape[1] + 3


def put(frame: np.ndarray, box: tuple[int, int, int, int], table: np.ndarray) -> None:
    x0, y0, x1, y1 = box
    big = cv2.resize(table.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    frame[y0:y1, x0:x1][big > 0] = WHITE


def teams(frame: np.ndarray, value: int | None = 3) -> None:
    put(frame, reader.TEAMS_LABEL_BOX, reader.TEAMS_LABEL)
    if value is not None:
        put(frame, reader.TEAMS_DIGIT_BOX, reader.TEAMS_GLYPHS[str(value)])


def self_unit(
    frame: np.ndarray,
    bar: tuple[int, int] = BAR,
    *,
    health: int | None = 5400,
    fill: float = 1.0,
    colour: tuple[int, ...] = GREEN,
    ammo: float = 3,
    strip: bool = True,
    tall: int = 17,
) -> None:
    x, y = bar
    w, t = reader.BAR_W, tall
    frame[y - 2 : y + t + 2, x - 2 : x + w + 2] = BLACK
    frame[y : y + t, x : x + w] = NAVY
    frame[y : y + t, x : x + round(w * fill)] = colour
    if strip:
        frame[y + 20 : y + 33, x - 1 : x + w + 1] = BLACK
        frame[y + 22 : y + 31, x : x + w] = NAVY
        for i in range(reader.SEGMENTS):
            sx = x + 1 + int(reader.SEGMENT_PITCH * i)
            part = min(max(ammo - i, 0.0), 1.0)
            frame[y + 22 : y + 31, sx : sx + 28] = NAVY
            done = RELOAD if part < 1 else ORANGE
            frame[y + 22 : y + 31, sx : sx + round(28 * part)] = done
    if health is not None:
        text(frame, health, x + w // 2, y - 7, 13, WHITE)


def cubes(frame: np.ndarray, value: int, bar: tuple[int, int] = BAR, *, badge: bool = True) -> None:
    x, y = bar
    if badge:
        centre = (x + 22, y - 60)
        cv2.fillPoly(
            frame,
            [
                np.array(
                    [
                        (centre[0] - 15, centre[1]),
                        (centre[0], centre[1] - 15),
                        (centre[0] + 15, centre[1]),
                        (centre[0], centre[1] + 15),
                    ]
                )
            ],
            BADGE,
        )
    text(frame, value, x + 60, y - 66, 19, CUBE_GREEN)


def enemy(frame: np.ndarray, x: int, y: int, value: int, colour: tuple[int, ...] = RED) -> None:
    w = 76
    frame[y - 2 : y + 14, x - 2 : x + w + 2] = BLACK
    frame[y : y + 12, x : x + w] = colour
    text(frame, value, x + w // 2, y - 8, 13, WHITE)


def hud() -> np.ndarray:
    frame = blank()
    teams(frame)
    self_unit(frame)
    return frame


@pytest.mark.parametrize("value", [1, 2, 3, 4])
def test_teams_left_reads_every_digit(value: int) -> None:
    frame = blank()
    teams(frame, value)
    assert reader.read(frame).teams_left == value


def test_a_label_with_no_digit_is_none() -> None:
    frame = blank()
    teams(frame, None)
    assert reader.read(frame).teams_left is None


def test_a_digit_with_no_label_is_none() -> None:
    frame = blank()
    teams(frame, 2)
    frame[:, :250] = FLOOR
    assert reader.read(frame).teams_left is None


@pytest.mark.parametrize("floor", [FLOOR, GREEN, RED, NAVY, RELOAD])
def test_an_empty_frame_reads_nothing(floor: tuple[int, ...]) -> None:
    r = reader.read(blank(floor))
    assert r == reader.Reading()
    assert all(v is None for k, v in dataclasses.asdict(r).items() if k != "enemy_health")
    assert r.enemy_health == ()


def test_a_frame_of_any_other_size_reads_nothing() -> None:
    # The HUD stays at its own coordinates, so only the size guard can refuse these.
    frame = hud()
    assert reader.read(frame).teams_left == 3
    assert reader.read(frame[:800, :1500]) == reader.Reading()
    assert reader.read(np.pad(frame, ((0, 100), (0, 100), (0, 0)))) == reader.Reading()


def test_a_grayscale_frame_of_the_right_size_reads_nothing() -> None:
    frame = cv2.cvtColor(hud(), cv2.COLOR_BGR2GRAY)
    assert frame.shape == (reader.FRAME_H, reader.FRAME_W)
    assert reader.read(frame) == reader.Reading()


def test_a_bgra_frame_of_the_right_size_reads_nothing() -> None:
    frame = cv2.cvtColor(hud(), cv2.COLOR_BGR2BGRA)
    assert frame.shape == (reader.FRAME_H, reader.FRAME_W, 4)
    assert reader.read(frame) == reader.Reading()


def test_reading_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        reader.Reading().teams_left = 3  # type: ignore[misc]


@pytest.mark.parametrize("value", [800, 4200, 5400, 6789, 11000])
def test_full_bar_found_by_the_centre_fallback_with_no_boxes(value: int) -> None:
    frame = blank()
    self_unit(frame, health=value)
    r = reader.read(frame)
    assert r.self_health == value
    assert r.self_health_fraction == 1.0
    assert r.self_ammo == 3


def test_every_digit_reads() -> None:
    for value in (1234, 5678, 9012):
        frame = blank()
        self_unit(frame, health=value)
        assert reader.read(frame).self_health == value


def test_the_self_box_finds_a_bar_the_centre_window_cannot() -> None:
    frame = blank()
    bar = (1100, 500)
    self_unit(frame, bar)
    assert reader.read(frame).self_health is None
    box = Box(
        "self", 0.9, bar[0] + reader.BAR_W / 2 - 50, bar[1] - reader.BOX_BAR_DY - 50, 100, 100
    )
    assert reader.read(frame, [box]).self_health == 5400


def test_a_stray_self_box_falls_back_to_the_centre() -> None:
    frame = blank()
    self_unit(frame)
    assert reader.read(frame, [Box("self", 0.6, 100, 700, 90, 90)]).self_health == 5400


def test_another_class_of_box_is_not_a_self_anchor() -> None:
    frame = blank()
    bar = (1100, 500)
    self_unit(frame, bar)
    box = Box("enemy", 0.9, bar[0], bar[1] + 30, 90, 90)
    assert reader.read(frame, [box]).self_health is None


@pytest.mark.parametrize(("fill", "colour"), [(0.8, GREEN), (0.5, YELLOW), (0.2, RED), (0.08, RED)])
def test_part_bars_measure_their_fill_in_any_colour(fill: float, colour: tuple[int, ...]) -> None:
    frame = blank()
    self_unit(frame, health=2345, fill=fill, colour=colour)
    r = reader.read(frame)
    assert r.self_health == 2345
    assert r.self_health_fraction == pytest.approx(fill, abs=0.05)


def test_a_part_bar_without_digits_is_not_believed() -> None:
    frame = blank()
    self_unit(frame, health=None, fill=0.3)
    r = reader.read(frame)
    assert r.self_health is None
    assert r.self_health_fraction is None
    assert r.self_ammo is None


def test_a_full_bar_without_digits_keeps_its_fill() -> None:
    frame = blank()
    self_unit(frame, health=None)
    r = reader.read(frame)
    assert r.self_health is None
    assert r.self_health_fraction == 1.0


def test_a_covered_bar_has_no_fill() -> None:
    frame = blank()
    self_unit(frame, fill=0.6)
    frame[BAR[1] + 6 : BAR[1] + 17, BAR[0] + 73 : BAR[0] + 90] = (200, 30, 200)
    r = reader.read(frame)
    assert r.self_health == 5400
    assert r.self_health_fraction is None


@pytest.mark.parametrize(("ammo", "seen"), [(0, 0), (1, 1), (2, 2), (3, 3), (1.5, 1), (0.2, 0)])
def test_ammo_counts_loaded_segments_only(ammo: float, seen: int) -> None:
    frame = blank()
    self_unit(frame, ammo=ammo)
    assert reader.read(frame).self_ammo == seen


@pytest.mark.parametrize("floor", [FLOOR, RELOAD])
def test_a_hidden_strip_is_no_ammo_reading(floor: tuple[int, ...]) -> None:
    # RELOAD is a floor the track mask matches, so the segments alone would read as empty.
    frame = blank(floor)
    self_unit(frame, strip=False)
    r = reader.read(frame)
    assert r.self_health == 5400
    assert r.self_ammo is None


def test_a_covered_segment_is_no_ammo_reading() -> None:
    frame = blank()
    self_unit(frame)
    frame[BAR[1] + 22 : BAR[1] + 31, BAR[0] + 64 : BAR[0] + 79] = (200, 30, 200)
    r = reader.read(frame)
    assert r.self_health == 5400
    assert r.self_ammo is None


@pytest.mark.parametrize("value", [1, 4, 7, 13])
def test_cube_count_beside_the_badge(value: int) -> None:
    frame = blank()
    self_unit(frame)
    cubes(frame, value)
    assert reader.read(frame).cube_count == value


def test_no_badge_is_none_and_not_zero() -> None:
    frame = blank()
    self_unit(frame)
    assert reader.read(frame).cube_count is None
    cubes(frame, 4, badge=False)
    assert reader.read(frame).cube_count is None


def test_no_self_bar_means_no_cube_reading() -> None:
    frame = blank()
    cubes(frame, 4)
    assert reader.read(frame).cube_count is None


def test_enemy_numbers_read_on_their_bars() -> None:
    frame = blank()
    enemy(frame, 300, 300, 8351, RED)
    enemy(frame, 1100, 500, 4480, YELLOW)
    assert reader.read(frame).enemy_health == (8351, 4480)


@pytest.mark.parametrize(("second", "row"), [(1100, 500), (400, 300)])
def test_two_enemies_at_equal_health_both_count(second: int, row: int) -> None:
    frame = blank()
    enemy(frame, 300, 300, 3200)
    enemy(frame, second, row, 3200, YELLOW)
    assert reader.read(frame).enemy_health == (3200, 3200)


def test_one_enemy_on_a_split_bar_counts_once() -> None:
    frame = blank()
    enemy(frame, 300, 300, 835)
    frame[300:312, 316:318] = BLACK
    assert reader.read(frame).enemy_health == (835,)


@pytest.mark.parametrize("colour", [CYAN, GREEN])
def test_a_teammate_bar_is_not_an_enemy(colour: tuple[int, ...]) -> None:
    frame = blank()
    enemy(frame, 300, 300, 5040, colour)
    assert reader.read(frame).enemy_health == ()


def test_the_self_bar_is_not_an_enemy() -> None:
    frame = blank()
    # Short enough to pass the size check for another player's bar, so only the skip refuses it.
    self_unit(frame, fill=0.4, colour=RED, health=2500, tall=14)
    r = reader.read(frame)
    assert r.self_health == 2500
    assert r.enemy_health == ()


def test_teammates_alive_has_no_source() -> None:
    frame = blank()
    teams(frame)
    self_unit(frame)
    assert reader.read(frame).teammates_alive is None


DEAD_LABELS = [
    (reader.BACK_LABEL_BOX, reader.BACK_LABEL),
    (reader.EXIT_LABEL_BOX, reader.EXIT_LABEL),
]


@pytest.mark.parametrize(("box", "table"), DEAD_LABELS)
def test_a_dead_label_reads_dead_with_no_bar(box, table) -> None:
    frame = blank()
    teams(frame)
    put(frame, box, table)
    reading = reader.read(frame)
    assert reading.self_dead is True
    assert reading.teams_left == 3
    assert reading.self_health is None


@pytest.mark.parametrize(("box", "table"), DEAD_LABELS)
def test_a_dead_label_beats_a_bar_on_screen(box, table) -> None:
    """A watched teammate's bar sits where ours would."""
    frame = blank()
    self_unit(frame)
    put(frame, box, table)
    assert reader.read(frame).self_dead is True


def test_a_self_bar_with_no_label_is_alive() -> None:
    frame = blank()  # the bar's digits land inside the Back in box
    teams(frame)
    self_unit(frame)
    assert reader.read(frame).self_dead is False


@pytest.mark.parametrize("floor", [FLOOR, WHITE, BLACK])
def test_no_bar_and_no_label_is_not_read(floor: tuple[int, ...]) -> None:
    frame = blank(floor)
    teams(frame)
    assert reader.read(frame).self_dead is None


def test_the_countdown_popping_in_oversized_still_reads_dead() -> None:
    x0, y0, x1, y1 = reader.BACK_LABEL_BOX
    frame = blank()
    put(frame, (x0 - 3, y0 - 1, x1 + 3, y1 + 1), reader.BACK_LABEL)
    assert reader.read(frame).self_dead is True


@pytest.mark.parametrize(("box", "table"), DEAD_LABELS)
def test_a_label_off_its_spot_is_not_dead(box, table) -> None:
    x0, y0, x1, y1 = box
    frame = blank()
    put(frame, (x0 + 200, y0 - 100, x1 + 200, y1 - 100), table)
    assert reader.read(frame).self_dead is None


def test_read_leaves_the_frame_alone() -> None:
    frame = blank()
    self_unit(frame)
    before = frame.copy()
    reader.read(frame)
    assert np.array_equal(frame, before)
