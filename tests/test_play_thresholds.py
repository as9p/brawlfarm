"""Per-class thresholds, the ONNX decode and the preprocessing, on hand-built numbers only.

Nothing here loads a model: the arrays are small enough to work the expected boxes out by hand,
which is the point. The two training scripts are only read as text, never imported, because
importing them would pull in torch.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.play import classes, export, thresholds

SCRIPTS = Path(__file__).resolve().parents[1] / "tools" / "play"


def _det(class_index: int, score: float, box: tuple[float, float, float, float]) -> dict:
    x, y, w, h = box
    return {"class_index": class_index, "score": score, "x": x, "y": y, "w": w, "h": h}


def test_iou_identical_boxes_is_one():
    assert thresholds.iou((10, 20, 30, 40), (10, 20, 30, 40)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert thresholds.iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    assert thresholds.iou((0, 0, 10, 10), (10, 0, 10, 10)) == 0.0


def test_iou_half_overlap():
    # 50 pixels of overlap over a union of 150
    assert thresholds.iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)


def test_match_detections_does_not_match_one_truth_box_twice():
    truth = [{"class_index": 0, "x": 0, "y": 0, "w": 10, "h": 10}]
    dets = [
        _det(0, 0.9, (0, 0, 10, 10)),
        _det(0, 0.8, (1, 1, 10, 10)),
    ]
    assert thresholds.match_detections(dets, truth) == [True, False]


def test_match_detections_does_not_match_across_classes():
    truth = [{"class_index": 1, "x": 0, "y": 0, "w": 10, "h": 10}]
    dets = [_det(0, 0.9, (0, 0, 10, 10))]
    assert thresholds.match_detections(dets, truth) == [False]


def test_match_detections_takes_the_better_box_for_the_higher_score():
    truth = [
        {"class_index": 0, "x": 0, "y": 0, "w": 10, "h": 10},
        {"class_index": 0, "x": 100, "y": 0, "w": 10, "h": 10},
    ]
    dets = [
        _det(0, 0.9, (101, 0, 10, 10)),
        _det(0, 0.7, (0, 0, 10, 10)),
        _det(0, 0.5, (400, 400, 10, 10)),
    ]
    assert thresholds.match_detections(dets, truth) == [True, True, False]


def test_pick_returns_the_floor_when_every_detection_is_true():
    scores = [0.9, 0.6, 0.35]
    assert thresholds.pick(scores, [True, True, True]) == thresholds.FLOOR


def test_pick_clears_one_low_scored_false_positive():
    scores = [0.91, 0.82, 0.73, 0.40]
    is_tp = [True, True, True, False]
    assert thresholds.pick(scores, is_tp) == pytest.approx(0.41)


def test_pick_returns_the_ceiling_without_detections():
    assert thresholds.pick([], []) == thresholds.CEILING


def test_pick_returns_the_ceiling_when_precision_is_never_reached():
    assert thresholds.pick([0.9, 0.8], [False, False]) == thresholds.CEILING


# precision 0.90 from 0.30 up, and 0.98 only once both false positives are cleared at 0.61
_NINETY_SCORES = [0.9] * 9 + [0.35] * 9 + [0.6, 0.4]
_NINETY_HITS = [True] * 18 + [False, False]


def test_target_for_keeps_tap_anchors_at_the_strict_precision():
    assert thresholds.target_for("play_button") == thresholds.MIN_PRECISION == 0.98
    assert thresholds.target_for("skull_star", other=0.5) == thresholds.MIN_PRECISION


def test_target_for_gives_other_classes_the_looser_precision():
    assert thresholds.target_for("self") == thresholds.MIN_PRECISION_OTHER == 0.90
    assert thresholds.target_for("enemy", other=0.8) == 0.8


def test_pick_thresholds_splits_tap_anchors_from_other_classes():
    names = ["self", "play_button"]
    scores = {0: _NINETY_SCORES, 1: _NINETY_SCORES}
    hits = {0: _NINETY_HITS, 1: _NINETY_HITS}
    picked, validation = export.pick_thresholds(
        names, scores, hits, {0: 18, 1: 18}, thresholds.MIN_PRECISION_OTHER
    )
    assert picked["self"] == thresholds.FLOOR
    assert picked["play_button"] == pytest.approx(0.61)
    assert validation["self"]["min_precision"] == 0.90
    assert validation["play_button"]["min_precision"] == 0.98
    assert validation["self"]["precision_at_threshold"] == 0.9


def test_export_flag_sets_the_other_precision_and_is_passed_through():
    assert export.parser().parse_args([]).min_precision_other == thresholds.MIN_PRECISION_OTHER
    args = export.parser().parse_args(["--min-precision-other", "1"])
    assert args.min_precision_other == 1.0
    picked, validation = export.pick_thresholds(
        ["self", "play_button"],
        {0: _NINETY_SCORES, 1: _NINETY_SCORES},
        {0: _NINETY_HITS, 1: _NINETY_HITS},
        {0: 18, 1: 18},
        args.min_precision_other,
    )
    assert picked["self"] == pytest.approx(0.61)
    assert validation["self"]["min_precision"] == 1.0
    assert validation["play_button"]["min_precision"] == thresholds.MIN_PRECISION


@pytest.mark.parametrize("value", ["0", "-0.5", "1.01", "abc"])
def test_export_flag_rejects_a_precision_outside_zero_to_one(value):
    with pytest.raises(SystemExit) as raised:
        export.parser().parse_args(["--min-precision-other", value])
    assert raised.value.code == 2


def test_export_flags_for_the_other_classes_exclude_each_other():
    assert export.parser().parse_args([]).best_f1_other is False
    assert export.parser().parse_args(["--best-f1-other"]).best_f1_other is True
    with pytest.raises(SystemExit) as raised:
        export.parser().parse_args(["--best-f1-other", "--min-precision-other", "0.8"])
    assert raised.value.code == 2


# 18 of 20 boxes true against 30 real ones is F1 0.72 up to 0.35, against 0.46 at 0.61, where the
# precision is clean and half the truth is gone. 0.35 is the top of that tie.
def test_pick_best_f1_trades_precision_for_recall():
    assert thresholds.pick_best_f1("enemy", _NINETY_SCORES, _NINETY_HITS, 30) == 0.35
    assert thresholds.pick(_NINETY_SCORES, _NINETY_HITS) == pytest.approx(0.61)


def test_pick_best_f1_counts_the_truth_no_detection_reached():
    scores = [0.9] * 30 + [0.5] * 30
    hits = [True] * 30 + [True] * 10 + [False] * 20
    assert thresholds.pick_best_f1("enemy", scores, hits, 40) == pytest.approx(0.90)
    assert thresholds.pick_best_f1("enemy", scores, hits, 200) == pytest.approx(0.50)


def test_pick_best_f1_finds_a_threshold_above_the_noise():
    scores = [0.9] * 30 + [0.5] * 30
    hits = [True] * 30 + [False] * 30
    assert thresholds.pick_best_f1("enemy", scores, hits, 30) == pytest.approx(0.90)


def test_pick_best_f1_takes_the_higher_threshold_on_a_tie():
    assert thresholds.pick_best_f1("enemy", [0.7] * 30, [True] * 30, 30) == pytest.approx(0.70)


def test_pick_best_f1_returns_the_ceiling_with_too_little_truth():
    scores = [0.9] * 29
    hits = [True] * 29
    assert thresholds.pick_best_f1("enemy", scores, hits, 29) == thresholds.CEILING


def test_pick_best_f1_returns_the_ceiling_without_a_true_positive():
    assert thresholds.pick_best_f1("enemy", [], [], 30) == thresholds.CEILING
    assert thresholds.pick_best_f1("enemy", [0.9, 0.8], [False, False], 30) == thresholds.CEILING


@pytest.mark.parametrize("name", sorted(classes.TAP_ANCHORS))
def test_best_f1_can_never_apply_to_a_tap_anchor(name):
    assert thresholds.rule_for(name, best_f1=True) == thresholds.RULE_PRECISION
    with pytest.raises(ValueError, match="tap anchor"):
        thresholds.pick_best_f1(name, _NINETY_SCORES, _NINETY_HITS, 30)

    # scores where the two rules disagree, so a leak would move the number
    asked = export.pick_thresholds(
        [name], {0: _NINETY_SCORES}, {0: _NINETY_HITS}, {0: 30}, 0.5, best_f1=True
    )
    plain = export.pick_thresholds([name], {0: _NINETY_SCORES}, {0: _NINETY_HITS}, {0: 30}, 0.5)
    assert asked == plain
    assert asked[0][name] == pytest.approx(0.61)
    assert asked[1][name]["min_precision"] == thresholds.MIN_PRECISION
    assert asked[1][name]["rule"] == thresholds.RULE_PRECISION


def test_every_class_has_exactly_one_rule_under_best_f1():
    for name in classes.CLASSES:
        expected = (
            thresholds.RULE_PRECISION if name in classes.TAP_ANCHORS else thresholds.RULE_BEST_F1
        )
        assert thresholds.rule_for(name, best_f1=True) == expected
        assert thresholds.rule_for(name) == thresholds.RULE_PRECISION


def test_pick_thresholds_records_the_rule_of_a_best_f1_class():
    picked, validation = export.pick_thresholds(
        ["self", "play_button"],
        {0: _NINETY_SCORES, 1: _NINETY_SCORES},
        {0: _NINETY_HITS, 1: _NINETY_HITS},
        {0: 30, 1: 30},
        thresholds.MIN_PRECISION_OTHER,
        best_f1=True,
    )
    assert picked == {"self": 0.35, "play_button": pytest.approx(0.61)}
    assert validation["self"]["rule"] == thresholds.RULE_BEST_F1
    assert validation["self"]["certified"] is True
    assert validation["self"]["min_precision"] is None
    assert validation["self"]["f1_at_threshold"] == pytest.approx(2 * 18 / (20 + 30), abs=1e-4)
    assert validation["play_button"]["rule"] == thresholds.RULE_PRECISION
    assert "f1_at_threshold" not in validation["play_button"]


def test_pick_thresholds_records_precision_rule_without_best_f1():
    _picked, validation = export.pick_thresholds(
        ["self"], {0: _NINETY_SCORES}, {0: _NINETY_HITS}, {0: 30}, thresholds.MIN_PRECISION_OTHER
    )
    assert validation["self"]["rule"] == thresholds.RULE_PRECISION


def test_best_f1_class_with_too_little_truth_gets_the_ceiling_uncertified(capsys):
    picked, validation = export.pick_thresholds(
        ["enemy"],
        {0: _NINETY_SCORES},
        {0: _NINETY_HITS},
        {0: thresholds.MIN_TRUTH - 1},
        thresholds.MIN_PRECISION_OTHER,
        best_f1=True,
    )
    assert picked["enemy"] == thresholds.CEILING
    assert validation["enemy"]["certified"] is False
    assert validation["enemy"]["rule"] == thresholds.RULE_CEILING
    assert "WARNING: enemy is not certified: 29 validation truth boxes" in capsys.readouterr().out


def test_best_f1_class_without_a_true_positive_gets_the_ceiling_uncertified(capsys):
    picked, validation = export.pick_thresholds(
        ["enemy"], {0: [0.9]}, {0: [False]}, {0: 500}, 0.9, best_f1=True
    )
    assert picked["enemy"] == thresholds.CEILING
    assert validation["enemy"]["certified"] is False
    assert validation["enemy"]["rule"] == thresholds.RULE_CEILING
    assert "WARNING: enemy is not certified: 0.95 is the fallback" in capsys.readouterr().out


def test_a_precision_fallback_is_recorded_as_the_ceiling():
    _picked, validation = export.pick_thresholds(
        ["play_button"], {0: []}, {0: []}, {0: 500}, thresholds.MIN_PRECISION_OTHER, best_f1=True
    )
    assert validation["play_button"]["rule"] == thresholds.RULE_CEILING


def test_min_truth_is_thirty_and_coco_warns_against_the_same_number():
    from tools.play import coco

    assert thresholds.MIN_TRUTH == 30
    assert coco.TRUTH_MIN == thresholds.MIN_TRUTH


def test_pick_thresholds_certifies_a_class_only_with_enough_validation_truth(capsys):
    picked, validation = export.pick_thresholds(
        ["self", "play_button"],
        {0: _NINETY_SCORES, 1: _NINETY_SCORES},
        {0: _NINETY_HITS, 1: _NINETY_HITS},
        {0: thresholds.MIN_TRUTH, 1: thresholds.MIN_TRUTH - 1},
        thresholds.MIN_PRECISION_OTHER,
    )
    assert validation["self"]["truth"] == 30
    assert validation["self"]["certified"] is True
    assert validation["play_button"]["truth"] == 29
    assert validation["play_button"]["certified"] is False
    # an uncertified threshold is still written: certification is a label, not a veto
    assert picked["play_button"] == pytest.approx(0.61)
    out = capsys.readouterr().out
    assert "WARNING: play_button is not certified: 29 validation truth boxes, fewer than 30" in out
    assert "WARNING: self" not in out


def test_pick_thresholds_marks_the_fallback_uncertified_however_much_truth_there_is(capsys):
    # play_button has no detection at all; self has detections that never reach its target
    picked, validation = export.pick_thresholds(
        ["play_button", "self"],
        {0: [], 1: [0.5, 0.6]},
        {0: [], 1: [False, False]},
        {0: 500, 1: 500},
        thresholds.MIN_PRECISION_OTHER,
    )
    assert picked == {"play_button": thresholds.CEILING, "self": thresholds.CEILING}
    assert validation["play_button"]["certified"] is False
    assert validation["self"]["certified"] is False
    out = capsys.readouterr().out
    assert "WARNING: play_button is not certified: 0.95 is the fallback" in out
    assert "WARNING: self is not certified: 0.95 is the fallback" in out


def test_pick_thresholds_names_both_reasons_when_a_class_has_both(capsys):
    export.pick_thresholds(["close_x"], {0: []}, {0: []}, {0: 0}, thresholds.MIN_PRECISION_OTHER)
    out = capsys.readouterr().out
    assert (
        "WARNING: close_x is not certified: 0 validation truth boxes, fewer than 30; "
        "0.95 is the fallback" in out
    )


def test_pick_thresholds_warns_about_nothing_when_every_class_is_certified(capsys):
    _picked, validation = export.pick_thresholds(
        ["self", "play_button"],
        {0: _NINETY_SCORES, 1: _NINETY_SCORES},
        {0: _NINETY_HITS, 1: _NINETY_HITS},
        {0: 30, 1: 31},
        thresholds.MIN_PRECISION_OTHER,
    )
    assert all(row["certified"] for row in validation.values())
    assert "WARNING" not in capsys.readouterr().out


def test_decode_reads_the_expected_box_and_ignores_the_last_column():
    dets = np.zeros((1, 2, 4), dtype=np.float32)
    dets[0, 0] = (0.5, 0.5, 0.2, 0.4)
    dets[0, 1] = (0.1, 0.1, 0.1, 0.1)
    logits = np.full((1, 2, 3), -6.0, dtype=np.float32)
    logits[0, 0, 0] = 2.0  # class index 0, the only score above the floor
    logits[0, 0, 2] = 8.0  # the background column, which is never a class

    found = thresholds.decode(dets, logits, 1600, 900)

    assert len(found) == 1
    box = found[0]
    assert box["class_index"] == 0
    assert box["score"] == pytest.approx(1 / (1 + np.exp(-2.0)), abs=1e-6)
    assert (box["x"], box["y"], box["w"], box["h"]) == pytest.approx((640.0, 270.0, 320.0, 360.0))


def test_decode_keeps_every_class_above_the_floor_sorted_by_score():
    dets = np.zeros((1, 1, 4), dtype=np.float32)
    dets[0, 0] = (0.5, 0.5, 0.5, 0.5)
    logits = np.array([[[1.0, 2.0, -9.0, 9.0]]], dtype=np.float32)

    found = thresholds.decode(dets, logits, 384, 384)

    assert [box["class_index"] for box in found] == [1, 0]


def test_preprocess_shape_and_values():
    frame = np.full((900, 1600, 3), 128, dtype=np.uint8)

    batch = thresholds.preprocess(frame)

    assert batch.shape == (1, 3, 384, 384)
    assert batch.dtype == np.float32
    for channel in range(3):
        want = (128 / 255 - thresholds.MEAN[channel]) / thresholds.STD[channel]
        assert batch[0, channel] == pytest.approx(want, abs=1e-5)


def test_preprocess_swaps_the_colour_channels():
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # blue in BGR becomes the last channel in RGB

    batch = thresholds.preprocess(frame)

    assert batch[0, 2].mean() > batch[0, 0].mean()


@pytest.mark.parametrize("name", ["train.py", "export.py"])
def test_script_starts_with_inline_metadata(name):
    text = (SCRIPTS / name).read_text(encoding="utf-8")
    assert text.startswith("# /// script")
    assert "rfdetr[train,onnx]==1.10.1" in text


def test_thresholds_imports_only_numpy_and_the_standard_library():
    """The isolated scripts import this module, and that environment has no brawlfarm in it.

    Only top-level imports count: preprocess imports cv2 inside its body on purpose, so the
    module stays importable wherever numpy alone is installed.
    """
    tree = ast.parse((SCRIPTS / "thresholds.py").read_text(encoding="utf-8"))
    allowed = set(sys.stdlib_module_names) | {"numpy"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for module in names:
            assert module.split(".")[0] in allowed, module
