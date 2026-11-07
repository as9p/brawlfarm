"""Scoring a model against the template verdicts: the never-tap bar, recall and an honest verdict.

No model and no template picture is real here: the session is a fake returning hand-built
arrays and the matcher is a fake, so what is under test is the decoding, the per-class
threshold, the verdict per frame, the PASS, FAIL or INCONCLUSIVE over all of them and which
frames get scored at all, not what a real net happens to believe.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from brawlfarm.core.vision import Match
from tools.play import classes, coco, dataset, score

PLAY = classes.CLASSES.index("play_button")
CLOSE = classes.CLASSES.index("close_x")
SKULL = classes.CLASSES.index("skull_star")
ENEMY = classes.CLASSES.index("enemy")
HASH = "0123456789abcdef"
# the tap anchors a template labels: the only ones whose misses the report can count
LABELLED = sorted(classes.TEMPLATE_CLASS.values())


class _Session:
    """A stand-in for an onnxruntime session: fixed outputs, in the order it was given them."""

    def __init__(self, outputs: list[tuple[str, np.ndarray]]):
        self.outputs = outputs
        self.feeds: list[dict] = []

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name, _ in self.outputs]

    def run(self, wanted, feeds):
        assert wanted is None
        self.feeds.append(feeds)
        return [array for _, array in self.outputs]


def _arrays(rows: list[tuple[int, float, tuple[float, float, float, float]]]):
    """Hand-built RF-DETR outputs: one query per row, as (class index, score, cxcywh in 0..1)."""
    dets = np.zeros((1, len(rows), 4), dtype=np.float32)
    logits = np.full((1, len(rows), len(classes.CLASSES) + 1), -20.0, dtype=np.float32)
    for query, (index, wanted, box) in enumerate(rows):
        dets[0, query] = box
        logits[0, query, index] = math.log(wanted / (1 - wanted))
    return dets, logits


def _meta(thresholds: dict[str, float], *, smoke: bool = False) -> dict:
    return {
        "input": {"name": "input", "size": 384},
        "outputs": {"boxes": "dets", "logits": "labels"},
        "classes": list(classes.CLASSES),
        "thresholds": thresholds,
        "training_set_hash": HASH,
        "smoke": smoke,
    }


def _session(rows, *, reversed_outputs: bool = False) -> _Session:
    dets, logits = _arrays(rows)
    outputs = [("dets", dets), ("labels", logits)]
    return _Session(list(reversed(outputs)) if reversed_outputs else outputs)


def _frame(h: int = dataset.FRAME_H, w: int = dataset.FRAME_W) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _finder(matches: dict[str, Match]):
    def find(frame, name, threshold=None):
        match = matches.get(name)
        return match, 0.9 if match is not None else 0.1

    return find


def _box(cls: str, x: int, y: int, w: int, h: int, sc: float = 0.9) -> dict:
    return {"class": cls, "x": x, "y": y, "w": w, "h": h, "score": sc}


def test_detect_keeps_only_a_score_at_or_above_the_per_class_threshold():
    session = _session([(PLAY, 0.9, (0.3, 0.5, 0.1, 0.1)), (CLOSE, 0.4, (0.9, 0.1, 0.05, 0.05))])

    found = score.detect(session, _meta({"play_button": 0.5, "close_x": 0.6}), _frame())

    assert [box["class"] for box in found] == ["play_button"]
    assert found[0]["x"] == pytest.approx(400, abs=0.01)
    assert found[0]["y"] == pytest.approx(405, abs=0.01)
    assert found[0]["w"] == pytest.approx(160, abs=0.01)
    assert found[0]["h"] == pytest.approx(90, abs=0.01)
    assert found[0]["score"] > 0.89
    assert session.feeds[0]["input"].shape == (1, 3, 384, 384)


def test_detect_takes_the_outputs_by_name_not_by_position():
    rows = [(PLAY, 0.8, (0.3, 0.5, 0.1, 0.1))]
    straight = score.detect(_session(rows), _meta({"play_button": 0.5}), _frame())
    swapped = score.detect(
        _session(rows, reversed_outputs=True), _meta({"play_button": 0.5}), _frame()
    )

    assert straight == swapped != []


def test_detect_disbelieves_a_class_with_no_threshold():
    session = _session([(PLAY, 0.9, (0.3, 0.5, 0.1, 0.1))])

    assert score.detect(session, _meta({}), _frame()) == []


def test_template_truth_has_a_box_for_a_match_and_none_for_the_rest():
    find = _finder({"play": Match(name="play", confidence=0.9, x=480, y=270, w=160, h=90)})

    truth = score.template_truth(_frame(), find=find)

    assert set(truth) == set(classes.TEMPLATE_CLASS.values())
    assert truth["play_button"] == {
        "class": "play_button",
        "x": 400,
        "y": 225,
        "w": 160,
        "h": 90,
        "score": 0.9,
    }
    assert truth["close_x"] is None


def test_compare_calls_an_overlapping_detection_a_true_positive():
    truth = {"play_button": _box("play_button", 400, 225, 160, 90), "close_x": None}
    found = [{"class": "play_button", "x": 405, "y": 230, "w": 160, "h": 90, "score": 0.9}]

    assert score.compare(found, truth)["play_button"] == "tp"


def test_compare_calls_a_template_box_with_no_detection_a_false_negative():
    truth = {"play_button": _box("play_button", 400, 225, 160, 90)}

    assert score.compare([], truth)["play_button"] == "fn"
    far = [{"class": "play_button", "x": 1000, "y": 700, "w": 160, "h": 90, "score": 0.9}]
    assert score.compare(far, truth)["play_button"] == "fn"


def test_compare_calls_a_detection_the_templates_do_not_see_a_false_positive():
    truth = {"play_button": None}
    found = [{"class": "play_button", "x": 400, "y": 225, "w": 160, "h": 90, "score": 0.9}]

    assert score.compare(found, truth)["play_button"] == "fp"


def test_compare_calls_silence_on_both_sides_a_true_negative():
    assert score.compare([], {"play_button": None})["play_button"] == "tn"


def report_of(rows: list[dict]) -> dict:
    return score.report(rows)["classes"]


def _measured(tp: int, fn: int = 0) -> list[dict]:
    """Rows where every tap anchor with a template is found `tp` times and missed `fn` times."""
    found = [dict.fromkeys(LABELLED, "tp") for _ in range(tp)]
    return found + [dict.fromkeys(LABELLED, "fn") for _ in range(fn)]


def test_the_exit_codes_are_pass_0_fail_1_inconclusive_2():
    assert score.EXIT_CODES == {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}


def test_the_anchors_with_no_label_source_are_the_tap_anchors_with_no_template():
    assert score.UNVERIFIABLE == {"skull_star", "team_up_panel", "event_tab"}
    assert score.LABELLED == set(LABELLED)


def test_report_sums_the_verdicts_per_class():
    rows = [
        {"play_button": "tp", "close_x": "tn"},
        {"play_button": "fn", "close_x": "tn"},
        {"play_button": "tp", "close_x": "tn"},
    ]

    totals = report_of(rows)["play_button"]

    assert totals["tp"] == 2 and totals["fn"] == 1 and totals["fp"] == 0 and totals["tn"] == 0
    assert totals["precision"] == 1.0
    assert totals["recall"] == round(2 / 3, 4)


def test_report_passes_when_every_labelled_anchor_is_measured_and_found():
    summary = score.report(_measured(30))

    assert summary["verdict"] == "PASS" and summary["pass"] is True
    assert summary["fail_reasons"] == [] and summary["inconclusive_reasons"] == []
    assert summary["tap_anchors"]["close_x"] == {
        "truth": 30,
        "tp": 30,
        "fn": 0,
        "fp": 0,
        "recall": 1.0,
    }


def test_report_lists_the_anchors_with_no_label_source_and_lets_them_decide_nothing():
    summary = score.report(_measured(30))

    assert summary["no_label_source"] == ["event_tab", "skull_star", "team_up_panel"]
    assert set(summary["tap_anchors"]) == set(LABELLED)
    assert summary["verdict"] == "PASS"


def test_report_on_misses_only_is_inconclusive_below_the_truth_minimum_and_a_fail_at_it():
    # The 2026-11-04 retrain: every control missed, and too few of them to measure.
    short = score.report(_measured(0, fn=29))
    enough = score.report(_measured(0, fn=30))

    assert short["verdict"] == "INCONCLUSIVE" and short["pass"] is False
    assert short["fail_reasons"] == []
    assert any("play_button" in reason for reason in short["inconclusive_reasons"])
    assert enough["verdict"] == "FAIL" and enough["pass"] is False
    assert enough["tap_anchors"]["play_button"] == {
        "truth": 30,
        "tp": 0,
        "fn": 30,
        "fp": 0,
        "recall": 0.0,
    }
    assert any("play_button" in reason for reason in enough["fail_reasons"])


def test_report_holds_recall_and_truth_to_the_bars_it_is_given():
    rows = _measured(15, fn=15)

    assert score.report(rows)["verdict"] == "PASS"
    assert score.report(rows, min_recall=0.6)["verdict"] == "FAIL"
    assert score.report(rows, min_truth=31)["verdict"] == "INCONCLUSIVE"
    assert score.report(_measured(0, fn=29), min_truth=29)["verdict"] == "FAIL"


def test_report_is_inconclusive_on_true_negatives_alone():
    summary = score.report([dict.fromkeys(LABELLED, "tn") for _ in range(500)])

    assert summary["verdict"] == "INCONCLUSIVE" and summary["pass"] is False
    assert summary["tap_anchor_false_positives"] == 0
    assert summary["tap_anchors"]["play_button"]["truth"] == 0


def test_report_is_inconclusive_with_nothing_to_report():
    summary = score.report([])

    assert summary["verdict"] == "INCONCLUSIVE" and summary["pass"] is False
    assert any("nothing is proven" in reason for reason in summary["inconclusive_reasons"])


def test_report_fails_on_a_single_tap_anchor_false_positive():
    summary = score.report([{"play_button": "tp"}, {"play_button": "fp"}])

    assert summary["tap_anchor_false_positives"] == 1
    # One truth positive is far too few to measure, and the FAIL still wins.
    assert summary["verdict"] == "FAIL" and summary["pass"] is False
    assert summary["fail_reasons"] and summary["inconclusive_reasons"]


def test_report_lets_a_caveat_stop_a_pass_but_never_a_fail():
    held = score.report(_measured(30), caveats=["a smoke model"])

    assert held["verdict"] == "INCONCLUSIVE"
    assert held["inconclusive_reasons"] == ["a smoke model"]
    assert score.report(_measured(0, fn=30), caveats=["a smoke model"])["verdict"] == "FAIL"


def test_compare_keeps_an_extra_detection_beside_a_true_positive():
    truth = {"play_button": _box("play_button", 400, 225, 160, 90)}
    found = [
        {"class": "play_button", "x": 405, "y": 230, "w": 160, "h": 90, "score": 0.9},
        {"class": "play_button", "x": 1000, "y": 700, "w": 160, "h": 90, "score": 0.8},
    ]

    assert score.compare(found, truth)["play_button"] == "tp+extra"


def test_compare_calls_two_detections_on_one_template_box_a_plain_true_positive():
    truth = {"play_button": _box("play_button", 400, 225, 160, 90)}
    found = [
        {"class": "play_button", "x": 400, "y": 225, "w": 160, "h": 90, "score": 0.9},
        {"class": "play_button", "x": 410, "y": 235, "w": 160, "h": 90, "score": 0.8},
    ]

    assert score.compare(found, truth)["play_button"] == "tp"


def test_report_counts_an_extra_as_a_true_and_a_false_positive():
    totals = report_of([{"play_button": "tp+extra"}])["play_button"]

    assert totals["tp"] == 1 and totals["fp"] == 1
    assert totals["precision"] == 0.5


def test_report_fails_on_an_extra_tap_anchor_detection():
    summary = score.report([{"play_button": "tp"}, {"play_button": "tp+extra"}])

    assert summary["tap_anchor_false_positives"] == 1
    assert summary["verdict"] == "FAIL" and summary["pass"] is False


def test_report_passes_on_an_extra_for_a_class_that_is_not_a_tap_anchor():
    summary = score.report([*_measured(30), {"power_cube": "tp+extra"}])

    assert summary["classes"]["power_cube"]["fp"] == 1
    assert summary["tap_anchor_false_positives"] == 0
    assert summary["verdict"] == "PASS"


def _write_frames(folder: Path) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "good.jpg"), _frame())
    cv2.imwrite(str(folder / "small.jpg"), _frame(450, 800))
    (folder / "broken.png").write_bytes(b"not a picture")
    return ["broken.png", "good.jpg", "small.jpg"]


def _write_coco(root: Path, *, training_set_hash: str = HASH) -> None:
    """A build laid out the way coco.py writes one, plus a training frame it must not score."""
    test = root / "coco" / "test"
    images = [
        {"id": number, "file_name": name, "width": dataset.FRAME_W, "height": dataset.FRAME_H}
        for number, name in enumerate(_write_frames(test), start=1)
    ]
    (test / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": [], "categories": []}), encoding="utf-8"
    )
    (root / "coco" / "split.json").write_text(
        json.dumps({"smoke": False, "training_set_hash": training_set_hash}), encoding="utf-8"
    )
    trained = root / "frames" / "session"
    trained.mkdir(parents=True)
    cv2.imwrite(str(trained / "trained.jpg"), _frame())


def _write_model(models: Path) -> None:
    """The files main checks for before it loads anything; the fake load_model ignores them."""
    models.mkdir(parents=True, exist_ok=True)
    (models / "play.onnx").write_bytes(b"not a real model")
    (models / "play.json").write_text("{}", encoding="utf-8")


def _fake_load_model(session, meta):
    def load_model(models, *, cuda=False):
        return session, meta

    return load_model


def _all_found(*, smoke: bool = False):
    """A model and templates that agree on every tap anchor with a template, side by side."""
    queries, matches = [], {}
    for number, (template, cls) in enumerate(classes.TEMPLATE_CLASS.items()):
        middle = 0.1 + 0.15 * number
        queries.append((classes.CLASSES.index(cls), 0.9, (middle, 0.5, 0.1, 0.1)))
        matches[template] = Match(
            name=template, confidence=0.9, x=round(middle * dataset.FRAME_W), y=450, w=160, h=90
        )
    meta = _meta(dict.fromkeys(classes.TEMPLATE_CLASS.values(), 0.5), smoke=smoke)
    return _session(queries), meta, _finder(matches)


def _main(tmp_path, monkeypatch, session, meta, find, *flags: str) -> tuple[int, dict]:
    """score.main on the dataset root under tmp_path: the exit code and what score.json says."""
    root = tmp_path / "root"
    _write_model(root / "models")
    monkeypatch.setattr(score, "load_model", _fake_load_model(session, meta))
    code = score.main(["--root", str(root), *flags], find=find)
    return code, json.loads((root / "models" / "score.json").read_text(encoding="utf-8"))


def test_main_scores_only_the_held_out_test_split(tmp_path, monkeypatch, capsys):
    _write_coco(tmp_path / "root")
    session = _session([(PLAY, 0.9, (0.3, 0.5, 0.1, 0.1))])
    find = _finder({"play": Match(name="play", confidence=0.9, x=480, y=450, w=160, h=90)})

    code, written = _main(tmp_path, monkeypatch, session, _meta({"play_button": 0.5}), find)

    assert written["frames"] == {"scored": 1, "skipped": 2}
    assert written["held_out"] is True
    assert written["tap_anchors"]["play_button"] == {
        "truth": 1,
        "tp": 1,
        "fn": 0,
        "fp": 0,
        "recall": 1.0,
    }
    assert written["training_set_hash"] == HASH
    assert written["smoke"] is False
    assert written["generated"].endswith("+00:00")
    # One frame of one control shows the model can find it, not that it does.
    assert written["verdict"] == "INCONCLUSIVE" and code == 2
    assert str(tmp_path / "root" / "coco") not in capsys.readouterr().out


def test_main_passes_when_every_labelled_anchor_is_found_often_enough(
    tmp_path, monkeypatch, capsys
):
    _write_coco(tmp_path / "root")
    session, meta, find = _all_found()

    code, written = _main(tmp_path, monkeypatch, session, meta, find, "--min-truth", "1")

    assert code == 0
    assert written["verdict"] == "PASS" and written["pass"] is True
    assert written["fail_reasons"] == [] and written["inconclusive_reasons"] == []
    assert written["no_label_source"] == ["event_tab", "skull_star", "team_up_panel"]
    assert written["min_truth"] == 1 and written["min_recall"] == 0.5
    assert "PASS" in capsys.readouterr().out


def test_main_on_misses_only_is_fail_or_inconclusive_by_truth_count(tmp_path, monkeypatch):
    _write_coco(tmp_path / "root")
    _found, meta, find = _all_found()
    blind = _session([(PLAY, 0.1, (0.3, 0.5, 0.1, 0.1))])

    unsure, written = _main(tmp_path, monkeypatch, blind, meta, find)
    failed, measured = _main(tmp_path, monkeypatch, blind, meta, find, "--min-truth", "1")

    assert unsure == 2 and written["verdict"] == "INCONCLUSIVE"
    assert failed == 1 and measured["verdict"] == "FAIL"
    assert measured["tap_anchors"]["close_x"]["fn"] == 1


def test_main_fails_on_a_tap_anchor_the_templates_do_not_see(tmp_path, monkeypatch, capsys):
    _write_coco(tmp_path / "root")
    session = _session([(PLAY, 0.9, (0.3, 0.5, 0.1, 0.1))])

    code, written = _main(tmp_path, monkeypatch, session, _meta({"play_button": 0.5}), _finder({}))

    assert code == 1
    assert written["verdict"] == "FAIL" and written["pass"] is False
    assert written["tap_anchor_false_positives"] == 1
    assert written["tap_anchor_failures"] == [{"file": "good.jpg", "class": "play_button"}]
    assert "good.jpg" in capsys.readouterr().out


def test_main_reports_a_tap_anchor_with_no_template_as_unverified(tmp_path, monkeypatch):
    _write_coco(tmp_path / "root")
    session = _session([(SKULL, 0.9, (0.3, 0.5, 0.1, 0.1))])

    code, written = _main(tmp_path, monkeypatch, session, _meta({"skull_star": 0.5}), _finder({}))

    assert written["unverified_tap_anchor_detections"] == [
        {"file": "good.jpg", "class": "skull_star"}
    ]
    assert written["tap_anchor_false_positives"] == 0
    assert "skull_star" in written["no_label_source"]
    # Nothing with a template was on the frame, so nothing was measured either.
    assert written["verdict"] == "INCONCLUSIVE" and code == 2


def test_main_is_inconclusive_when_no_frame_could_be_scored(tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    _write_coco(root)
    # Listed in the build but gone from disk: skipped, not a crash.
    (root / "coco" / "test" / "good.jpg").unlink()
    session = _session([(PLAY, 0.9, (0.3, 0.5, 0.1, 0.1))])

    code, written = _main(tmp_path, monkeypatch, session, _meta({"play_button": 0.5}), _finder({}))

    assert code == 2
    assert written["verdict"] == "INCONCLUSIVE" and written["pass"] is False
    assert written["frames"] == {"scored": 0, "skipped": 3}
    assert "nothing is proven" in capsys.readouterr().out


def test_main_fails_on_an_extra_tap_anchor_detection(tmp_path, monkeypatch):
    _write_coco(tmp_path / "root")
    session = _session([(PLAY, 0.9, (0.3, 0.5, 0.1, 0.1)), (PLAY, 0.8, (0.8, 0.8, 0.1, 0.1))])
    find = _finder({"play": Match(name="play", confidence=0.9, x=480, y=450, w=160, h=90)})

    code, written = _main(tmp_path, monkeypatch, session, _meta({"play_button": 0.5}), find)

    assert code == 1
    assert written["verdict"] == "FAIL"
    assert written["tap_anchor_failures"] == [{"file": "good.jpg", "class": "play_button"}]
    assert written["classes"]["play_button"] == {
        "tp": 1,
        "fp": 1,
        "fn": 0,
        "tn": 0,
        "precision": 0.5,
        "recall": 1.0,
    }


def test_main_is_inconclusive_on_a_smoke_model(tmp_path, monkeypatch):
    _write_coco(tmp_path / "root")
    session, meta, find = _all_found(smoke=True)

    code, written = _main(tmp_path, monkeypatch, session, meta, find, "--min-truth", "1")

    assert code == 2
    assert written["verdict"] == "INCONCLUSIVE" and written["smoke"] is True
    assert any("smoke" in reason for reason in written["inconclusive_reasons"])


def test_main_is_inconclusive_without_a_coco_build(tmp_path, monkeypatch, capsys):
    session, meta, find = _all_found()

    code, written = _main(tmp_path, monkeypatch, session, meta, find, "--min-truth", "1")

    assert code == 2
    assert written["verdict"] == "INCONCLUSIVE"
    assert written["frames"] == {"scored": 0, "skipped": 0}
    assert any("coco.py" in reason for reason in written["inconclusive_reasons"])
    assert "coco.py" in capsys.readouterr().out


def test_main_is_inconclusive_on_a_build_the_model_did_not_train_on(tmp_path, monkeypatch):
    _write_coco(tmp_path / "root", training_set_hash="fedcba9876543210")
    session, meta, find = _all_found()

    code, written = _main(tmp_path, monkeypatch, session, meta, find, "--min-truth", "1")

    assert code == 2
    assert written["frames"]["scored"] == 1
    assert any("hash" in reason for reason in written["inconclusive_reasons"])


def test_main_folder_scan_is_never_a_pass(tmp_path, monkeypatch):
    root = tmp_path / "root"
    _write_frames(root / "frames" / "session")
    _write_frames(tmp_path / "home" / "calibration" / "recordings")
    monkeypatch.setattr(score.settings, "default_home", lambda: tmp_path / "home")
    session, meta, find = _all_found()

    code, written = _main(
        tmp_path, monkeypatch, session, meta, find, "--scan-folders", "--min-truth", "1"
    )

    assert code == 2
    assert written["verdict"] == "INCONCLUSIVE" and written["held_out"] is False
    assert written["frames"] == {"scored": 2, "skipped": 4}
    assert any("folder scan" in reason for reason in written["inconclusive_reasons"])


def test_main_frames_flag_is_a_folder_scan_and_never_a_pass(tmp_path, monkeypatch, capsys):
    frames = tmp_path / "shots"
    _write_frames(frames)
    _write_coco(tmp_path / "root")
    session, meta, find = _all_found()

    code, written = _main(
        tmp_path, monkeypatch, session, meta, find, "--frames", str(frames), "--min-truth", "1"
    )

    assert code == 2
    assert written["frames"] == {"scored": 1, "skipped": 2}
    assert any("folder scan" in reason for reason in written["inconclusive_reasons"])
    assert str(frames) not in capsys.readouterr().out


@pytest.mark.parametrize(
    "flags", [["--min-truth", "0"], ["--min-recall", "0"], ["--min-recall", "1.5"]]
)
def test_main_refuses_a_bar_that_measures_nothing(flags):
    with pytest.raises(SystemExit) as stopped:
        score.main(flags)

    assert stopped.value.code == 2


def _fake_ort(monkeypatch, active: list[str]) -> list[list[str]]:
    """Swap onnxruntime's session for one that ends up on `active`, whatever it was asked for."""
    asked: list[list[str]] = []

    class Session:
        def __init__(self, path, providers):
            asked.append(providers)

        def get_providers(self):
            return active

    monkeypatch.setattr(score.ort, "InferenceSession", Session)
    monkeypatch.setattr(score.ort, "preload_dlls", lambda: None, raising=False)
    return asked


def test_load_model_stops_when_cuda_was_asked_for_and_not_obtained(tmp_path, monkeypatch):
    _write_model(tmp_path)
    _fake_ort(monkeypatch, ["CPUExecutionProvider"])

    with pytest.raises(SystemExit) as stopped:
        score.load_model(tmp_path, cuda=True)

    assert "play-gpu" in str(stopped.value) and "docs/install.md" in str(stopped.value)


def test_load_model_keeps_a_cuda_session_and_a_cpu_one_nobody_asked_cuda_for(tmp_path, monkeypatch):
    _write_model(tmp_path)
    on_gpu = _fake_ort(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    score.load_model(tmp_path, cuda=True)
    on_cpu = _fake_ort(monkeypatch, ["CPUExecutionProvider"])
    score.load_model(tmp_path)

    assert on_gpu == [["CUDAExecutionProvider", "CPUExecutionProvider"]]
    assert on_cpu == [["CPUExecutionProvider"]]


def test_main_with_cuda_stops_before_scoring_anything_on_the_cpu(tmp_path, monkeypatch):
    root = tmp_path / "root"
    _write_coco(root)
    _write_model(root / "models")
    _fake_ort(monkeypatch, ["CPUExecutionProvider"])

    with pytest.raises(SystemExit) as stopped:
        score.main(["--root", str(root), "--cuda"], find=_finder({}))

    assert "play-gpu" in str(stopped.value)
    assert not (root / "models" / "score.json").exists()


LIVE_FRAME = "session-000001.jpg"
VIDEO_FRAME = "yt-AAAAAAAAAAA-000001.jpg"
# where a model that says (ENEMY, 0.9, (0.3, 0.5, 0.1, 0.1)) puts its box, and somewhere else
ON_DETECTION = [400, 405, 160, 90]
OFF_DETECTION = [1200, 100, 160, 90]


def _label(cls: str, bbox: list[int]) -> dict:
    x, y, w, h = bbox
    return {"class_index": classes.category_id(cls) - 1, "x": x, "y": y, "w": w, "h": h}


def _enemy(sc: float = 0.9, *, bbox: list[int] = ON_DETECTION) -> dict:
    return _box("enemy", *bbox, sc) | {"class_index": ENEMY}


def _write_labelled_coco(root: Path, labels: dict[str, list[tuple[str, list[int]]]]) -> None:
    """A build whose test split holds these frames, each with these (class, bbox) labels."""
    test = root / "coco" / "test"
    test.mkdir(parents=True)
    images, annotations = [], []
    for number, (name, boxes) in enumerate(labels.items(), start=1):
        cv2.imwrite(str(test / name), _frame())
        images.append(
            {"id": number, "file_name": name, "width": dataset.FRAME_W, "height": dataset.FRAME_H}
        )
        for cls, bbox in boxes:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": number,
                    "category_id": classes.category_id(cls),
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
    (test / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": []}),
        encoding="utf-8",
    )
    (root / "coco" / "split.json").write_text(
        json.dumps({"smoke": False, "training_set_hash": HASH}), encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("name", "group"),
    [
        (VIDEO_FRAME, score.YOUTUBE),
        (f"{coco.YOUTUBE_PREFIX}-AAAAAAAAAA-000002.jpg", score.YOUTUBE),
        (LIVE_FRAME, score.LIVE),
        ("20260916-101500-000001.jpg", score.LIVE),
        ("myt-000001.jpg", score.LIVE),
        ("youtube-000001.jpg", score.LIVE),
    ],
)
def test_source_group_goes_by_the_youtube_prefix_of_the_file_name(name, group):
    assert score.source_group(name) == group


def test_the_in_match_classes_are_the_ones_no_tap_anchor_covers():
    assert score.IN_MATCH == ("self", "enemy", "teammate", "power_cube", "box", "bush")


def test_tally_counts_a_hit_a_miss_and_an_invention():
    labels = [_label("enemy", ON_DETECTION), _label("enemy", [100, 100, 160, 90])]

    counts = score.tally([_enemy(), _enemy(0.8, bbox=OFF_DETECTION)], labels)

    assert counts["enemy"] == {"tp": 1, "fp": 1, "fn": 1}
    assert counts["self"] == {"tp": 0, "fp": 0, "fn": 0}


def test_tally_counts_a_second_box_on_one_object_as_a_false_positive():
    counts = score.tally([_enemy(), _enemy(0.8)], [_label("enemy", ON_DETECTION)])

    assert counts["enemy"] == {"tp": 1, "fp": 1, "fn": 0}


def test_tally_never_matches_a_box_to_a_label_of_another_class():
    counts = score.tally([_enemy()], [_label("teammate", ON_DETECTION)])

    assert counts["enemy"] == {"tp": 0, "fp": 1, "fn": 0}
    assert counts["teammate"] == {"tp": 0, "fp": 0, "fn": 1}


def test_tally_leaves_a_tap_anchor_out():
    assert "play_button" not in score.tally([_box("play_button", 0, 0, 10, 10)], [])


def test_by_source_keeps_the_groups_apart():
    hit = score.tally([_enemy()], [_label("enemy", ON_DETECTION)])
    miss = score.tally([], [_label("enemy", ON_DETECTION)])
    invented = score.tally([_enemy()], [])

    sources = score.by_source(
        {score.LIVE: [hit, miss], score.YOUTUBE: [hit, hit, hit, invented]}, min_truth=1
    )

    assert sources[score.LIVE]["frames"] == 2
    assert sources[score.LIVE]["classes"]["enemy"] == {
        "truth": 2,
        "tp": 1,
        "fp": 0,
        "fn": 1,
        "precision": 1.0,
        "recall": 0.5,
        "enough_truth": True,
    }
    assert sources[score.YOUTUBE]["frames"] == 4
    assert sources[score.YOUTUBE]["classes"]["enemy"] == {
        "truth": 3,
        "tp": 3,
        "fp": 1,
        "fn": 0,
        "precision": 0.75,
        "recall": 1.0,
        "enough_truth": True,
    }


def test_by_source_says_a_class_has_too_little_truth_below_the_minimum_and_not_at_it():
    hit = score.tally([_enemy()], [_label("enemy", ON_DETECTION)])

    below = score.by_source({score.LIVE: [hit] * 2}, min_truth=3)[score.LIVE]["classes"]
    at = score.by_source({score.LIVE: [hit] * 3}, min_truth=3)[score.LIVE]["classes"]

    assert below["enemy"]["enough_truth"] is False
    assert at["enemy"]["enough_truth"] is True


def test_by_source_reports_nothing_measured_as_zero_and_too_little_truth():
    count = score.by_source({score.LIVE: []})[score.LIVE]

    assert count["frames"] == 0
    assert set(count["classes"]) == set(score.IN_MATCH)
    assert count["classes"]["enemy"] == {
        "truth": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": 0.0,
        "recall": 0.0,
        "enough_truth": False,
    }


def test_main_scores_live_captures_apart_from_youtube(tmp_path, monkeypatch, capsys):
    _write_labelled_coco(
        tmp_path / "root",
        {
            LIVE_FRAME: [("enemy", ON_DETECTION)],
            VIDEO_FRAME: [("enemy", OFF_DETECTION), ("bush", OFF_DETECTION)],
        },
    )
    session = _session([(ENEMY, 0.9, (0.3, 0.5, 0.1, 0.1))])

    _code, written = _main(
        tmp_path, monkeypatch, session, _meta({"enemy": 0.5}), _finder({}), "--min-truth", "2"
    )

    live, youtube = written["sources"]["live"], written["sources"]["youtube"]
    assert live["frames"] == 1 and youtube["frames"] == 1
    assert live["classes"]["enemy"] == {
        "truth": 1,
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "enough_truth": False,
    }
    assert youtube["classes"]["enemy"] == {
        "truth": 1,
        "tp": 0,
        "fp": 1,
        "fn": 1,
        "precision": 0.0,
        "recall": 0.0,
        "enough_truth": False,
    }
    assert youtube["classes"]["bush"]["fn"] == 1
    assert "too little truth, under 2" in capsys.readouterr().out


def test_main_counts_only_the_frames_it_scored(tmp_path, monkeypatch):
    root = tmp_path / "root"
    _write_labelled_coco(root, {LIVE_FRAME: [("enemy", ON_DETECTION)]})
    cv2.imwrite(str(root / "coco" / "test" / LIVE_FRAME), _frame(450, 800))

    _code, written = _main(tmp_path, monkeypatch, _session([]), _meta({}), _finder({}))

    assert written["frames"] == {"scored": 0, "skipped": 1}
    assert written["sources"]["live"]["frames"] == 0
    assert written["sources"]["live"]["classes"]["enemy"]["truth"] == 0


def test_main_says_when_a_group_has_no_frames(tmp_path, monkeypatch, capsys):
    _write_labelled_coco(tmp_path / "root", {LIVE_FRAME: []})

    _main(tmp_path, monkeypatch, _session([]), _meta({}), _finder({}))

    assert "youtube  no frames scored" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("templates_see_it", "flags", "verdict"),
    [
        (True, ("--min-truth", "1"), "PASS"),
        (True, (), "INCONCLUSIVE"),
        (False, ("--min-truth", "1"), "FAIL"),
    ],
)
def test_main_keeps_its_verdict_whatever_the_in_match_numbers_say(
    tmp_path, monkeypatch, templates_see_it, flags, verdict
):
    """The in-match labels are all missed here, and the run has to end as it would without them."""
    runs = []
    for folder, labels in (("bare", []), ("labelled", [("enemy", OFF_DETECTION)] * 40)):
        _write_labelled_coco(tmp_path / folder / "root", {LIVE_FRAME: labels, VIDEO_FRAME: labels})
        session, meta, find = _all_found()
        find = find if templates_see_it else _finder({})
        runs.append(_main(tmp_path / folder, monkeypatch, session, meta, find, *flags))

    (bare_code, bare), (labelled_code, labelled) = runs
    assert labelled["verdict"] == bare["verdict"] == verdict
    assert labelled_code == bare_code == score.EXIT_CODES[verdict]
    for key in ("pass", "fail_reasons", "inconclusive_reasons", "classes", "tap_anchors"):
        assert labelled[key] == bare[key]
    assert labelled["sources"]["live"]["classes"]["enemy"]["recall"] == 0.0


def test_main_folder_scan_has_no_labels_to_report(tmp_path, monkeypatch, capsys):
    _write_frames(tmp_path / "shots")
    session, meta, find = _all_found()

    _code, written = _main(
        tmp_path, monkeypatch, session, meta, find, "--frames", str(tmp_path / "shots")
    )

    assert written["sources"] is None
    assert "a folder scan has no labels" in capsys.readouterr().out


def test_main_without_a_coco_build_has_no_labels_to_report(tmp_path, monkeypatch):
    session, meta, find = _all_found()

    code, written = _main(tmp_path, monkeypatch, session, meta, find)

    assert code == 2
    assert written["sources"] is None
