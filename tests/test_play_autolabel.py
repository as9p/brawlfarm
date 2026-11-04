"""The auto-labeller: in-match test, merge, the task file, and a run over a tiny dataset."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from tools.play import autolabel, brawlers, coco, dataset, prelabel, segment, zones

H, W = dataset.FRAME_H, dataset.FRAME_W


def _hsv(sat):
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 1] = sat
    hsv[..., 2] = 150
    return hsv


def test_task_for_takes_a_model_version():
    box = {"class": "bush", "x": 10, "y": 20, "w": 30, "h": 40, "score": 0.8}
    task = prelabel.task_for("frames/a/a-000001.jpg", [box], model_version="autolabel-v1")
    assert task["predictions"][0]["model_version"] == "autolabel-v1"
    default = prelabel.task_for("frames/a/a-000001.jpg", [box])
    assert default["predictions"][0]["model_version"] == prelabel.MODEL_VERSION


def test_in_match_rules():
    play = [{"class": "play_button", "x": 1, "y": 1, "w": 5, "h": 5, "score": 0.9}]
    assert not autolabel.in_match({"hud": True}, _hsv(200), play)
    assert autolabel.in_match({"hud": True}, _hsv(0), [])
    assert not autolabel.in_match({"hud": False}, _hsv(200), [])
    assert autolabel.in_match({}, _hsv(200), [])
    assert not autolabel.in_match({}, _hsv(0), [])


def test_merge_keeps_templates_drops_hud_boxes_and_clips():
    zone = zones.zone_mask()
    template = [{"class": "play_button", "x": 1350, "y": 700, "w": 200, "h": 120, "score": 0.9}]
    on_button = {"class": "self", "x": 1200, "y": 630, "w": 50, "h": 50, "score": 0.8}
    edge = {"class": "enemy", "x": 1560, "y": 300, "w": 80, "h": 90, "score": 0.7}
    out = autolabel.merge(template, [on_button, edge], zone)
    assert out == [
        template[0],
        {"class": "enemy", "x": 1560, "y": 300, "w": 40, "h": 90, "score": 0.7},
    ]


def test_merged_task_round_trips_through_coco():
    bush = {"class": "bush", "x": 600, "y": 400, "w": 200, "h": 120, "score": 0.95}
    boxes = autolabel.merge([], [bush], zones.zone_mask())
    task = prelabel.task_for("frames/a/a-000001.jpg", boxes, model_version=autolabel.MODEL_VERSION)
    labels = coco.from_labelstudio([task], use_predictions=True)
    assert labels == {
        "frames/a/a-000001.jpg": [{"class": "bush", "x": 600, "y": 400, "w": 200, "h": 120}]
    }


def _dataset(root, n=2):
    index = dataset.Index(root)
    rels = []
    for i in range(1, n + 1):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[...] = (40, 90, 150)  # a saturated floor in BGR: is_game_frame says yes
        rel = dataset.save_frame(root, "alpha", i, frame)
        index.add(
            file=rel,
            source="alpha",
            kind="video",
            t=float(i),
            hash_=i,
            teams_left=1.0,
            pad=(0, 0, 0, 0),
        )
        rels.append(rel)
    return rels


class _FakeOcr:
    def __init__(self, result):
        self.result = result

    def __call__(self, img):
        return self.result, None


ENEMY = brawlers.Bar(
    cls="enemy", cx=800.0, bw=100.0, bot=300, named=True, colour="red", why="bar", ammo=0.0
)


def _tasks(root):
    return json.loads((root / "labelstudio" / autolabel.OUT_NAME).read_text(encoding="utf-8"))


def test_main_without_an_index_fails(tmp_path):
    assert autolabel.main(["--root", str(tmp_path), "--no-sam"]) == 1


def test_main_writes_a_task_per_frame_with_templates(tmp_path, monkeypatch):
    rels = _dataset(tmp_path)
    play = {"class": "play_button", "x": 700, "y": 700, "w": 200, "h": 100, "score": 0.9}
    calls = iter([[play], []])
    monkeypatch.setattr(prelabel, "boxes_for", lambda frame, **kw: next(calls))
    monkeypatch.setattr(autolabel, "ocr_engine", lambda: _FakeOcr([]))
    assert autolabel.main(["--root", str(tmp_path), "--no-sam"]) == 0
    tasks = _tasks(tmp_path)
    assert [t["data"]["file"] for t in tasks] == rels
    labels = tasks[0]["predictions"][0]["result"][0]["value"]["rectanglelabels"]
    assert labels == ["play_button"]
    assert tasks[1]["predictions"] == []
    assert (tmp_path / "labelstudio" / "config.xml").is_file()


def test_main_hands_bars_to_the_sam_stage(tmp_path, monkeypatch):
    rels = _dataset(tmp_path, n=1)
    monkeypatch.setattr(prelabel, "boxes_for", lambda frame, **kw: [])
    monkeypatch.setattr(autolabel, "ocr_engine", lambda: _FakeOcr([]))
    monkeypatch.setattr(brawlers, "classify", lambda *a, **k: [ENEMY])
    seen = {}

    def fake_segment(root, prompts, out, model, resume=False):
        seen["prompts"] = [json.loads(x) for x in prompts.read_text(encoding="utf-8").splitlines()]
        seen["model"] = model
        box = {
            "class": "enemy",
            "x": 755,
            "y": 310,
            "w": 90,
            "h": 95,
            "score": 0.9,
            "fallback": False,
        }
        out.write_text(json.dumps({"file": rels[0], "boxes": [box]}) + "\n", encoding="utf-8")

    monkeypatch.setattr(autolabel, "run_segment", fake_segment)
    assert autolabel.main(["--root", str(tmp_path)]) == 0
    assert seen["prompts"] == [{"file": rels[0], "bars": [ENEMY.to_json()]}]
    assert seen["model"] == autolabel.SAM_MODEL
    labels = coco.from_labelstudio(_tasks(tmp_path), use_predictions=True)
    assert labels == {rels[0]: [{"class": "enemy", "x": 755, "y": 310, "w": 90, "h": 95}]}


def test_no_sam_uses_the_fixed_box(tmp_path, monkeypatch):
    rels = _dataset(tmp_path, n=1)
    monkeypatch.setattr(prelabel, "boxes_for", lambda frame, **kw: [])
    monkeypatch.setattr(autolabel, "ocr_engine", lambda: _FakeOcr([]))
    monkeypatch.setattr(brawlers, "classify", lambda *a, **k: [ENEMY])
    monkeypatch.setattr(autolabel, "run_segment", lambda *a: pytest.fail("SAM must not run"))
    assert autolabel.main(["--root", str(tmp_path), "--no-sam"]) == 0
    labels = coco.from_labelstudio(_tasks(tmp_path), use_predictions=True)
    assert labels == {rels[0]: [{"class": "enemy", "x": 752, "y": 315, "w": 95, "h": 100}]}


def test_sam_model_matches_the_segment_default():
    assert autolabel.SAM_MODEL == segment.DEFAULT_MODEL


def test_a_failed_sam_pass_writes_nothing(tmp_path, monkeypatch):
    _dataset(tmp_path, n=1)
    monkeypatch.setattr(prelabel, "boxes_for", lambda frame, **kw: [])
    monkeypatch.setattr(autolabel, "ocr_engine", lambda: _FakeOcr([]))
    monkeypatch.setattr(brawlers, "classify", lambda *a, **k: [ENEMY])

    def failing_segment(root, prompts, out, model, resume=False):
        raise subprocess.CalledProcessError(1, ["uv", "run", "segment.py"])

    monkeypatch.setattr(autolabel, "run_segment", failing_segment)
    assert autolabel.main(["--root", str(tmp_path)]) == 1
    assert not (tmp_path / "labelstudio" / autolabel.OUT_NAME).exists()


def test_sam_is_skipped_when_no_frame_has_bars(tmp_path, monkeypatch):
    rels = _dataset(tmp_path, n=1)
    monkeypatch.setattr(prelabel, "boxes_for", lambda frame, **kw: [])
    monkeypatch.setattr(autolabel, "ocr_engine", lambda: _FakeOcr([]))
    monkeypatch.setattr(brawlers, "classify", lambda *a, **k: [])
    monkeypatch.setattr(autolabel, "run_segment", lambda *a: pytest.fail("SAM must not run"))
    assert autolabel.main(["--root", str(tmp_path)]) == 0
    assert [t["data"]["file"] for t in _tasks(tmp_path)] == rels


def test_a_frame_that_fails_is_reported_and_skipped(tmp_path, monkeypatch, capsys):
    rels = _dataset(tmp_path)
    monkeypatch.setattr(prelabel, "boxes_for", lambda frame, **kw: [])
    monkeypatch.setattr(autolabel, "ocr_engine", lambda: _FakeOcr([]))
    outcomes = iter([RuntimeError("classify broke"), []])

    def flaky_classify(*a, **k):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(brawlers, "classify", flaky_classify)
    assert autolabel.main(["--root", str(tmp_path), "--no-sam"]) == 0
    assert [t["data"]["file"] for t in _tasks(tmp_path)] == rels[1:]
    out = capsys.readouterr().out
    assert f"{rels[0]}: RuntimeError: classify broke" in out
    assert "1 failed" in out


def _pass1(root):
    path = root / "autolabel" / autolabel.PASS1_NAME
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _count_labels(monkeypatch):
    """Wrap label_frame so a test can see which frames went through the first pass."""
    seen = []
    real = autolabel.label_frame

    def counting(bgr, row, zone, ocr):
        seen.append(row["file"])
        return real(bgr, row, zone, ocr)

    monkeypatch.setattr(autolabel, "label_frame", counting)
    return seen


def _bars_everywhere(monkeypatch):
    monkeypatch.setattr(prelabel, "boxes_for", lambda frame, **kw: [])
    monkeypatch.setattr(autolabel, "ocr_engine", lambda: _FakeOcr([]))
    monkeypatch.setattr(brawlers, "classify", lambda *a, **k: [ENEMY])


def test_first_pass_writes_a_checkpoint_line_per_labelled_frame(tmp_path, monkeypatch):
    rels = _dataset(tmp_path, n=3)
    monkeypatch.setattr(prelabel, "boxes_for", lambda frame, **kw: [])
    monkeypatch.setattr(autolabel, "ocr_engine", lambda: _FakeOcr([]))
    outcomes = iter([[ENEMY], RuntimeError("classify broke"), []])

    def flaky_classify(*a, **k):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(brawlers, "classify", flaky_classify)
    assert autolabel.main(["--root", str(tmp_path), "--no-sam"]) == 0
    lines = _pass1(tmp_path)
    assert [line["file"] for line in lines] == [rels[0], rels[2]]
    assert lines[0] == {
        "version": autolabel.MODEL_VERSION,
        "file": rels[0],
        "template": [],
        "terrain": lines[0]["terrain"],
        "bars": [ENEMY.to_json()],
        "in_match": True,
    }
    assert lines[1]["bars"] == []


def _fresh_then(tmp_path, monkeypatch, n=3):
    """A fresh run's tasks, and a label_frame counter armed for the next run."""
    rels = _dataset(tmp_path, n=n)
    _bars_everywhere(monkeypatch)
    assert autolabel.main(["--root", str(tmp_path), "--no-sam"]) == 0
    return rels, _tasks(tmp_path), _count_labels(monkeypatch)


def test_resume_labels_only_the_frames_missing_from_the_checkpoint(tmp_path, monkeypatch, capsys):
    rels, fresh, seen = _fresh_then(tmp_path, monkeypatch)
    path = tmp_path / "autolabel" / autolabel.PASS1_NAME
    first = path.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(first + "\n", encoding="utf-8")
    capsys.readouterr()
    assert autolabel.main(["--root", str(tmp_path), "--no-sam", "--resume"]) == 0
    assert seen == rels[1:]
    assert _tasks(tmp_path) == fresh
    assert [line["file"] for line in _pass1(tmp_path)] == rels
    assert "1 frames from the checkpoint" in capsys.readouterr().out


def test_resume_ignores_a_cut_off_last_line(tmp_path, monkeypatch):
    rels, fresh, seen = _fresh_then(tmp_path, monkeypatch)
    path = tmp_path / "autolabel" / autolabel.PASS1_NAME
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[0] + "\n" + lines[1][: len(lines[1]) // 2], encoding="utf-8")
    assert autolabel.main(["--root", str(tmp_path), "--no-sam", "--resume"]) == 0
    assert seen == rels[1:]
    assert _tasks(tmp_path) == fresh
    assert [line["file"] for line in _pass1(tmp_path)] == rels


def test_resume_redoes_lines_from_another_model_version(tmp_path, monkeypatch):
    rels, fresh, seen = _fresh_then(tmp_path, monkeypatch)
    path = tmp_path / "autolabel" / autolabel.PASS1_NAME
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lines[1]["version"] = "autolabel-v0"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    assert autolabel.main(["--root", str(tmp_path), "--no-sam", "--resume"]) == 0
    assert seen == [rels[1]]
    assert _tasks(tmp_path) == fresh
    assert all(line["version"] == autolabel.MODEL_VERSION for line in _pass1(tmp_path))


def test_without_resume_the_checkpoint_is_not_reused(tmp_path, monkeypatch):
    rels, fresh, seen = _fresh_then(tmp_path, monkeypatch)
    boxes = tmp_path / "autolabel" / "boxes.jsonl"
    boxes.write_text(json.dumps({"file": rels[0], "boxes": []}) + "\n", encoding="utf-8")
    assert autolabel.main(["--root", str(tmp_path), "--no-sam"]) == 0
    assert seen == rels
    assert [line["file"] for line in _pass1(tmp_path)] == rels
    assert not boxes.exists()
    assert _tasks(tmp_path) == fresh


def test_resume_hands_the_flag_to_the_sam_stage_and_counts_done_frames(
    tmp_path, monkeypatch, capsys
):
    rels = _dataset(tmp_path, n=2)
    _bars_everywhere(monkeypatch)
    box = {"class": "enemy", "x": 755, "y": 310, "w": 90, "h": 95, "score": 0.9, "fallback": False}
    work = tmp_path / "autolabel"
    work.mkdir()
    (work / "boxes.jsonl").write_text(
        json.dumps({"file": rels[0], "boxes": [box]}) + "\n", encoding="utf-8"
    )
    seen = {}

    def fake_segment(root, prompts, out, model, resume=False):
        seen["resume"] = resume
        with out.open("a", encoding="utf-8") as dst:
            dst.write(json.dumps({"file": rels[1], "boxes": [box]}) + "\n")

    monkeypatch.setattr(autolabel, "run_segment", fake_segment)
    assert autolabel.main(["--root", str(tmp_path), "--resume"]) == 0
    assert seen["resume"] is True
    assert "1 SAM frames already done" in capsys.readouterr().out
    labels = coco.from_labelstudio(_tasks(tmp_path), use_predictions=True)
    assert set(labels) == set(rels)


@pytest.mark.parametrize("resume", [False, True])
def test_run_segment_passes_resume_only_when_asked(tmp_path, monkeypatch, resume):
    calls = []
    monkeypatch.setattr(autolabel.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(autolabel.subprocess, "run", lambda args, **kw: calls.append(args))
    prompts, out = tmp_path / "prompts.jsonl", tmp_path / "boxes.jsonl"
    autolabel.run_segment(tmp_path, prompts, out, "some-model", resume=resume)
    assert calls[0][:2] == ["uv", "run"]
    assert ("--resume" in calls[0]) is resume
