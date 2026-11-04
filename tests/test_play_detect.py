"""The run-time detector: pre-processing and decoding pinned to the tools that exported the
model, per-class thresholds judged one class at a time, a missing model told apart from a
broken one, and an import that costs nothing while shadow mode is off."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from brawlfarm.play import detect
from tools.play import classes, export, thresholds


class FakeSession:
    """An onnxruntime session as far as detect.py is concerned: run() and get_providers()."""

    def __init__(
        self,
        path: Path,
        *,
        dets: Any = None,
        logits: Any = None,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
    ) -> None:
        self.path = path
        self.dets = dets
        self.logits = logits
        self.providers = list(providers)
        self.asked: list[str] = []
        self.feeds: dict[str, Any] = {}

    def run(self, names: list[str], feeds: dict[str, Any]) -> list[Any]:
        self.asked = list(names)
        self.feeds = dict(feeds)
        given = {"dets": self.dets, "labels": self.logits}
        return [given[name] for name in names]

    def get_providers(self) -> list[str]:
        return list(self.providers)


def meta_doc(**over: Any) -> dict[str, Any]:
    """A play.json the way tools/play/export.py writes it, with two classes."""
    doc: dict[str, Any] = {
        "model": "play-v0",
        "rfdetr": "1.3.0",
        "input": {
            "name": "input",
            "size": 384,
            "layout": "NCHW",
            "color": "RGB",
            "mean": list(detect.MEAN),
            "std": list(detect.STD),
        },
        "outputs": {"boxes": "dets", "logits": "labels", "box_format": "cxcywh"},
        "classes": ["self", "enemy"],
        "thresholds": {"self": 0.5, "enemy": 0.8},
        "training_set_hash": "a1b2c3",
        "smoke": True,
        "validation": {},
        "exported": "2026-09-20T00:00:00+00:00",
    }
    doc.update(over)
    return doc


def write_model(
    folder: Path,
    meta: dict[str, Any] | None = None,
    *,
    model: bool = True,
    text: str | None = None,
) -> Path:
    """A models folder on disk. The .onnx is a placeholder: no test opens it."""
    folder.mkdir(parents=True, exist_ok=True)
    if model:
        (folder / detect.MODEL_FILE).write_bytes(b"onnx goes here")
    if text is not None:
        (folder / detect.META_FILE).write_text(text, encoding="utf-8")
    elif meta is not None:
        (folder / detect.META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    return folder


def load(folder: Path, **session: Any) -> detect.Detector:
    made: list[FakeSession] = []

    def factory(path: Path) -> FakeSession:
        made.append(FakeSession(path, **session))
        return made[-1]

    loaded = detect.Detector.load(folder, session_factory=factory)
    loaded.fake = made[0]  # type: ignore[attr-defined]
    return loaded


def test_mean_and_std_are_the_tools_constants():
    assert detect.MEAN == thresholds.MEAN
    assert detect.STD == thresholds.STD


def test_preprocess_matches_the_tools():
    rng = np.random.default_rng(20260920)
    bgr = rng.integers(0, 256, size=(900, 1600, 3), dtype=np.uint8)
    mine = detect.preprocess(bgr, 384)
    theirs = thresholds.preprocess(bgr, 384)
    assert mine.shape == (1, 3, 384, 384)
    assert mine.dtype == theirs.dtype
    assert np.array_equal(mine, theirs)


def test_decode_matches_the_tools():
    rng = np.random.default_rng(626)
    dets = rng.random((300, 4))
    names = classes.CLASSES
    logits = rng.normal(0.0, 3.0, size=(300, len(names) + 1))
    bars = {name: 0.30 + 0.04 * index for index, name in enumerate(names)}

    mine = detect.decode(dets, logits, 1600, 900, names, bars)
    every = thresholds.decode(dets, logits, 1600, 900, floor=0.0)
    theirs = [box for box in every if box["score"] >= bars[names[box["class_index"]]]]

    # the thresholds have to bite for the comparison to mean anything
    assert 0 < len(theirs) < len(every)
    assert len(mine) == len(theirs)
    for got, want in zip(mine, theirs, strict=True):
        assert got.cls == names[want["class_index"]]
        assert got.score == pytest.approx(want["score"], abs=1e-9)
        assert got.x == pytest.approx(want["x"], abs=1e-9)
        assert got.y == pytest.approx(want["y"], abs=1e-9)
        assert got.w == pytest.approx(want["w"], abs=1e-9)
        assert got.h == pytest.approx(want["h"], abs=1e-9)


def test_decode_takes_a_batch_of_one():
    dets = np.array([[[0.5, 0.5, 0.25, 0.5]]])
    logits = np.array([[[0.0, -5.0, -9.0]]])
    boxes = detect.decode(dets, logits, 1600, 900, ("self", "enemy"), {"self": 0.5, "enemy": 0.8})
    assert [box.cls for box in boxes] == ["self"]


def test_each_class_is_judged_by_its_own_threshold():
    # sigmoid(0.0) is exactly 0.5, the self threshold: a box at its threshold is kept.
    dets = np.array(
        [[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]], dtype=np.float64
    )
    logits = np.array([[0.0, -5.0, -9.0], [-0.1, 2.0, -9.0], [0.41, 0.41, -9.0]], dtype=np.float64)
    boxes = detect.decode(dets, logits, 1600, 900, ("self", "enemy"), {"self": 0.5, "enemy": 0.8})

    # query 0: self at exactly 0.5 kept, enemy far below dropped.
    # query 1: self just under 0.5 dropped, enemy at 0.88 kept.
    # query 2: one score of about 0.60 passes self and fails enemy.
    assert sorted((box.cls, round(box.score, 3)) for box in boxes) == [
        ("enemy", 0.881),
        ("self", 0.5),
        ("self", 0.601),
    ]
    assert [box.score for box in boxes] == sorted((box.score for box in boxes), reverse=True)


def test_models_dir_follows_home(tmp_path, monkeypatch):
    from brawlfarm.core import config

    monkeypatch.setattr(config, "HOME_DIR", tmp_path)
    assert detect.models_dir() == tmp_path / "models"


def test_load_missing_for_an_empty_folder(tmp_path):
    with pytest.raises(detect.ModelMissing):
        detect.Detector.load(tmp_path, session_factory=FakeSession)


def test_load_missing_when_only_one_file_is_there(tmp_path):
    only_model = write_model(tmp_path / "onnx")
    with pytest.raises(detect.ModelMissing):
        detect.Detector.load(only_model, session_factory=FakeSession)

    only_meta = write_model(tmp_path / "json", meta_doc(), model=False)
    with pytest.raises(detect.ModelMissing):
        detect.Detector.load(only_meta, session_factory=FakeSession)


@pytest.mark.parametrize(
    ("meta", "key"),
    [
        (meta_doc(thresholds={"self": 0.5}), "enemy"),
        (meta_doc(thresholds={"self": 0.0, "enemy": 0.8}), "self"),
        (meta_doc(thresholds={"self": 1.5, "enemy": 0.8}), "self"),
        (meta_doc(thresholds={"self": "0.5", "enemy": 0.8}), "self"),
        (meta_doc(classes=["self", "self"]), "classes"),
        (meta_doc(classes=[]), "classes"),
        (meta_doc(classes="self"), "classes"),
        (meta_doc(input={"name": "input", "size": 0}), "size"),
        (meta_doc(input={"name": "input", "size": 384.5}), "size"),
        (meta_doc(input={"name": 7, "size": 384}), "name"),
        (meta_doc(outputs={"boxes": "dets"}), "logits"),
        (meta_doc(outputs={"boxes": None, "logits": "labels"}), "boxes"),
        (meta_doc(input=["input", 384]), "input"),
    ],
)
def test_load_invalid_names_the_key(tmp_path, meta, key):
    folder = write_model(tmp_path / key.replace(".", "_") / str(abs(hash(str(meta)))), meta)
    with pytest.raises(detect.ModelInvalid) as caught:
        detect.Detector.load(folder, session_factory=FakeSession)
    assert key in str(caught.value)


def test_load_invalid_for_a_json_that_is_not_an_object(tmp_path):
    folder = write_model(tmp_path, text="[1, 2, 3]")
    with pytest.raises(detect.ModelInvalid):
        detect.Detector.load(folder, session_factory=FakeSession)


def test_load_invalid_for_a_json_that_is_not_json(tmp_path):
    folder = write_model(tmp_path, text="{oops")
    with pytest.raises(detect.ModelInvalid):
        detect.Detector.load(folder, session_factory=FakeSession)


def test_load_carries_the_model_facts(tmp_path):
    folder = write_model(tmp_path, meta_doc())
    loaded = load(folder, providers=("CUDAExecutionProvider", "CPUExecutionProvider"))

    assert loaded.name == "play-v0"
    assert loaded.smoke is True
    assert loaded.training_set_hash == "a1b2c3"
    assert loaded.classes == ("self", "enemy")
    assert loaded.size == 384
    assert loaded.provider == "CUDAExecutionProvider"
    assert loaded.fake.path == folder / detect.MODEL_FILE


def test_load_takes_the_certification_export_writes(tmp_path):
    """export.py marks each class's threshold certified or not. The detector runs in shadow mode
    only and reads none of it, so the extra keys load and change nothing."""
    picked, validation = export.pick_thresholds(
        ["self", "enemy"], {0: [0.9] * 40, 1: []}, {0: [True] * 40, 1: []}, {0: 40, 1: 0}, 0.9
    )
    assert validation["self"]["certified"] is True
    assert validation["enemy"]["certified"] is False

    loaded = load(write_model(tmp_path, meta_doc(thresholds=picked, validation=validation)))
    assert loaded.classes == ("self", "enemy")
    assert picked == {"self": 0.30, "enemy": 0.95}


def test_load_defaults_smoke_and_hash(tmp_path):
    meta = meta_doc()
    del meta["smoke"]
    del meta["training_set_hash"]
    loaded = load(write_model(tmp_path, meta))
    assert loaded.smoke is False
    assert loaded.training_set_hash == ""


def test_detect_returns_pixel_boxes(tmp_path):
    dets = np.array([[[0.5, 0.5, 0.25, 0.5]]], dtype=np.float32)
    logits = np.array([[[3.0, -9.0, -9.0]]], dtype=np.float32)
    loaded = load(write_model(tmp_path, meta_doc()), dets=dets, logits=logits)

    boxes = loaded.detect(np.zeros((900, 1600, 3), dtype=np.uint8))
    assert len(boxes) == 1
    assert (boxes[0].cls, boxes[0].x, boxes[0].y, boxes[0].w, boxes[0].h) == (
        "self",
        600.0,
        225.0,
        400.0,
        450.0,
    )
    # the two outputs are asked for by name, boxes first
    assert loaded.fake.asked == ["dets", "labels"]
    assert loaded.fake.feeds["input"].shape == (1, 3, 384, 384)


def test_detect_scales_to_the_frame_it_was_given(tmp_path):
    dets = np.array([[[0.5, 0.5, 0.25, 0.5]]], dtype=np.float32)
    logits = np.array([[[3.0, -9.0, -9.0]]], dtype=np.float32)
    loaded = load(write_model(tmp_path, meta_doc()), dets=dets, logits=logits)

    boxes = loaded.detect(np.zeros((450, 800, 3), dtype=np.uint8))
    assert (boxes[0].x, boxes[0].y, boxes[0].w, boxes[0].h) == (300.0, 112.5, 200.0, 225.0)


def test_detect_invalid_when_the_logits_have_the_wrong_width(tmp_path):
    dets = np.array([[[0.5, 0.5, 0.25, 0.5]]], dtype=np.float32)
    logits = np.array([[[3.0, -9.0, -9.0, -9.0]]], dtype=np.float32)
    loaded = load(write_model(tmp_path, meta_doc()), dets=dets, logits=logits)

    with pytest.raises(detect.ModelInvalid) as caught:
        loaded.detect(np.zeros((900, 1600, 3), dtype=np.uint8))
    assert "4" in str(caught.value) and "3" in str(caught.value)


def test_importing_the_module_pulls_nothing_heavy():
    code = (
        "import sys, brawlfarm.play.detect; "
        "print([m for m in ('onnxruntime', 'av', 'cv2') if m in sys.modules])"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"
