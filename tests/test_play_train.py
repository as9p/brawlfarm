"""train.py's anchor oversampling, on a hand-built COCO folder with placeholder frames.

Nothing here imports torch or rfdetr: the training call is a stand-in that records what it was
handed. rfdetr's dataset is torchvision's CocoDetection, one sample per image id of the COCO
file, so the feed is read back the same way, and a copy that would collapse into its original
in training goes missing here too.
"""

from __future__ import annotations

import json
import os
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from tools.play import classes, train

CATEGORIES = [
    {"id": 1, "name": "self"},
    {"id": 2, "name": "play_button"},
    {"id": 3, "name": "enemy"},
]
# per split: (frame name, category ids of its boxes); b and d hold the play_button tap anchor
FRAMES = {
    "train": [("a.jpg", [1]), ("b.jpg", [2, 3]), ("c.jpg", []), ("d.jpg", [2])],
    "valid": [("e.jpg", [2]), ("f.jpg", [1])],
    "test": [("g.jpg", [2])],
}


def _write_coco(root: Path, frames: dict | None = None) -> Path:
    """A COCO build the way coco.py lays it out: a folder per split, its frames beside the file."""
    coco = root / "coco"
    for split, listed in (frames or FRAMES).items():
        folder = coco / split
        folder.mkdir(parents=True)
        images, boxes = [], []
        for image_id, (name, categories) in enumerate(listed, start=1):
            (folder / name).write_bytes(b"placeholder")
            images.append({"id": image_id, "file_name": name, "width": 1600, "height": 900})
            for category in categories:
                boxes.append(
                    {
                        "id": len(boxes) + 1,
                        "image_id": image_id,
                        "category_id": category,
                        "bbox": [10, 20, 30, 40],
                        "area": 1200,
                        "iscrowd": 0,
                    }
                )
        doc = {"images": images, "annotations": boxes, "categories": CATEGORIES}
        (folder / train.ANNOTATIONS).write_text(json.dumps(doc), encoding="utf-8")
    return coco


def _write_split(root: Path) -> None:
    summary = {
        "sources": {},
        "frames": {"train": 4, "valid": 2, "test": 1},
        "boxes": {},
        "training_set_hash": "0" * 64,
        "smoke": False,
    }
    (root / "coco" / "split.json").write_text(json.dumps(summary), encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def fake_rfdetr(monkeypatch):
    """rfdetr.RFDETRNano as a stand-in that records every train call and trains nothing."""
    calls: list[dict] = []

    class Nano:
        def train(self, **kwargs):
            calls.append(kwargs)

    module = types.ModuleType("rfdetr")
    module.RFDETRNano = Nano
    monkeypatch.setitem(sys.modules, "rfdetr", module)
    # No torch either, unless a test hands in a stand-in: the device print must not load it.
    monkeypatch.setitem(sys.modules, "torch", None)
    return calls


def _fake_torch(
    monkeypatch, cuda: bool, name: str = "NVIDIA GeForce RTX 4080 SUPER", mps: bool | None = None
) -> None:
    """torch as a stand-in that sees a CUDA GPU or does not, with no GPU and no real torch.

    mps None leaves out torch.backends entirely, like an older torch with no backends.mps.
    """
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(
        is_available=lambda: cuda, get_device_name=lambda device=None: name
    )
    if mps is not None:
        module.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    monkeypatch.setitem(sys.modules, "torch", module)


def test_the_fixture_has_one_tap_anchor_class():
    assert "play_button" in classes.TAP_ANCHORS
    assert not {"self", "enemy"} & classes.TAP_ANCHORS


def test_build_feed_puts_every_train_image_with_a_tap_anchor_in_n_times(tmp_path):
    coco = _write_coco(tmp_path)
    assert train.build_feed(coco, tmp_path / "feed", 4) == 2
    feed = _read(tmp_path / "feed" / "train" / train.ANNOTATIONS)
    copies = Counter(Path(image["file_name"]).name for image in feed["images"])
    assert copies == {"a.jpg": 1, "b.jpg": 4, "c.jpg": 1, "d.jpg": 4}


def test_every_copy_survives_a_reader_that_keys_samples_by_image_id(tmp_path):
    """pycocotools fills coco.imgs and coco.anns by id and torchvision's CocoDetection is one
    sample per key of coco.imgs. Rebuilt the same way, a copy that kept its original's image id
    or box ids would be lost, and so would be lost in training."""
    coco = _write_coco(tmp_path)
    train.build_feed(coco, tmp_path / "feed", 3)
    feed = _read(tmp_path / "feed" / "train" / train.ANNOTATIONS)
    imgs = {image["id"]: image for image in feed["images"]}
    anns = {box["id"]: box for box in feed["annotations"]}
    assert len(imgs) == len(feed["images"]) == 4 + 2 * 2
    assert len(anns) == len(feed["annotations"]) == 4 + 3 * 2

    per_image: dict[int, list[int]] = defaultdict(list)
    for box in anns.values():
        per_image[box["image_id"]].append(box["category_id"])
    samples: dict[str, list[list[int]]] = defaultdict(list)
    for image_id in sorted(imgs):
        samples[Path(imgs[image_id]["file_name"]).name].append(sorted(per_image[image_id]))
    assert samples == {
        "a.jpg": [[1]],
        "b.jpg": [[2, 3]] * 3,
        "c.jpg": [[]],
        "d.jpg": [[2]] * 3,
    }
    assert feed["categories"] == CATEGORIES


def test_build_feed_never_repeats_valid_or_test(tmp_path):
    coco = _write_coco(tmp_path)
    train.build_feed(coco, tmp_path / "feed", 4)
    for split in ("valid", "test"):
        source = _read(coco / split / train.ANNOTATIONS)
        feed = _read(tmp_path / "feed" / split / train.ANNOTATIONS)
        assert feed["annotations"] == source["annotations"]
        assert [image["id"] for image in feed["images"]] == [
            image["id"] for image in source["images"]
        ]
        assert [Path(image["file_name"]).name for image in feed["images"]] == [
            image["file_name"] for image in source["images"]
        ]


def test_build_feed_points_every_image_at_its_frame_in_the_coco_build(tmp_path):
    coco = _write_coco(tmp_path)
    feed = tmp_path / "feed"
    train.build_feed(coco, feed, 4)
    for split in train.SPLITS:
        for image in _read(feed / split / train.ANNOTATIONS)["images"]:
            # rfdetr opens Path(split folder) / file_name, and an absolute name wins that join
            opened = feed / split / image["file_name"]
            assert Path(image["file_name"]).is_absolute()
            assert opened == coco / split / Path(image["file_name"]).name
            assert opened.is_file()
        # no frame is copied: the feed holds its annotations and nothing else
        assert [path.name for path in (feed / split).iterdir()] == [train.ANNOTATIONS]


def test_build_feed_of_repeat_one_writes_nothing(tmp_path):
    coco = _write_coco(tmp_path)
    assert train.build_feed(coco, tmp_path / "feed", 1) == 0
    assert not (tmp_path / "feed").exists()


def test_build_feed_with_no_tap_anchor_in_train_writes_nothing(tmp_path):
    frames = {**FRAMES, "train": [("a.jpg", [1]), ("c.jpg", [])]}
    coco = _write_coco(tmp_path, frames)
    assert train.build_feed(coco, tmp_path / "feed", 4) == 0
    assert not (tmp_path / "feed").exists()


def test_main_trains_on_a_feed_with_each_anchor_image_four_times(tmp_path, fake_rfdetr, capsys):
    _write_coco(tmp_path)
    _write_split(tmp_path)
    assert train.main(["--root", str(tmp_path)]) == 0

    (run,) = (tmp_path / "runs").iterdir()
    assert [call["dataset_dir"] for call in fake_rfdetr] == [str(run / "feed")]
    feed = _read(run / "feed" / "train" / train.ANNOTATIONS)
    assert len(feed["images"]) == 4 + 2 * 3
    out = capsys.readouterr().out
    assert "2 train images hold a tap anchor, each in the training feed 4 times" in out


def test_main_with_anchor_repeat_one_trains_on_the_coco_build(tmp_path, fake_rfdetr, capsys):
    _write_coco(tmp_path)
    _write_split(tmp_path)
    assert train.main(["--root", str(tmp_path), "--anchor-repeat", "1"]) == 0

    (run,) = (tmp_path / "runs").iterdir()
    assert [call["dataset_dir"] for call in fake_rfdetr] == [str(tmp_path / "coco")]
    assert not (run / "feed").exists()
    assert "anchor oversampling is off" in capsys.readouterr().out


def test_main_with_no_tap_anchor_in_train_trains_on_the_coco_build(tmp_path, fake_rfdetr, capsys):
    _write_coco(tmp_path, {**FRAMES, "train": [("a.jpg", [1]), ("c.jpg", [])]})
    _write_split(tmp_path)
    assert train.main(["--root", str(tmp_path)]) == 0

    assert [call["dataset_dir"] for call in fake_rfdetr] == [str(tmp_path / "coco")]
    assert "no train image holds a tap anchor" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["0", "-2", "1.5", "abc"])
def test_main_refuses_an_anchor_repeat_below_one(tmp_path, value, fake_rfdetr):
    with pytest.raises(SystemExit) as raised:
        train.main(["--root", str(tmp_path), "--anchor-repeat", value])
    assert raised.value.code == 2
    assert fake_rfdetr == []


def test_describe_device_names_the_gpu_with_cuda(monkeypatch, capsys):
    _fake_torch(monkeypatch, cuda=True)
    train.describe_device()
    out = capsys.readouterr().out
    assert "training device cuda (NVIDIA GeForce RTX 4080 SUPER)" in out
    assert "WARNING" not in out


def test_describe_device_warns_loudly_on_the_cpu(monkeypatch, capsys):
    _fake_torch(monkeypatch, cuda=False)
    train.describe_device()
    out = capsys.readouterr().out
    assert "training device cpu" in out
    warnings = [line for line in out.splitlines() if line.startswith("WARNING")]
    assert len(warnings) >= 3
    text = " ".join(warnings)
    assert "very slow" in text
    assert "CUDA build" in text
    assert "uv run tools/play/train.py" in text
    assert "NVIDIA display driver" in text
    assert "docs/play-training.md" in text
    # the play-gpu extra brings onnxruntime-gpu, not torch, so it is not the fix here
    assert "play-gpu" not in text


def test_describe_device_names_mps_and_does_not_warn(monkeypatch, capsys):
    _fake_torch(monkeypatch, cuda=False, mps=True)
    train.describe_device()
    out = capsys.readouterr().out
    assert "training device mps" in out
    assert "training device cpu" not in out
    assert "WARNING" not in out


def test_describe_device_does_not_crash_without_torch(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)
    train.describe_device()
    assert "training device unknown" in capsys.readouterr().out


def test_main_prints_the_device_right_before_training(tmp_path, fake_rfdetr, monkeypatch, capsys):
    _fake_torch(monkeypatch, cuda=False)
    nano = sys.modules["rfdetr"].RFDETRNano
    monkeypatch.setattr(nano, "train", lambda self, **kwargs: print("training starts"))
    _write_coco(tmp_path)
    _write_split(tmp_path)
    assert train.main(["--root", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert out.index("tap anchor") < out.index("training device cpu") < out.index("training starts")


def _ckpts(run: Path, names_and_mtimes: dict[str, int]) -> Path:
    """Empty checkpoint files in run, each stamped with a fixed modification time."""
    run.mkdir(parents=True, exist_ok=True)
    for name, mtime in names_and_mtimes.items():
        (run / name).write_bytes(b"")
        os.utime(run / name, (mtime, mtime))
    return run


def _names(paths: list[Path]) -> list[str]:
    return [path.name for path in paths]


def test_checkpoints_newest_first_sorts_numbered_files_by_epoch_not_text(tmp_path):
    run = _ckpts(
        tmp_path / "r", {"checkpoint_9.ckpt": 30, "checkpoint_10.ckpt": 10, "checkpoint_1.ckpt": 20}
    )

    got = train.checkpoints_newest_first(run)

    assert _names(got) == ["checkpoint_10.ckpt", "checkpoint_9.ckpt", "checkpoint_1.ckpt"]


def test_checkpoints_newest_first_leaves_out_pth_files(tmp_path):
    run = _ckpts(tmp_path / "r", {"checkpoint_3.ckpt": 10, "checkpoint_3.pth": 20, "best.pth": 30})

    assert _names(train.checkpoints_newest_first(run)) == ["checkpoint_3.ckpt"]


def test_checkpoints_newest_first_puts_a_fresh_last_ckpt_first(tmp_path):
    run = _ckpts(
        tmp_path / "r", {"checkpoint_1.ckpt": 10, "checkpoint_3.ckpt": 20, "last.ckpt": 30}
    )

    assert _names(train.checkpoints_newest_first(run)) == [
        "last.ckpt",
        "checkpoint_3.ckpt",
        "checkpoint_1.ckpt",
    ]


def test_checkpoints_newest_first_puts_a_stale_last_ckpt_behind_the_newer_numbered_files(tmp_path):
    # a run resumed at interval 1 leaves an old last.ckpt behind: it must not lead
    run = _ckpts(
        tmp_path / "r",
        {"last.ckpt": 10, "checkpoint_1.ckpt": 5, "checkpoint_3.ckpt": 20, "checkpoint_5.ckpt": 30},
    )

    assert _names(train.checkpoints_newest_first(run)) == [
        "checkpoint_5.ckpt",
        "checkpoint_3.ckpt",
        "last.ckpt",
        "checkpoint_1.ckpt",
    ]


def test_checkpoints_newest_first_of_a_lone_last_ckpt_or_nothing(tmp_path):
    assert _names(train.checkpoints_newest_first(_ckpts(tmp_path / "a", {"last.ckpt": 1}))) == [
        "last.ckpt"
    ]
    assert train.checkpoints_newest_first(_ckpts(tmp_path / "b", {})) == []


def _loader(outcomes: dict[str, object]):
    """A loader that returns {"epoch": n} for a name mapped to n and raises an exception mapped."""

    def load(path: Path) -> dict:
        outcome = outcomes[path.name]
        if isinstance(outcome, Exception):
            raise outcome
        return {"epoch": outcome}

    return load


def test_first_loadable_skips_a_damaged_file_and_returns_the_next(tmp_path, capsys):
    damaged = tmp_path / "last.ckpt"
    good = tmp_path / "checkpoint_3.ckpt"
    load = _loader({"last.ckpt": KeyError("filename 'storages' not found"), good.name: 3})

    assert train.first_loadable([damaged, good], load) == (good, 3)
    assert "skipping" in capsys.readouterr().out


def test_first_loadable_takes_the_first_that_loads_and_goes_no_further(tmp_path):
    first, second = tmp_path / "checkpoint_5.ckpt", tmp_path / "checkpoint_3.ckpt"
    asked: list[str] = []

    def load(path: Path) -> dict:
        asked.append(path.name)
        return {"epoch": 5}

    assert train.first_loadable([first, second], load) == (first, 5)
    assert asked == [first.name]


def test_first_loadable_is_none_when_nothing_loads_or_nothing_is_offered(tmp_path):
    load = _loader({"a.ckpt": OSError("short read"), "b.ckpt": EOFError("truncated")})

    assert train.first_loadable([tmp_path / "a.ckpt", tmp_path / "b.ckpt"], load) is None
    assert train.first_loadable([], load) is None


def test_newest_checkpoint_run_skips_a_newer_folder_with_no_checkpoint(tmp_path):
    runs = tmp_path / "runs"
    _ckpts(runs / "20260101-000000", {"checkpoint_1.ckpt": 1})
    (runs / "20260102-000000").mkdir()
    (runs / "20260103-000000").mkdir()
    (runs / "20260103-000000" / "metrics.csv").write_text("x", encoding="utf-8")

    assert train.newest_checkpoint_run(runs) == runs / "20260101-000000"


def test_newest_checkpoint_run_takes_the_last_name_among_runs_with_checkpoints(tmp_path):
    runs = tmp_path / "runs"
    _ckpts(runs / "20260101-000000", {"last.ckpt": 1})
    _ckpts(runs / "20260102-000000", {"last.ckpt": 1})

    assert train.newest_checkpoint_run(runs) == runs / "20260102-000000"


def test_newest_checkpoint_run_is_none_with_no_checkpoint_or_no_runs_folder(tmp_path):
    runs = tmp_path / "runs"
    (runs / "20260101-000000").mkdir(parents=True)
    (runs / "stray.ckpt").write_bytes(b"")  # a file, not a run folder

    assert train.newest_checkpoint_run(runs) is None
    assert train.newest_checkpoint_run(tmp_path / "missing") is None


def test_pick_checkpoint_with_a_path_tries_that_file_alone(tmp_path):
    runs = tmp_path / "runs"
    _ckpts(runs / "20260102-000000", {"checkpoint_9.ckpt": 1})
    named = _ckpts(runs / "20260101-000000", {"checkpoint_2.ckpt": 1}) / "checkpoint_2.ckpt"

    assert train.pick_checkpoint(named, runs, _loader({named.name: 2})) == (named, 2)


def test_pick_checkpoint_with_a_path_that_does_not_load_is_none(tmp_path, capsys):
    named = tmp_path / "checkpoint_2.ckpt"
    load = _loader({named.name: RuntimeError("bad")})

    assert train.pick_checkpoint(named, tmp_path, load) is None
    assert "nothing to resume" in capsys.readouterr().out


def test_pick_checkpoint_bare_resume_falls_back_inside_the_newest_run_only(tmp_path):
    runs = tmp_path / "runs"
    _ckpts(runs / "20260101-000000", {"checkpoint_7.ckpt": 1})
    newest = _ckpts(runs / "20260102-000000", {"checkpoint_3.ckpt": 10, "last.ckpt": 20})
    load = _loader({"last.ckpt": KeyError("storages"), "checkpoint_3.ckpt": 3})

    assert train.pick_checkpoint(True, runs, load) == (newest / "checkpoint_3.ckpt", 3)


def test_pick_checkpoint_bare_resume_never_goes_back_to_an_older_run(tmp_path):
    runs = tmp_path / "runs"
    _ckpts(runs / "20260101-000000", {"checkpoint_7.ckpt": 1})
    _ckpts(runs / "20260102-000000", {"last.ckpt": 20})

    load = _loader({"last.ckpt": EOFError("truncated"), "checkpoint_7.ckpt": 7})
    assert train.pick_checkpoint(True, runs, load) is None


def test_pick_checkpoint_bare_resume_with_nothing_to_resume_says_so(tmp_path, capsys):
    assert train.pick_checkpoint(True, tmp_path / "runs", _loader({})) is None
    assert "nothing to resume" in capsys.readouterr().out


@pytest.fixture
def picked(monkeypatch):
    """Stand-in for pick_checkpoint that records what --resume parsed to and finds nothing."""
    seen: list[object] = []
    monkeypatch.setattr(train, "pick_checkpoint", lambda wanted, runs: seen.append(wanted))
    return seen


def test_main_resume_with_no_value_means_the_newest_run(tmp_path, fake_rfdetr, picked):
    _write_coco(tmp_path)
    _write_split(tmp_path)

    assert train.main(["--root", str(tmp_path), "--resume"]) == 1

    assert picked == [True]
    assert fake_rfdetr == []


def test_main_resume_with_a_path_hands_that_path_over(tmp_path, fake_rfdetr, picked):
    _write_coco(tmp_path)
    _write_split(tmp_path)

    assert train.main(["--root", str(tmp_path), "--resume", str(tmp_path / "x.ckpt")]) == 1

    assert picked == [tmp_path / "x.ckpt"]


def test_main_without_resume_starts_a_fresh_run_and_passes_no_checkpoint(
    tmp_path, fake_rfdetr, picked
):
    _write_coco(tmp_path)
    _write_split(tmp_path)

    assert train.main(["--root", str(tmp_path)]) == 0

    assert picked == []
    assert fake_rfdetr[0]["resume"] is None
    assert fake_rfdetr[0]["checkpoint_interval"] == train.CHECKPOINT_INTERVAL


def test_main_passes_the_checkpoint_interval_through(tmp_path, fake_rfdetr):
    _write_coco(tmp_path)
    _write_split(tmp_path)

    assert train.main(["--root", str(tmp_path), "--checkpoint-interval", "5"]) == 0

    assert fake_rfdetr[0]["checkpoint_interval"] == 5


@pytest.mark.parametrize("value", ["0", "-1", "2.5", "abc"])
def test_main_refuses_a_checkpoint_interval_below_one(tmp_path, value, fake_rfdetr):
    with pytest.raises(SystemExit) as raised:
        train.main(["--root", str(tmp_path), "--checkpoint-interval", value])
    assert raised.value.code == 2
    assert fake_rfdetr == []
