"""The Label Studio export turned into the COCO folder RF-DETR trains on.

Synthetic tiny JPEGs only, and never a real export file: the shapes here are the ones the
tool has to survive, including a cancelled annotation and a path that tries to leave the set.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.play import classes, coco, train


def _task(rel: str, boxes: list[tuple[str, float, float, float, float]], **kw) -> dict:
    result = [
        {
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "original_width": 1600,
            "original_height": 900,
            "value": {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "rotation": 0,
                "rectanglelabels": [cls],
            },
        }
        for cls, x, y, w, h in boxes
    ]
    task = {"data": {"image": "/data/local-files/?d=" + rel, "file": rel}}
    task["annotations"] = [{"was_cancelled": False, "result": result}]
    task.update(kw)
    return task


def _write_frames(root: Path, sources: list[str], per_source: int = 2) -> list[str]:
    rels = []
    for source in sources:
        for n in range(1, per_source + 1):
            rel = f"frames/{source}/{source}-{n:06d}.jpg"
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            picture = np.full((90, 160, 3), n * 20, dtype=np.uint8)
            ok, buf = cv2.imencode(".jpg", picture)
            assert ok
            path.write_bytes(buf.tobytes())
            rels.append(rel)
    return rels


def _write_named(root: Path, source: str, names: list[str]) -> list[str]:
    rels = []
    for n, name in enumerate(names, start=1):
        rel = f"frames/{source}/{name}"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", np.full((90, 160, 3), n % 250, dtype=np.uint8))
        assert ok
        path.write_bytes(buf.tobytes())
        rels.append(rel)
    return rels


def _session_names(windows: int) -> list[str]:
    """Two frames per five-minute window of a capture: the window's first second and its last."""
    names = []
    for window in range(windows):
        for offset in (0, 299):
            when = datetime(2026, 11, 4, 6, 0) + timedelta(minutes=5 * window, seconds=offset)
            names.append(f"{when:%Y%m%d-%H%M%S}-{len(names):05d}.jpg")
    return names


def test_split_key_of_a_session_frame_is_its_five_minute_window():
    first = coco.split_key("capture-20261104b", "20261104-063000-00120.jpg")
    last = coco.split_key("capture-20261104b", "20261104-063459-00133.jpg")
    after = coco.split_key("capture-20261104b", "20261104-063500-00134.jpg")

    assert coco.WINDOW_MINUTES == 5
    assert first == last == "capture-20261104b@20261104-0630"
    assert after == "capture-20261104b@20261104-0635"


def test_split_key_reads_a_time_that_follows_the_source_name():
    key = coco.split_key("capture-20261104b", "capture-20261104b-20261104-063119-000133.jpg")

    assert key == "capture-20261104b@20261104-0630"


def test_split_key_keeps_a_youtube_source_whole():
    assert coco.split_key("yt-00000000001", "20261104-063119-00133.jpg") == "yt-00000000001"
    # frames.py numbers a video's frames too, and a video is still never cut into blocks
    assert coco.split_key("yt-00000000001", "yt-00000000001-000999.jpg") == "yt-00000000001"


def test_split_key_of_a_numbered_session_frame_is_its_block():
    assert coco.SESSION_BLOCK_FRAMES == 60
    # how frames.py names a saved frame: the source, then a number, no time
    assert coco.split_key("capture-20261104", "capture-20261104-000003.jpg") == (
        "capture-20261104@b0"
    )
    # a recorder session is named after its start, which is not the frame's time
    assert coco.split_key("20260916-223234", "20260916-223234-000123.jpg") == ("20260916-223234@b2")


def test_split_key_starts_a_block_at_each_multiple_of_the_block_size():
    size = coco.SESSION_BLOCK_FRAMES
    keys = {
        n: coco.split_key("capture-20261104", f"capture-20261104-{n:06d}.jpg")
        for n in (0, size - 1, size, 2 * size - 1, 2 * size)
    }

    assert keys[0] == keys[size - 1] == "capture-20261104@b0"
    assert keys[size] == keys[2 * size - 1] == "capture-20261104@b1"
    assert keys[2 * size] == "capture-20261104@b2"


@pytest.mark.parametrize(
    ("frames", "fewest", "most"),
    [
        # a 30-minute capture at one frame per 3 s
        (600, 8, 12),
        # the recorder session in the dataset
        (138, 2, 3),
    ],
)
def test_split_key_cuts_a_sitting_into_a_handful_of_blocks(frames, fewest, most):
    keys = {coco.split_key("capture", f"capture-{n:06d}.jpg") for n in range(frames)}

    assert fewest <= len(keys) <= most


def test_split_key_prefers_a_time_in_the_name_to_the_frame_number():
    timed = [
        coco.split_key("capture-20261104b", "capture-20261104b-20261104-063119-000133.jpg"),
        coco.split_key("capture-20261104b", "20261104-063119-000133.jpg"),
    ]

    assert timed == ["capture-20261104b@20261104-0630"] * 2
    assert coco.split_key("capture-20261104b", "capture-20261104b-000133.jpg") == (
        "capture-20261104b@b2"
    )


@pytest.mark.parametrize(
    ("source", "name"),
    [
        ("capture-20261104", "20261399-256199-00001.jpg"),
        ("capture-20261104", "capture-20261104-20261399-256199.jpg"),
        ("capture-20261104", "0001-menu.jpg"),
        # another source's number, a number with more after it, not a JPEG, no number at all
        ("capture-20261104", "capture-other-000003.jpg"),
        ("capture-20261104", "capture-20261104-000003-menu.jpg"),
        ("capture-20261104", "capture-20261104-000003.png"),
        ("capture-20261104", "capture-20261104-.jpg"),
    ],
)
def test_split_key_falls_back_to_the_whole_source_without_a_time_or_a_number(source, name):
    assert coco.split_key(source, name) == source


def test_split_key_is_the_same_every_run():
    names = _session_names(6)
    numbered = [f"capture-20261104b-{n:06d}.jpg" for n in range(600)]

    first = [coco.split_key("capture-20261104b", name) for name in names + numbered]

    assert first == [coco.split_key("capture-20261104b", name) for name in names + numbered]
    assert len(set(first[: len(names)])) == 6
    assert len(set(first[len(names) :])) == 10


def test_only_the_template_anchors_have_a_label_source():
    assert coco.LABELLED_ANCHORS == frozenset(classes.TEMPLATE_CLASS.values())
    assert classes.TAP_ANCHORS - coco.LABELLED_ANCHORS == {
        "skull_star",
        "team_up_panel",
        "event_tab",
    }


def test_split_of_is_stable_and_not_the_same_for_every_source():
    names = [f"yt-{n:011d}" for n in range(20)]

    first = [coco.split_of(name) for name in names]

    assert first == [coco.split_of(name) for name in names]
    assert len(set(first)) > 1


def test_assign_splits_of_ten_sources_leaves_no_split_empty():
    sources = [f"yt-{n:011d}" for n in range(10)]

    assigned = coco.assign_splits(sources)

    assert set(assigned) == set(sources)
    assert set(assigned.values()) == {"train", "valid", "test"}


def test_assign_splits_repairs_a_train_the_hash_left_empty():
    # These three hash to valid, valid, test, which used to train on nothing at all.
    assigned = coco.assign_splits(["src3", "src8", "src11"])

    assert sorted(assigned.values()) == ["test", "train", "valid"]


@pytest.mark.parametrize("count", range(3, 13))
def test_assign_splits_fills_every_split_and_keeps_train_the_largest(count):
    sources = [f"s{n}" for n in range(count)]

    assigned = coco.assign_splits(sources)

    sizes = {
        name: sum(1 for split in assigned.values() if split == name) for name, _ in coco.SPLITS
    }
    assert min(sizes.values()) >= 1
    assert sizes["train"] >= max(sizes["valid"], sizes["test"])


def test_assign_splits_does_not_depend_on_the_order_of_its_input():
    sources = [f"s{n}" for n in range(7)]

    assigned = coco.assign_splits(sources)

    assert assigned == coco.assign_splits(sources)
    assert assigned == coco.assign_splits(list(reversed(sources)))


def test_assign_splits_of_two_sources_keeps_them_all_in_train():
    assigned = coco.assign_splits(["alpha", "beta"])

    assert assigned == {"alpha": "train", "beta": "train"}


def test_from_labelstudio_reads_annotations_in_pixels():
    labels = coco.from_labelstudio(
        [_task("frames/a/a-000001.jpg", [("play_button", 25, 25, 10, 10)])]
    )

    assert labels == {
        "frames/a/a-000001.jpg": [{"class": "play_button", "x": 400, "y": 225, "w": 160, "h": 90}]
    }


def test_from_labelstudio_falls_back_to_the_image_url_for_the_file():
    task = _task("frames/a/a-000001.jpg", [])
    del task["data"]["file"]

    assert list(coco.from_labelstudio([task])) == ["frames/a/a-000001.jpg"]


def test_from_labelstudio_skips_a_cancelled_annotation():
    task = _task("frames/a/a-000001.jpg", [("play_button", 25, 25, 10, 10)])
    task["annotations"][0]["was_cancelled"] = True

    assert coco.from_labelstudio([task]) == {}


def test_from_labelstudio_reads_predictions_only_with_the_flag():
    task = _task("frames/a/a-000001.jpg", [])
    task["predictions"] = [
        _task("frames/a/a-000001.jpg", [("close_x", 50, 50, 5, 5)])["annotations"][0]
    ]

    assert coco.from_labelstudio([task]) == {"frames/a/a-000001.jpg": []}
    from_predictions = coco.from_labelstudio([task], use_predictions=True)
    assert from_predictions["frames/a/a-000001.jpg"][0]["class"] == "close_x"


def test_from_labelstudio_keeps_a_frame_with_no_prediction_as_a_negative():
    task = _task("frames/a/a-000001.jpg", [])
    task["predictions"] = []

    assert coco.from_labelstudio([task], use_predictions=True) == {"frames/a/a-000001.jpg": []}
    del task["predictions"]
    assert coco.from_labelstudio([task], use_predictions=True) == {"frames/a/a-000001.jpg": []}


def test_from_labelstudio_skips_a_task_nobody_has_annotated():
    task = _task("frames/a/a-000001.jpg", [])
    task["annotations"] = []

    assert coco.from_labelstudio([task]) == {}


def test_from_labelstudio_keeps_an_annotation_with_an_empty_result_as_a_negative():
    assert coco.from_labelstudio([_task("frames/a/a-000001.jpg", [])]) == {
        "frames/a/a-000001.jpg": []
    }


def test_from_labelstudio_refuses_an_unknown_class():
    task = _task("frames/a/a-000001.jpg", [("brawler", 25, 25, 10, 10)])

    with pytest.raises(ValueError, match="brawler"):
        coco.from_labelstudio([task])


def test_build_writes_three_splits_covering_every_frame(tmp_path):
    root = tmp_path / "play"
    rels = _write_frames(root, ["alpha", "beta", "gamma"])
    labels = {rel: [{"class": "play_button", "x": 10, "y": 20, "w": 30, "h": 40}] for rel in rels}

    summary = coco.build(root, labels)

    total = 0
    for split, _ratio in coco.SPLITS:
        blob = json.loads(
            (root / "coco" / split / "_annotations.coco.json").read_text(encoding="utf-8")
        )
        assert [c["id"] for c in blob["categories"]] == list(range(1, len(classes.CLASSES) + 1))
        assert blob["categories"][0]["supercategory"] == "none"
        assert len(blob["images"]) == summary["frames"][split]
        for image in blob["images"]:
            assert (root / "coco" / split / image["file_name"]).exists()
        for annotation in blob["annotations"]:
            assert annotation["bbox"] == [10, 20, 30, 40]
            assert annotation["area"] == 1200
            assert annotation["iscrowd"] == 0
        total += len(blob["images"])
    assert total == 6
    assert summary["smoke"] is False
    assert summary["boxes"]["play_button"] == 6
    # Numbered frames.py names, so each source's frames make up its first block.
    assert set(summary["sources"]) == {"alpha@b0", "beta@b0", "gamma@b0"}
    assert json.loads((root / "coco" / "split.json").read_text(encoding="utf-8")) == summary


def test_build_training_set_hash_moves_with_a_single_pixel(tmp_path):
    root = tmp_path / "play"
    rels = _write_frames(root, ["alpha", "beta", "gamma"])
    labels = {rel: [{"class": "play_button", "x": 10, "y": 20, "w": 30, "h": 40}] for rel in rels}

    before = coco.build(root, labels)["training_set_hash"]
    again = coco.build(root, labels)["training_set_hash"]
    labels[rels[0]] = [{"class": "play_button", "x": 11, "y": 20, "w": 30, "h": 40}]
    after = coco.build(root, labels)["training_set_hash"]

    assert before == again
    assert before != after


def test_build_keeps_a_frame_with_no_boxes(tmp_path):
    root = tmp_path / "play"
    rels = _write_frames(root, ["alpha", "beta", "gamma"])
    labels = {rel: [] for rel in rels}

    summary = coco.build(root, labels)

    assert sum(summary["frames"].values()) == 6
    assert summary["boxes"]["play_button"] == 0


def test_build_of_two_sources_is_a_smoke_layout(tmp_path):
    root = tmp_path / "play"
    rels = _write_frames(root, ["alpha", "beta"])
    labels = {rel: [] for rel in rels}

    summary = coco.build(root, labels)

    assert summary["smoke"] is True
    assert summary["frames"] == {"train": 4, "valid": 4, "test": 4}


def test_build_clears_a_previous_coco_folder(tmp_path):
    root = tmp_path / "play"
    rels = _write_frames(root, ["alpha", "beta", "gamma"])
    stale = root / "coco" / "train" / "stale-000001.jpg"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"old")

    coco.build(root, {rel: [] for rel in rels})

    assert not stale.exists()


def test_build_refuses_a_file_outside_the_frames_folder(tmp_path):
    root = tmp_path / "play"
    _write_frames(root, ["alpha", "beta", "gamma"])

    with pytest.raises(ValueError):
        coco.build(root, {"../../secret.jpg": []})


def test_main_builds_from_an_export_file(tmp_path, capsys):
    root = tmp_path / "play"
    rels = _write_frames(root, ["alpha", "beta", "gamma"])
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps([_task(rel, [("play_button", 25, 25, 10, 10)]) for rel in rels]),
        encoding="utf-8",
    )

    code = coco.main([str(export), "--root", str(root)])

    assert code == 0
    assert (root / "coco" / "split.json").exists()
    assert "6 frames" in capsys.readouterr().out


def test_build_refuses_to_write_an_empty_training_set(tmp_path):
    root = tmp_path / "play"
    _write_frames(root, ["alpha"])

    with pytest.raises(ValueError, match="no training frames"):
        coco.build(root, {})


def test_build_splits_a_session_by_window_and_a_video_whole(tmp_path):
    root = tmp_path / "play"
    videos = ["yt-00000000001", "yt-00000000002", "yt-00000000003"]
    rels = _write_frames(root, videos)
    rels += _write_named(root, "capture-20261104b", _session_names(24))

    summary = coco.build(root, {rel: [] for rel in rels})

    windows = [key for key in summary["sources"] if key.startswith("capture-20261104b@")]
    assert len(windows) == 24
    assert set(videos) <= set(summary["sources"])
    assert "capture-20261104b" not in summary["sources"]
    # The point of the windows: one long sitting no longer lands on a single side of the split.
    assert len({summary["sources"][key] for key in windows}) > 1
    assert sum(summary["frames"].values()) == len(rels)


def test_build_splits_a_numbered_session_by_block_and_a_video_whole(tmp_path):
    root = tmp_path / "play"
    videos = ["yt-00000000001", "yt-00000000002", "yt-00000000003"]
    rels = _write_frames(root, videos)
    # How frames.py saves a session: numbered names with no time, here two frames per block.
    numbers = range(0, 24 * coco.SESSION_BLOCK_FRAMES, coco.SESSION_BLOCK_FRAMES // 2)
    rels += _write_named(
        root, "capture-20261104", [f"capture-20261104-{n:06d}.jpg" for n in numbers]
    )
    labels = {rel: [] for rel in rels}

    summary = coco.build(root, labels)

    blocks = [key for key in summary["sources"] if key.startswith("capture-20261104@b")]
    assert len(blocks) == 24
    assert set(videos) <= set(summary["sources"])
    assert "capture-20261104" not in summary["sources"]
    assert len({summary["sources"][key] for key in blocks}) > 1
    assert sum(summary["frames"].values()) == len(rels)
    assert coco.build(root, dict(reversed(list(labels.items())))) == summary


def test_build_gives_the_same_split_whatever_the_order_of_the_labels(tmp_path):
    root = tmp_path / "play"
    rels = _write_frames(root, ["yt-00000000001"])
    rels += _write_named(root, "capture-20261104b", _session_names(12))
    labels = {rel: [{"class": "play_button", "x": 1, "y": 2, "w": 3, "h": 4}] for rel in rels}

    first = coco.build(root, labels)
    again = coco.build(root, dict(reversed(list(labels.items()))))

    assert again == first


def test_main_prints_every_class_and_warns_on_thin_anchor_truth(tmp_path, capsys):
    root = tmp_path / "play"
    rels = _write_frames(root, ["alpha", "beta", "gamma"])
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps([_task(rel, [("play_button", 25, 25, 10, 10)]) for rel in rels]),
        encoding="utf-8",
    )

    code = coco.main([str(export), "--root", str(root)])

    out = capsys.readouterr().out
    # A warning, never a refusal.
    assert code == 0
    assert "  play_button 6\n" in out
    assert "  close_x 0\n" in out
    summary = json.loads((root / "coco" / "split.json").read_text(encoding="utf-8"))
    per_split = ", ".join(
        f"{name} {summary['split_boxes'][name]['play_button']}" for name, _ in coco.SPLITS
    )
    assert f"  play_button: {per_split}\n" in out
    for anchor in ("skull_star", "team_up_panel", "event_tab"):
        assert f"  {anchor}: no label source\n" in out
        assert f"WARNING: {anchor}" not in out
    assert "WARNING: play_button has" in out
    assert "WARNING: close_x has 0 truth boxes in valid and 0 in test" in out


def test_main_does_not_warn_once_valid_and_test_hold_enough_truth(tmp_path, capsys):
    root = tmp_path / "play"
    # Two sources make a smoke layout, so valid and test each hold all 30 frames.
    rels = _write_named(root, "alpha", [f"alpha-{n:06d}.jpg" for n in range(15)])
    rels += _write_named(root, "beta", [f"beta-{n:06d}.jpg" for n in range(15)])
    boxes = [(anchor, 25, 25, 10, 10) for anchor in sorted(coco.LABELLED_ANCHORS)]
    export = tmp_path / "export.json"
    export.write_text(json.dumps([_task(rel, boxes) for rel in rels]), encoding="utf-8")

    assert coco.main([str(export), "--root", str(root)]) == 0

    assert "WARNING" not in capsys.readouterr().out


def test_train_describe_lists_classes_with_no_boxes(capsys):
    summary = {
        "sources": {"alpha": "train"},
        "smoke": False,
        "frames": {"train": 1, "valid": 0, "test": 0},
        "boxes": {"close_x": 0, "play_button": 3},
        "training_set_hash": "0" * 64,
    }

    train.describe(summary)

    out = capsys.readouterr().out
    assert "  close_x 0\n" in out
    assert "  play_button 3\n" in out


def test_train_describe_counts_split_keys_not_sources(capsys):
    summary = {
        "sources": {"capture@b0": "train", "capture@b1": "valid", "capture@b2": "test"},
        "smoke": False,
        "frames": {"train": 1, "valid": 1, "test": 1},
        "boxes": {"play_button": 3},
        "training_set_hash": "0" * 64,
    }

    train.describe(summary)

    out = capsys.readouterr().out
    assert "3 split keys: test 1, train 1, valid 1\n" in out
    assert "sources" not in out
