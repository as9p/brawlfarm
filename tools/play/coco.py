"""The Label Studio export turned into the COCO folder RF-DETR trains on.

The split is by source video, never by frame: consecutive frames of one video look alike, so a
frame-wise split would put near-copies on both sides of it and score a model on pictures it has
already seen. A source is assigned by hashing its name, which keeps the split stable when more
videos are added later. A recorder or capture session is one long sitting, though, so it is
split by time window instead: each window of WINDOW_MINUTES is its own split key. Frames with
no boxes are kept: a detector needs to be shown the screens where there is nothing to find.

    uv run python tools/play/coco.py export.json
    uv run python tools/play/coco.py export.json --root D:/play-data --from-predictions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from tools.play import classes, dataset

SPLITS = (("train", 0.8), ("valid", 0.1), ("test", 0.1))
# Below this many split keys a real split would leave a side empty, so the kit builds a smoke
# layout instead: the same frames in all three folders, good enough to prove the pipeline runs.
SMOKE_MIN = 3
# Hashed whole, a session puts every frame of it, and every rare tap anchor it shows, on one
# side of the split, which leaves valid and test with no anchor truth at all. So a session frame
# is keyed by the window of this many minutes its file name's time falls in.
WINDOW_MINUTES = 5
# ingest_youtube.py names a video's source yt-<video id>; a video is always split whole.
YOUTUBE_PREFIX = "yt-"
# Joins a session to its window in a split key. A source name can never hold it.
WINDOW_MARK = "@"
STAMP_RE = re.compile(r"(?<!\d)(\d{8})-(\d{6})(?!\d)")
# A tap anchor needs this many truth boxes in valid and in test before a threshold or a score
# drawn from that split means anything.
TRUTH_MIN = 30
# The templates are the only label source for tap anchors. An anchor no template maps to has no
# truth anywhere, so it is reported as having no label source rather than warned about.
LABELLED_ANCHORS = classes.TAP_ANCHORS & frozenset(classes.TEMPLATE_CLASS.values())


def _frame_time(source: str, file_name: str) -> datetime | None:
    """The time in a frame's file name, or None when it carries none.

    The source's own name is cut off first: a recorder session is named after the second it
    started, and that stamp is the session's, not the frame's.
    """
    stem = Path(file_name).stem
    if stem.startswith(f"{source}-"):
        stem = stem[len(source) + 1 :]
    match = STAMP_RE.search(stem)
    if match is None:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def split_key(source: str, file_name: str) -> str:
    """What a frame is split by: its source, or `<session>@<window>` for a session frame.

    A YouTube video stays whole. Any other source whose frame names carry a time is cut into
    windows of WINDOW_MINUTES, each named after its first minute, so the frames of one window
    always land together and a frame gets the same key every run. A frame with no time in its
    name falls back to its whole source.
    """
    if source.startswith(YOUTUBE_PREFIX):
        return source
    when = _frame_time(source, file_name)
    if when is None:
        return source
    minutes = when.hour * 60 + when.minute
    start = minutes - minutes % WINDOW_MINUTES
    return f"{source}{WINDOW_MARK}{when:%Y%m%d}-{start // 60:02d}{start % 60:02d}"


def split_of(source: str) -> str:
    """Which split a source belongs to, from its name alone, the same answer every run."""
    point = int(hashlib.sha256(source.encode()).hexdigest()[:8], 16) / 2**32
    cumulative = 0.0
    for name, ratio in SPLITS:
        cumulative += ratio
        if point < cumulative:
            return name
    return SPLITS[-1][0]


def _largest_split(assigned: dict[str, str], *, at_least: int = 1) -> str | None:
    """The split holding the most sources, ties going to the one earlier in SPLITS."""
    counts = Counter(assigned.values())
    names = [name for name, _ratio in SPLITS]
    candidates = [name for name in names if counts[name] >= at_least]
    if not candidates:
        return None
    return max(candidates, key=lambda name: (counts[name], -names.index(name)))


def assign_splits(sources: list[str]) -> dict[str, str]:
    """Map each source to its split, then repair any split the hashing left empty.

    With a handful of sources the hash can easily put none of them in a split, and an empty
    train set trains on nothing while an empty valid set scores on nothing. Every empty split
    is filled from the split that currently holds the most sources, train first, and then the
    labels are swapped if need be so train is never the smaller side: the whole point of the
    ratios is that most of the data trains. A source never spans splits, and the repair reads
    only counts and sorted names, so the same sources give the same answer in any order.
    """
    assigned = {source: split_of(source) for source in sorted(sources)}
    if len(assigned) < SMOKE_MIN:
        return {source: "train" for source in assigned}
    for name, _ratio in SPLITS:
        if any(split == name for split in assigned.values()):
            continue
        donor = _largest_split(assigned, at_least=2)
        if donor is None:
            continue
        moved = sorted(source for source, split in assigned.items() if split == donor)[-1]
        assigned[moved] = name
    biggest = _largest_split(assigned)
    if biggest is not None and biggest != "train":
        swap = {"train": biggest, biggest: "train"}
        assigned = {source: swap.get(split, split) for source, split in assigned.items()}
    return assigned


def _file_of(task: dict) -> str:
    """The frame path a task points at, as this kit imported it."""
    data = task.get("data", {})
    if data.get("file"):
        return str(data["file"])
    image = str(data.get("image", ""))
    if "?d=" not in image:
        raise ValueError(f"task has no frame path: {data!r}")
    return image.split("?d=", 1)[1]


def _result_of(task: dict, *, use_predictions: bool) -> list[dict] | None:
    """The result list to read, or None when the task has nothing to take.

    A frame the templates drafted nothing on is a frame with no control on it, which is exactly
    the negative the detector has to learn from, so in predictions mode a missing prediction is
    an empty result and never a reason to drop the frame. In annotation mode a task nobody has
    opened is unlabelled and is dropped, while an annotation with an empty result is a labeller
    saying there is nothing here, and is kept.
    """
    if use_predictions:
        predictions = task.get("predictions") or []
        return predictions[0].get("result", []) if predictions else []
    for annotation in task.get("annotations") or []:
        # A cancelled annotation is the labeller saying the frame is unusable, not an empty one.
        if not annotation.get("was_cancelled"):
            return annotation.get("result", [])
    return None


def from_labelstudio(export: list[dict], *, use_predictions: bool = False) -> dict[str, list[dict]]:
    """Read an export into frame path -> boxes in pixels.

    Label Studio stores a box as percentages of the picture it showed, so the pixels come back
    through that picture's size rather than the dataset's, which keeps the maths honest if a
    frame was ever imported at another size.
    """
    labels: dict[str, list[dict]] = {}
    for task in export:
        result = _result_of(task, use_predictions=use_predictions)
        if result is None:
            continue
        boxes = []
        for item in result:
            value = item.get("value", {})
            names = value.get("rectanglelabels") or []
            if not names:
                continue
            cls = names[0]
            if cls not in classes.CLASSES:
                raise ValueError(f"unknown class in the export: {cls!r}")
            width = item.get("original_width", dataset.FRAME_W)
            height = item.get("original_height", dataset.FRAME_H)
            boxes.append(
                {
                    "class": cls,
                    "x": round(value["x"] * width / 100),
                    "y": round(value["y"] * height / 100),
                    "w": round(value["width"] * width / 100),
                    "h": round(value["height"] * height / 100),
                }
            )
        labels[_file_of(task)] = boxes
    return labels


def _source_of(root: Path, rel: str) -> tuple[Path, str]:
    """Resolve one export path against the root, refusing anything outside `frames/`.

    The path comes out of a file the owner exported, so it is untrusted: `..` in it would copy
    whatever it reached into the training folder.
    """
    frames = (root / "frames").resolve()
    path = (root / rel).resolve()
    if path == frames or frames not in path.parents:
        raise ValueError(f"frame path leaves the dataset: {rel!r}")
    return path, dataset.check_source(path.parent.name)


def _clear_coco(root: Path) -> Path:
    """Delete a previous build, after proving the folder is the one under this root."""
    coco = (root / "coco").resolve()
    if coco.name != "coco" or coco.parent != root.resolve():
        raise ValueError(f"refusing to clear {coco}")
    if coco.exists():
        shutil.rmtree(coco)
    return coco


def build(root: Path, labels: dict[str, list[dict]]) -> dict:
    """Write `coco/{train,valid,test}` plus `coco/split.json`, and return the summary."""
    root = Path(root)
    entries = []
    for rel in sorted(labels):
        path, source = _source_of(root, rel)
        entries.append((rel, path, split_key(source, path.name)))
    keys = sorted({key for _rel, _path, key in entries})
    assigned = assign_splits(keys)
    smoke = len(keys) < SMOKE_MIN

    per_split: dict[str, list[tuple[str, Path]]] = {name: [] for name, _ratio in SPLITS}
    for rel, path, key in entries:
        names = [name for name, _ratio in SPLITS] if smoke else [assigned[key]]
        for name in names:
            per_split[name].append((rel, path))

    if not per_split[SPLITS[0][0]]:
        # Last guard: an empty train folder is a training run that scores zero for no visible
        # reason, so refuse before deleting the previous build.
        raise ValueError("no training frames")

    coco = _clear_coco(root)
    categories = [
        {"id": classes.category_id(cls), "name": cls, "supercategory": "none"}
        for cls in classes.CLASSES
    ]
    frames: dict[str, int] = {}
    # Counted over the labels, not the folders: a smoke layout writes each frame three times.
    boxes: Counter[str] = Counter(
        box["class"] for rel, _path, _key in entries for box in labels[rel]
    )
    # Counted over each folder, so a smoke layout shows the same truth on every side, as it is.
    split_boxes = {
        name: Counter(box["class"] for rel, _path in per_split[name] for box in labels[rel])
        for name, _ratio in SPLITS
    }
    for name, _ratio in SPLITS:
        folder = coco / name
        folder.mkdir(parents=True, exist_ok=True)
        images = []
        annotations = []
        for image_id, (rel, path) in enumerate(per_split[name], start=1):
            blob = path.read_bytes()
            picture = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            if picture is None:
                raise ValueError(f"not a readable frame: {rel}")
            (folder / path.name).write_bytes(blob)
            height, width = picture.shape[:2]
            images.append(
                {"id": image_id, "file_name": path.name, "width": width, "height": height}
            )
            for box in labels[rel]:
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": classes.category_id(box["class"]),
                        "bbox": [box["x"], box["y"], box["w"], box["h"]],
                        "area": box["w"] * box["h"],
                        "iscrowd": 0,
                    }
                )
        (folder / "_annotations.coco.json").write_text(
            json.dumps(
                {"images": images, "annotations": annotations, "categories": categories}, indent=1
            ),
            encoding="utf-8",
        )
        frames[name] = len(images)

    lines = sorted(
        f"{rel}|{box['class']}|{box['x']}|{box['y']}|{box['w']}|{box['h']}"
        for rel, _path in per_split[SPLITS[0][0]]
        for box in labels[rel]
    )
    summary = {
        "sources": assigned,
        "smoke": smoke,
        "frames": frames,
        "boxes": {cls: boxes[cls] for cls in classes.CLASSES},
        "split_boxes": {
            name: {cls: split_boxes[name][cls] for cls in classes.CLASSES}
            for name, _ratio in SPLITS
        },
        "training_set_hash": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
    }
    (coco / "split.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("export", type=Path, help="the JSON file Label Studio exported")
    ap.add_argument("--root", type=Path, default=None, help="dataset root")
    ap.add_argument(
        "--from-predictions",
        action="store_true",
        help="take the template pre-labels instead of the labelled boxes",
    )
    args = ap.parse_args(argv)

    root = args.root if args.root is not None else dataset.default_root()
    export = json.loads(Path(args.export).read_text(encoding="utf-8"))
    labels = from_labelstudio(export, use_predictions=args.from_predictions)
    if not labels:
        print(f"{args.export} has no usable task")
        return 1

    summary = build(root, labels)
    counts = ", ".join(f"{name} {summary['frames'][name]}" for name, _ratio in SPLITS)
    sources = {key.split(WINDOW_MARK, 1)[0] for key in summary["sources"]}
    print(
        f"{len(labels)} frames from {len(sources)} sources in {len(summary['sources'])} "
        f"split keys: {counts}"
    )
    if summary["smoke"]:
        print("smoke layout: the same frames are in all three splits")
    # Every class, zeros included: a class with no boxes is the thing to notice.
    for cls in classes.CLASSES:
        print(f"  {cls} {summary['boxes'][cls]}")
    print(f"training set hash {summary['training_set_hash'][:16]}")
    report_anchor_truth(summary)
    return 0


def report_anchor_truth(summary: dict) -> None:
    """Print each tap anchor's truth per split, and warn loudly where valid or test is thin.

    Only a warning: the build is already written, and a thin split is a reason to collect more
    frames, not to throw away the ones there are.
    """
    per_split = summary["split_boxes"]
    anchors = [cls for cls in classes.CLASSES if cls in classes.TAP_ANCHORS]
    thin = []
    print("tap-anchor truth per split:")
    for cls in anchors:
        if cls not in LABELLED_ANCHORS:
            print(f"  {cls}: no label source")
            continue
        print(f"  {cls}: " + ", ".join(f"{name} {per_split[name][cls]}" for name, _ in SPLITS))
        valid, test = per_split["valid"][cls], per_split["test"][cls]
        if min(valid, test) < TRUTH_MIN:
            thin.append((cls, valid, test))
    if not thin:
        return
    print("")
    for cls, valid, test in thin:
        print(
            f"WARNING: {cls} has {valid} truth boxes in valid and {test} in test, "
            f"fewer than {TRUTH_MIN}"
        )
    print("WARNING: a threshold or a score drawn from so few boxes means nothing, so the model")
    print("WARNING: is not proven on these anchors. Collect more frames that show them.")
    print("")


if __name__ == "__main__":
    raise SystemExit(main())
