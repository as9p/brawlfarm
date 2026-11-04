# /// script
# requires-python = ">=3.13"
# dependencies = ["rfdetr[train,onnx]==1.10.1", "torch", "torchvision", "onnxruntime>=1.20", "opencv-python-headless"]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""Train RF-DETR nano on the COCO folder coco.py built.

Like export.py and segment.py, this script needs torch, so it is not part of the project
environment: the inline metadata above pins its own, with torch from the CUDA 12.8 index.
Run it from the repository root and it resolves everything else itself:

    uv run tools/play/train.py
    uv run tools/play/train.py --root D:/play-data --epochs 80 --batch 4 --accum 4

It refuses to start without coco/split.json, because a training run with no recorded split is
a model nobody can score later. num_workers is 0 on purpose: the dataloader's worker processes
deadlock on Windows.

Train images that show a tap anchor are rare, so each one goes into the training feed
--anchor-repeat times (4 by default; 1 turns it off). The feed is the COCO build's annotation
files rewritten into the run folder, run/feed, with the extra copies in train only: valid and
test go in once, and no frame is copied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.play import classes  # noqa: E402

ANNOTATIONS = "_annotations.coco.json"
SPLITS = ("train", "valid", "test")
# how many times a train image that shows a tap anchor goes into the training feed
ANCHOR_REPEAT = 4


def default_root() -> Path:
    """`<home>/datasets/play`, worked out by hand: this environment has no brawlfarm in it."""
    home = os.environ.get("BRAWLFARM_HOME")
    if home:
        return Path(home) / "datasets" / "play"
    local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(local) / "brawlfarm" / "datasets" / "play"


def describe(summary: dict) -> None:
    """Print what is about to be trained on, loudly enough that a smoke set is not mistaken."""
    counts = ", ".join(f"{name} {count}" for name, count in sorted(summary["frames"].items()))
    print(f"{len(summary['sources'])} split keys: {counts}")
    # Every class, zeros included: a class with no boxes trains nothing and should be seen.
    for name, count in sorted(summary["boxes"].items()):
        print(f"  {name} {count}")
    print(f"training set hash {summary['training_set_hash'][:16]}")
    if summary["smoke"]:
        print("")
        print("WARNING: this is a smoke set. The same frames are in all three splits, so the")
        print("WARNING: validation score is meaningless and the model must not ship.")
        print("")


def repeat_count(text: str) -> int:
    """An argparse type for --anchor-repeat: a whole number, 1 or more."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} is below 1")
    return value


def build_feed(coco: Path, feed: Path, repeat: int) -> int:
    """Write the training feed into feed, and return how many train images hold a tap anchor.

    Each train image that holds at least one tap-anchor box goes in `repeat` times. Every copy
    is a COCO image of its own, with a fresh image id and fresh box ids: rfdetr's dataset is one
    sample per image id, so a copy that kept its original's id would collapse back into it and
    train nothing. valid and test are written as they are and never repeated, so a threshold or
    a score is still drawn from frames counted once. No frame is copied: each file_name becomes
    the frame's absolute path in the COCO build, and rfdetr opens `Path(split folder) /
    file_name`, a join that an absolute path wins.

    Nothing is written when repeat is 1 or no train image holds a tap anchor; the caller then
    trains on the COCO build itself.
    """
    if repeat < 2 or not (coco / "train" / ANNOTATIONS).is_file():
        return 0
    docs = {
        split: json.loads((coco / split / ANNOTATIONS).read_text(encoding="utf-8"))
        for split in SPLITS
        if (coco / split / ANNOTATIONS).is_file()
    }
    train = docs["train"]
    names = {category["id"]: category["name"] for category in train["categories"]}
    anchored = {
        box["image_id"]
        for box in train["annotations"]
        if names[box["category_id"]] in classes.TAP_ANCHORS
    }
    if not anchored:
        return 0

    for split, doc in docs.items():
        folder = (coco / split).absolute()
        for image in doc["images"]:
            image["file_name"] = str(folder / image["file_name"])
    boxes: dict[int, list[dict]] = defaultdict(list)
    for box in train["annotations"]:
        boxes[box["image_id"]].append(box)
    originals = [image for image in train["images"] if image["id"] in anchored]
    next_image = max(image["id"] for image in train["images"]) + 1
    next_box = max(box["id"] for box in train["annotations"]) + 1
    for _ in range(repeat - 1):
        for image in originals:
            train["images"].append({**image, "id": next_image})
            for box in boxes[image["id"]]:
                train["annotations"].append({**box, "id": next_box, "image_id": next_image})
                next_box += 1
            next_image += 1

    for split, doc in docs.items():
        (feed / split).mkdir(parents=True, exist_ok=True)
        (feed / split / ANNOTATIONS).write_text(json.dumps(doc), encoding="utf-8")
    return len(anchored)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=None, help="dataset root")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    ap.add_argument(
        "--anchor-repeat",
        type=repeat_count,
        default=ANCHOR_REPEAT,
        help="how many times each train image with a tap anchor goes into the training feed "
        f"(default {ANCHOR_REPEAT}); 1 turns it off",
    )
    args = ap.parse_args(argv)

    root = args.root if args.root is not None else default_root()
    split = root / "coco" / "split.json"
    if not split.is_file():
        print(f"no {split}: run tools/play/coco.py on a Label Studio export first")
        return 1
    describe(json.loads(split.read_text(encoding="utf-8")))

    run = root / "runs" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    print(f"training into {run}")

    dataset = root / "coco"
    repeated = build_feed(dataset, run / "feed", args.anchor_repeat)
    if args.anchor_repeat == 1:
        print("anchor oversampling is off: every train image goes in once")
    elif repeated:
        held = "image holds" if repeated == 1 else "images hold"
        print(
            f"{repeated} train {held} a tap anchor, each in the training feed "
            f"{args.anchor_repeat} times: {repeated * (args.anchor_repeat - 1)} extra samples"
        )
        dataset = run / "feed"
    else:
        print("no train image holds a tap anchor: nothing to oversample")

    from rfdetr import RFDETRNano  # here, so a missing split fails before torch loads

    RFDETRNano().train(
        dataset_dir=str(dataset),
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.accum,
        output_dir=str(run),
        num_workers=0,
    )
    print(f"run folder {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
