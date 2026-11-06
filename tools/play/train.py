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
    uv run tools/play/train.py --resume
    uv run tools/play/train.py --resume D:/play-data/runs/20260918-101112/checkpoint_29.ckpt

It refuses to start without coco/split.json, because a training run with no recorded split is
a model nobody can score later. num_workers is 0 on purpose: the dataloader's worker processes
deadlock on Windows.

Train images that show a tap anchor are rare, so each one goes into the training feed
--anchor-repeat times (4 by default; 1 turns it off). The feed is the COCO build's annotation
files rewritten into the run folder, run/feed, with the extra copies in train only: valid and
test go in once, and no frame is copied.

A full checkpoint is kept every --checkpoint-interval epochs (2 by default; rfdetr's own default
is 10). --resume carries a crashed run on from one of them, in that checkpoint's run folder and on
that run's training feed, so --anchor-repeat has no effect on it. With no path it takes the newest
run that holds a checkpoint, and the newest checkpoint there that loads. Pass the --epochs, --batch
and --accum the run started with: the checkpoint does not set them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.play import classes  # noqa: E402

ANNOTATIONS = "_annotations.coco.json"
SPLITS = ("train", "valid", "test")
# how many times a train image that shows a tap anchor goes into the training feed
ANCHOR_REPEAT = 4
# epochs between the full checkpoints a run keeps
CHECKPOINT_INTERVAL = 2
# rfdetr's name for those checkpoints; the number is the epoch that finished, counted from 0
NUMBERED_CHECKPOINT = re.compile(r"checkpoint_(\d+)\.ckpt")


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


def describe_device() -> None:
    """Print the device rfdetr will train on, and shout when it is the CPU: a CPU run takes hours.

    This only reports. rfdetr still picks the device itself, and like this check it prefers CUDA,
    then Apple's MPS, then the CPU.
    """
    try:
        import torch  # here, like rfdetr, so the tests run without torch
    except ImportError:
        print("training device unknown: torch does not import")
        return
    if torch.cuda.is_available():
        print(f"training device cuda ({torch.cuda.get_device_name()})")
        return
    # an older torch has no backends.mps at all
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        print("training device mps")
        return
    print("training device cpu")
    print("")
    print("WARNING: torch sees no CUDA GPU, so this run trains on the CPU.")
    print("WARNING: CPU training is very slow: a 60-epoch run can take many hours.")
    print("WARNING: To train on the GPU, torch must be a CUDA build and the NVIDIA display driver")
    print("WARNING: must be new enough for CUDA 12.8. Start it as `uv run tools/play/train.py`:")
    print("WARNING: its inline metadata gives it its own environment with torch from the CUDA")
    print("WARNING: 12.8 index. A torch from any other environment may be a CPU-only build.")
    print("WARNING: See docs/play-training.md, Step 7: train, then run this again.")
    print("")


def repeat_count(text: str) -> int:
    """An argparse type for --anchor-repeat and --checkpoint-interval: a whole number, 1 or more."""
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


def load_checkpoint(path: Path) -> dict:
    """Load a whole checkpoint: nothing short of that proves the file is good."""
    import torch  # here, like rfdetr, so the tests run without torch

    # weights_only refuses a Lightning checkpoint, which holds more than tensors
    return torch.load(path, map_location="cpu", weights_only=False)


def newest_checkpoint_run(runs: Path) -> Path | None:
    """The last run folder by name that holds a checkpoint; the name is a UTC stamp.

    A crash before the first checkpoint leaves a newer folder with nothing to resume in it.
    """
    if not runs.is_dir():
        return None
    folders = sorted((folder for folder in runs.iterdir() if folder.is_dir()), reverse=True)
    return next((folder for folder in folders if any(folder.glob("*.ckpt"))), None)


def checkpoints_newest_first(run: Path) -> list[Path]:
    """Every full checkpoint in run, newest first.

    A numbered file sorts by the epoch in its name, which a copy of the folder cannot change.
    last.ckpt has no number, so it goes ahead of the first numbered file no newer than it. rfdetr
    rewrites it every epoch, so that is normally first, up to an interval ahead of the numbered
    files; if a crash damaged it, the load fails and the next file is tried. It is not always
    first because rfdetr does not write it at interval 1, so a run resumed at that interval leaves
    an old one behind.

    The .pth files are left out: they hold no optimizer state, so a resume restarts it cold.
    """

    def modified(path: Path) -> float:
        return path.stat().st_mtime

    def epoch(path: Path) -> int:
        return int(NUMBERED_CHECKPOINT.fullmatch(path.name)[1])

    files = list(run.glob("*.ckpt"))
    numbered = [path for path in files if NUMBERED_CHECKPOINT.fullmatch(path.name)]
    ordered = sorted(numbered, key=epoch, reverse=True)
    for path in sorted((path for path in files if path not in numbered), key=modified):
        no_newer = (at for at, other in enumerate(ordered) if modified(other) <= modified(path))
        ordered.insert(next(no_newer, len(ordered)), path)
    return ordered


def first_loadable(
    candidates: list[Path], load: Callable[[Path], dict] = load_checkpoint
) -> tuple[Path, int] | None:
    """The first candidate that loads with the epoch it last finished, or None when none does.

    A crash can leave a full-size file that does not load, so none is trusted until it has. A real
    last.ckpt failed with KeyError "filename 'storages' not found" while an older file worked.
    """
    for path in candidates:
        try:
            return path, int(load(path)["epoch"])
        except Exception as error:  # a damaged file raises many types; each one means skip it
            print(f"skipping {path}: {type(error).__name__}: {error}", flush=True)
    return None


def pick_checkpoint(
    wanted: Path | bool, runs: Path, load: Callable[[Path], dict] = load_checkpoint
) -> tuple[Path, int] | None:
    """The checkpoint --resume means, with the epoch it last finished.

    wanted is the file named on the command line, which is tried alone, or True for the newest run
    under runs that holds a checkpoint. Only that run is tried: an older one may have trained on
    other data, so going back to it is a choice to make by naming the file. Prints why and returns
    None when there is nothing to resume.
    """
    if wanted is True:
        run = newest_checkpoint_run(runs)
        if run is None:
            print(f"no run under {runs} holds a checkpoint: nothing to resume", flush=True)
            return None
        candidates, where = checkpoints_newest_first(run), run
    else:
        candidates, where = [wanted], wanted
    found = first_loadable(candidates, load)
    if found is None:
        print(f"no checkpoint loads from {where}: nothing to resume", flush=True)
    return found


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
    ap.add_argument(
        "--checkpoint-interval",
        type=repeat_count,
        default=CHECKPOINT_INTERVAL,
        help=f"keep a full checkpoint every this many epochs (default {CHECKPOINT_INTERVAL})",
    )
    ap.add_argument(
        "--resume",
        nargs="?",
        type=Path,
        const=True,
        default=None,
        metavar="PATH",
        help="carry a run on from this checkpoint, in its run folder; with no PATH, from the "
        "newest checkpoint that loads in the newest run that holds any",
    )
    args = ap.parse_args(argv)

    root = args.root if args.root is not None else default_root()
    split = root / "coco" / "split.json"
    if not split.is_file():
        print(f"no {split}: run tools/play/coco.py on a Label Studio export first")
        return 1
    describe(json.loads(split.read_text(encoding="utf-8")))

    checkpoint = None
    if args.resume is None:
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
    else:
        picked = pick_checkpoint(args.resume, root / "runs")
        if picked is None:
            return 1
        checkpoint, finished = picked
        start = finished + 1
        if start >= args.epochs:
            print(
                f"{checkpoint} already finished epoch {finished}, and --epochs {args.epochs} "
                f"stops at epoch {args.epochs - 1}: nothing to train",
                flush=True,
            )
            return 1
        # The same folder, because rfdetr only keeps metrics.csv and the best score there. The
        # feed already in it, because a new one could differ from what the checkpoint trained on.
        run = checkpoint.parent
        feed = run / "feed"
        dataset = feed if (feed / "train" / ANNOTATIONS).is_file() else root / "coco"
        print(f"resuming from {checkpoint}, last finished epoch {finished}", flush=True)
        print(f"start epoch {start}, training into {run} on {dataset}", flush=True)

    from rfdetr import RFDETRNano  # here, so a missing split fails before torch loads

    describe_device()
    RFDETRNano().train(
        dataset_dir=str(dataset),
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.accum,
        output_dir=str(run),
        num_workers=0,
        checkpoint_interval=args.checkpoint_interval,
        resume=None if checkpoint is None else str(checkpoint),
    )
    print(f"run folder {run}")
    if checkpoint is not None:
        # export takes the newest folder by name, and a resumed folder keeps its old name
        print(f"to export this run, name it: uv run tools/play/export.py --run {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
