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
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


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
    print(f"{len(summary['sources'])} sources: {counts}")
    # Every class, zeros included: a class with no boxes trains nothing and should be seen.
    for name, count in sorted(summary["boxes"].items()):
        print(f"  {name} {count}")
    print(f"training set hash {summary['training_set_hash'][:16]}")
    if summary["smoke"]:
        print("")
        print("WARNING: this is a smoke set. The same frames are in all three splits, so the")
        print("WARNING: validation score is meaningless and the model must not ship.")
        print("")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=None, help="dataset root")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
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

    from rfdetr import RFDETRNano  # here, so a missing split fails before torch loads

    RFDETRNano().train(
        dataset_dir=str(root / "coco"),
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
