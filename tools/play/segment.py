# /// script
# requires-python = ">=3.13"
# dependencies = ["torch", "torchvision", "transformers==5.20.1", "pillow", "numpy>=2.0", "opencv-python-headless"]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""SAM 2.1 boxes for the brawler bars the auto-labeller found.

autolabel.py runs this through uv between its passes. Like train.py it pins its own torch env,
so torch, transformers and PIL are imported only inside load_segmenter:

    uv run tools/play/segment.py --root D:/play-data --prompts prompts.jsonl --out boxes.jsonl

Each prompts line names a frame and the bars found on it. Each output line repeats the frame
and carries one box per bar, in bar order; a frame that is gone gets no boxes. --resume keeps
the lines already in --out and segments only the frames they do not cover.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.play import brawlers  # noqa: E402

DEFAULT_MODEL = "facebook/sam2.1-hiera-large"

# (BGR frame, bars) -> masks (len(bars), k, H, W) bool, scores (len(bars), k)
Segmenter = Callable[[np.ndarray, list[brawlers.Bar]], tuple[np.ndarray, np.ndarray]]


def load_segmenter(model_id: str = DEFAULT_MODEL) -> Segmenter:
    """SAM 2.1 on the GPU when there is one, prompted with every bar on a frame at once."""
    import torch
    from PIL import Image
    from transformers import Sam2Model, Sam2Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM {model_id} on {device}")
    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id).to(device).eval()

    def segment(frame: np.ndarray, bars: list[brawlers.Bar]) -> tuple[np.ndarray, np.ndarray]:
        height, width = frame.shape[:2]
        points, labels, boxes = [], [], []
        for bar in bars:
            bar_points, bar_labels, box = brawlers.sam_prompt(bar, width, height)
            points.append(bar_points)
            labels.append(bar_labels)
            boxes.append(box)
        pil_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(
            images=pil_rgb,
            input_points=[points],
            input_labels=[labels],
            input_boxes=[boxes],
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**inputs, multimask_output=True)
        masks = (
            processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0].numpy()
            > 0
        )
        return masks, out.iou_scores[0].float().cpu().numpy()

    return segment


def _read_frame(path: Path) -> np.ndarray | None:
    """imdecode, not imread: OpenCV cannot read a non-ASCII path on Windows."""
    if not path.exists():
        return None
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def done_lines(out: Path) -> list[str]:
    """The whole lines of an earlier boxes file; a last line cut off by a kill is left out."""
    if not out.exists():
        return []
    kept = []
    for line in out.read_text(encoding="utf-8").splitlines():
        try:
            blob = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(blob, dict) and "file" in blob:
            kept.append(line)
    return kept


def run(root: Path, prompts: Path, out: Path, segment: Segmenter, resume: bool = False) -> dict:
    """Write one boxes line per prompts line, in order, and count what was written.

    With resume, the whole lines already in out are kept and their frames skipped, so a run
    stopped part way picks up where it was. Every line is flushed as it is written.
    """
    summary = {"frames": 0, "boxes": 0, "fallbacks": 0, "missing": 0, "skipped": 0}
    kept = done_lines(out) if resume else []
    done = {json.loads(line)["file"] for line in kept}
    # Rewritten without a cut-off last line, so the new lines append after a whole one.
    out.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    with (
        prompts.open(encoding="utf-8") as src,
        out.open("a", encoding="utf-8") as dst,
    ):
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["file"] in done:
                summary["skipped"] += 1
                continue
            summary["frames"] += 1
            frame = _read_frame(root / row["file"])
            boxes: list[dict] = []
            if frame is None:
                # A frame deleted since the prompts were written still gets its line.
                summary["missing"] += 1
            else:
                bars = [brawlers.Bar.from_json(blob) for blob in row["bars"]]
                # SAM is never asked about a frame with nothing to prompt.
                if bars:
                    masks, scores = segment(frame, bars)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    boxes = [
                        brawlers.choose_box(masks[i], scores[i], bar, hsv)
                        for i, bar in enumerate(bars)
                    ]
            summary["boxes"] += len(boxes)
            summary["fallbacks"] += sum(1 for box in boxes if box["fallback"])
            dst.write(json.dumps({"file": row["file"], "boxes": boxes}) + "\n")
            dst.flush()
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, required=True, help="dataset root")
    ap.add_argument(
        "--prompts", type=Path, required=True, help="prompts JSONL: a frame and its bars per line"
    )
    ap.add_argument(
        "--out", type=Path, required=True, help="boxes JSONL to write, one line per prompts line"
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help="SAM 2.1 model id or local folder")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="keep the lines already in --out and skip their frames",
    )
    args = ap.parse_args(argv)

    if not args.prompts.exists():
        print(f"no prompts at {args.prompts}")
        return 1
    segment = load_segmenter(args.model)
    started = time.perf_counter()
    summary = run(args.root, args.prompts, args.out, segment, resume=args.resume)
    per_frame = (time.perf_counter() - started) / max(1, summary["frames"])
    print(
        f"{summary['frames']} frames, {summary['boxes']} boxes, "
        f"{summary['fallbacks']} fallbacks, {summary['missing']} missing, "
        f"{summary['skipped']} already done"
    )
    print(f"{per_frame:.2f} s per frame; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
