"""Auto-labels for the play detector: every frame in the index, no human step.

uv run python tools/play/autolabel.py
uv run python tools/play/autolabel.py --root D:/play-data --source yt-AAAAAAAAAAA --no-sam
uv run python tools/play/autolabel.py --resume
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from brawlfarm.core import jsonio
from tools.play import brawlers, classes, dataset, prelabel, segment, terrain, zones

MODEL_VERSION = "autolabel-v1"
OUT_NAME = "autolabel.json"
SAM_MODEL = "facebook/sam2.1-hiera-large"
# The first pass's checkpoint and the SAM output, both under <root>/autolabel/.
PASS1_NAME = "pass1.jsonl"
BOXES_NAME = "boxes.jsonl"


class FrameLabels(NamedTuple):
    """What the first pass found on one frame, before SAM turns its bars into boxes."""

    template: list[dict]
    terrain: list[dict]
    bars: list[brawlers.Bar]
    in_match: bool


def in_match(row: dict, hsv: np.ndarray, template: list[dict]) -> bool:
    """Whether a frame is mid-match, so the brawler and terrain passes should run on it.

    Every template is a menu button or card, so any template box means a menu. Past that, the
    index's HUD flag decides when the frame has one, and the colour spread decides when it does
    not.
    """
    if template:
        return False
    if row.get("hud") is not None:
        return bool(row["hud"])
    return terrain.is_game_frame(hsv)


def merge(template: list[dict], auto: list[dict], zone: np.ndarray) -> list[dict]:
    """The frame's final boxes: the template boxes first, then the auto boxes that survive.

    Auto boxes mostly inside a zone are dropped and the rest are clipped to the frame, skipping
    any that clip to nothing. Template boxes are never zone-filtered: menu buttons sit where the
    HUD does, so the HUD zones would drop exactly the boxes the templates exist to find.
    """
    kept: list[dict] = []
    for box in zones.drop_in_zones(auto, zone):
        left = max(0, box["x"])
        top = max(0, box["y"])
        right = min(dataset.FRAME_W, box["x"] + box["w"])
        bottom = min(dataset.FRAME_H, box["y"] + box["h"])
        if right <= left or bottom <= top:
            continue
        kept.append({**box, "x": left, "y": top, "w": right - left, "h": bottom - top})
    return list(template) + kept


def ocr_engine() -> Callable:
    """RapidOCR, imported here so loading this module never loads the OCR models."""
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def label_frame(bgr: np.ndarray, row: dict, zone: np.ndarray, ocr) -> FrameLabels:
    """The template boxes, and when the frame is mid-match its terrain boxes and brawler bars."""
    template = prelabel.boxes_for(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if not in_match(row, hsv, template):
        return FrameLabels(template, [], [], False)
    result, _ = ocr(bgr)
    anchors, solo = brawlers.anchors_from_ocr(result, zone)
    return FrameLabels(
        template,
        terrain.terrain_boxes(hsv, zone),
        brawlers.classify(bgr, hsv, anchors, solo),
        True,
    )


def fixed_boxes(bars: list[brawlers.Bar]) -> list[dict]:
    """The fixed-size box under every bar, for a run without the SAM pass."""
    return [brawlers.fallback_box(bar, dataset.FRAME_W, dataset.FRAME_H) for bar in bars]


def checkpoint_line(file: str, labels: FrameLabels) -> str:
    """One pass1.jsonl line: everything the later passes need for a frame, and no OCR text."""
    return json.dumps(
        {
            "version": MODEL_VERSION,
            "file": file,
            "template": labels.template,
            "terrain": labels.terrain,
            "bars": [bar.to_json() for bar in labels.bars],
            "in_match": labels.in_match,
        }
    )


def load_checkpoint(path: Path) -> tuple[dict[str, FrameLabels], int]:
    """The first-pass labels a stopped run left behind, by file, and how many were stale.

    A last line cut off by a kill and lines from another MODEL_VERSION are dropped, and the
    file is rewritten without them so new lines append after a whole one.
    """
    done: dict[str, FrameLabels] = {}
    kept: list[str] = []
    stale = 0
    for line in segment.done_lines(path):
        blob = json.loads(line)
        if blob.get("version") != MODEL_VERSION:
            stale += 1
            continue
        done[blob["file"]] = FrameLabels(
            blob["template"],
            blob["terrain"],
            [brawlers.Bar.from_json(bar) for bar in blob["bars"]],
            blob["in_match"],
        )
        kept.append(line)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    tmp.replace(path)
    return done, stale


def run_segment(root: Path, prompts: Path, out: Path, model: str, resume: bool = False) -> None:
    """Run segment.py in its own torch env; a failed run raises CalledProcessError.

    With resume, segment.py keeps the boxes already in out and segments only the rest.
    """
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is not on PATH; the SAM pass runs through uv run")
    args = [
        uv,
        "run",
        str(Path(__file__).with_name("segment.py")),
        "--root",
        str(root),
        "--prompts",
        str(prompts),
        "--out",
        str(out),
        "--model",
        model,
    ]
    if resume:
        args.append("--resume")
    subprocess.run(args, check=True)


def _zone(row: dict, exclusions: dict[str, list[zones.Rect]]) -> np.ndarray:
    return zones.zone_mask(row.get("pad") or (0, 0, 0, 0), exclusions.get(row["source"], []))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=None, help="dataset root")
    ap.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="NAME",
        help="auto-label only this source; repeat for more",
    )
    ap.add_argument(
        "--exclusions",
        type=Path,
        default=None,
        help="per-source extra zones (default: <root>/exclusions.json)",
    )
    ap.add_argument(
        "--no-sam", action="store_true", help="skip SAM; every brawler gets the fixed-size box"
    )
    ap.add_argument("--model", default=SAM_MODEL, help="SAM 2.1 model id or local folder")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue a stopped run from its checkpoint instead of starting over",
    )
    args = ap.parse_args(argv)

    root = args.root if args.root is not None else dataset.default_root()
    wanted = {dataset.check_source(name) for name in args.source} if args.source else None
    index = root / "index.jsonl"
    if not index.exists():
        print(f"no index at {index}; extract some frames first")
        return 1

    exclusions = zones.load_exclusions(
        args.exclusions if args.exclusions is not None else root / zones.EXCLUSIONS_FILE
    )
    ocr = ocr_engine()

    # A fresh run drops what an earlier run left; --resume picks it up instead.
    work = root / "autolabel"
    work.mkdir(parents=True, exist_ok=True)
    pass1 = work / PASS1_NAME
    out = work / BOXES_NAME
    done: dict[str, FrameLabels] = {}
    if not args.resume:
        pass1.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
    elif pass1.exists():
        done, stale = load_checkpoint(pass1)
        if stale:
            # Frames relabelled by newer code may carry other bars, so SAM starts over too.
            print(f"{stale} checkpoint lines are from another auto-labeller; redoing them")
            out.unlink(missing_ok=True)

    # First pass: templates, terrain and bars per frame. Frames and zone masks are dropped as
    # soon as their frame is done, so memory grows with the labels, not the frames. Each
    # labelled frame is appended to the checkpoint and flushed, so a stop keeps it.
    kept: list[tuple[dict, FrameLabels]] = []
    missing = 0
    failed = 0
    resumed = 0
    with pass1.open("a", encoding="utf-8") as checkpoint:
        for row in dataset.index_rows(root):
            if wanted is not None and row["source"] not in wanted:
                continue
            if row["file"] in done and (root / row["file"]).exists():
                kept.append((row, done[row["file"]]))
                resumed += 1
                continue
            bgr = prelabel._read_frame(root / row["file"])
            if bgr is None:
                # The index outlives the files, so a deleted frame is reported, not fatal.
                missing += 1
                continue
            try:
                labels = label_frame(bgr, row, _zone(row, exclusions), ocr)
            except Exception as exc:  # one bad frame must not cost the whole run
                print(f"{row['file']}: {type(exc).__name__}: {exc}")
                failed += 1
                continue
            kept.append((row, labels))
            checkpoint.write(checkpoint_line(row["file"], labels) + "\n")
            checkpoint.flush()

    # Second pass: a box per bar, from SAM or the fixed size. SAM never runs with nothing to do.
    with_bars = [(row, labels) for row, labels in kept if labels.bars]
    boxes: dict[str, list[dict]] = {}
    sam_done = 0
    if args.no_sam:
        boxes = {row["file"]: fixed_boxes(labels.bars) for row, labels in with_bars}
    elif with_bars:
        prompts = work / "prompts.jsonl"
        lines = [
            json.dumps({"file": row["file"], "bars": [bar.to_json() for bar in labels.bars]})
            for row, labels in with_bars
        ]
        prompts.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        if args.resume:
            prompted = {row["file"] for row, _ in with_bars}
            sam_done = sum(
                1 for line in segment.done_lines(out) if json.loads(line)["file"] in prompted
            )
        try:
            run_segment(root, prompts, out, args.model, resume=args.resume)
        except subprocess.CalledProcessError:
            print("the SAM pass failed; nothing written. Rerun it, or pass --no-sam")
            return 1
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                blob = json.loads(line)
                boxes[blob["file"]] = blob["boxes"]
        lost = sum(1 for row, _ in with_bars if row["file"] not in boxes)
        if lost:
            print(f"{lost} frames with bars are missing from the SAM output")

    # Third pass: the merged boxes and the task per frame.
    tasks = []
    per_class: Counter[str] = Counter()
    fixed = 0
    for row, labels in kept:
        brawler = brawlers.suppress(boxes.get(row["file"], []))
        merged = merge(labels.template, labels.terrain + brawler, _zone(row, exclusions))
        per_class.update(box["class"] for box in merged)
        fixed += sum(1 for box in merged if box.get("fallback"))
        tasks.append(prelabel.task_for(row["file"], merged, model_version=MODEL_VERSION))

    # Written only now that every pass is done, and atomically, so a failure leaves no half file.
    out_dir = root / "labelstudio"
    path = out_dir / OUT_NAME
    jsonio.atomic_write_json(path, tasks, indent=1)
    shutil.copyfile(prelabel.CONFIG_XML, out_dir / "config.xml")

    matched = sum(1 for _, labels in kept if labels.in_match)
    print(
        f"{len(tasks)} frames, {matched} in a match, {missing} missing, {failed} failed, "
        f"{fixed} fixed brawler boxes"
    )
    print(f"{resumed} frames from the checkpoint, {sam_done} SAM frames already done")
    for cls in classes.CLASSES:
        if per_class[cls]:
            print(f"  {cls} {per_class[cls]}")
    print(f"wrote {path} and {out_dir / 'config.xml'}")
    print(f"next: uv run python tools/play/coco.py {path} --root {root} --from-predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
