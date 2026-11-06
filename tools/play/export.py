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
"""Export a trained run to ONNX and score a confidence threshold for every class.

Runs in the same isolated environment as train.py, and like it takes nothing from the project
but tools/play/thresholds.py and tools/play/classes.py:

    uv run tools/play/export.py
    uv run tools/play/export.py --root D:/play-data --run 20260920-231500
    uv run tools/play/export.py --min-precision-other 0.85
    uv run tools/play/export.py --best-f1-other

The export is checked before it is believed. The library's own predict and a hand-rolled
onnxruntime pass over the same validation images must agree on nearly every box, which is what
catches a shifted column in the logits: an off-by-one there turns every class into its
neighbour, and every check downstream still looks fine. Only then does each class get the
lowest threshold that still buys its precision target on the validation set, and only then
are models/play.onnx and models/play.json written. A tap anchor's target is always
thresholds.MIN_PRECISION; every other class gets --min-precision-other, which defaults to
thresholds.MIN_PRECISION_OTHER. --best-f1-other replaces that target with the validation
threshold of best F1, on those classes only: a tap anchor keeps thresholds.MIN_PRECISION.

play.json also records, per class, how many validation truth boxes the threshold was drawn from
and whether it is certified: at least thresholds.MIN_TRUTH of them, and a threshold that really
reached its target rather than the 0.95 fallback. Every class that is not certified is printed.
Its `rule` says what set the threshold: precision, best_f1, or ceiling for that fallback. A
best_f1 class has no precision target, so its min_precision is null and it records its F1.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.play import thresholds  # noqa: E402

RFDETR_VERSION = "1.10.1"
INPUT_SIZE = 384
CHECKPOINT = "checkpoint_best_total.pth"
# the library's own detections are the reference the ONNX pass has to reproduce
PREDICT_THRESHOLD = 0.3
AGREE_IOU = 0.9
AGREE_MIN = 0.95


def default_root() -> Path:
    """`<home>/datasets/play`, worked out by hand: this environment has no brawlfarm in it."""
    home = os.environ.get("BRAWLFARM_HOME")
    if home:
        return Path(home) / "datasets" / "play"
    local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(local) / "brawlfarm" / "datasets" / "play"


def newest_run(runs: Path) -> Path | None:
    """The last run folder by name, which is a UTC stamp, so the newest one."""
    if not runs.is_dir():
        return None
    folders = sorted(folder for folder in runs.iterdir() if folder.is_dir())
    return folders[-1] if folders else None


def read_coco(path: Path) -> tuple[list[str], dict[str, list[dict]]]:
    """A COCO file as the class names in id order and the truth boxes per image file name.

    Class indices are category ids minus one, the numbering the model itself reports, so a
    detection and a truth box can be compared without another table in between.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    names = [cat["name"] for cat in sorted(data["categories"], key=lambda cat: cat["id"])]
    files = {image["id"]: image["file_name"] for image in data["images"]}
    truth: dict[str, list[dict]] = {name: [] for name in files.values()}
    for box in data["annotations"]:
        x, y, w, h = box["bbox"]
        truth[files[box["image_id"]]].append(
            {"class_index": box["category_id"] - 1, "x": x, "y": y, "w": w, "h": h}
        )
    return names, truth


def exported_onnx(run: Path, returned: object) -> Path | None:
    """Where the export landed: what the library returned, else the one ONNX file in the run."""
    if isinstance(returned, (str, Path)) and Path(returned).is_file():
        return Path(returned)
    found = sorted(run.glob("*.onnx"))
    return found[0] if len(found) == 1 else None


def library_boxes(detections: object) -> list[dict]:
    """A supervision Detections turned into the same box dictionaries decode produces.

    An empty Detections carries no confidence and no class id at all, which is a screen with
    nothing on it rather than a problem, so it becomes an empty list.
    """
    if detections.confidence is None or detections.class_id is None:
        return []
    boxes = []
    for (x1, y1, x2, y2), score, class_id in zip(
        detections.xyxy, detections.confidence, detections.class_id, strict=True
    ):
        boxes.append(
            {
                "class_index": int(class_id),
                "score": float(score),
                "x": float(x1),
                "y": float(y1),
                "w": float(x2 - x1),
                "h": float(y2 - y1),
            }
        )
    return boxes


def agrees(box: dict, found: list[dict]) -> bool:
    """Whether some ONNX box has the same class and covers the library's box almost exactly."""
    reference = (box["x"], box["y"], box["w"], box["h"])
    return any(
        other["class_index"] == box["class_index"]
        and thresholds.iou(reference, (other["x"], other["y"], other["w"], other["h"])) >= AGREE_IOU
        for other in found
    )


def precision(text: str) -> float:
    """An argparse type for a precision target: a number in (0, 1]."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError(f"{value} is outside (0, 1]")
    return value


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=None, help="dataset root")
    ap.add_argument("--run", default=None, help="run folder, or its name under runs/")
    other = ap.add_mutually_exclusive_group()
    other.add_argument(
        "--min-precision-other",
        type=precision,
        default=thresholds.MIN_PRECISION_OTHER,
        help="precision target for every class that is not a tap anchor "
        f"(default {thresholds.MIN_PRECISION_OTHER}); tap anchors always use "
        f"{thresholds.MIN_PRECISION}",
    )
    other.add_argument(
        "--best-f1-other",
        action="store_true",
        help="pick every class that is not a tap anchor by best F1 on the validation split "
        f"instead of a precision target; tap anchors still use {thresholds.MIN_PRECISION}",
    )
    return ap


def pick_thresholds(
    names: list[str],
    scores: dict[int, list[float]],
    hits: dict[int, list[bool]],
    truth_counts: dict[int, int],
    min_precision_other: float,
    *,
    best_f1: bool = False,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Every class's threshold, and the validation numbers play.json records next to it.

    A threshold is certified only when the validation split held at least thresholds.MIN_TRUTH
    truth boxes of its class and the threshold really reached its precision target. The CEILING
    that thresholds.pick falls back to when no threshold does is still written, so the class
    stays effectively off, but it is never certified, however many boxes there were. Every class
    that is not certified is printed with its reasons.

    With `best_f1`, a class that is not a tap anchor takes pick_best_f1 instead. It has no target
    to reach, so it is certified when it has the truth boxes and a true positive at its threshold;
    without them it gets the same CEILING, certified false. A tap anchor never takes this branch.
    """
    picked: dict[str, float] = {}
    validation: dict[str, dict] = {}
    uncertified: list[tuple[str, list[str]]] = []
    for index, name in enumerate(names):
        truth = truth_counts[index]
        rule = thresholds.rule_for(name, best_f1=best_f1)
        target = None
        if rule == thresholds.RULE_BEST_F1:
            threshold = thresholds.pick_best_f1(name, scores[index], hits[index], truth)
        else:
            target = thresholds.target_for(name, other=min_precision_other)
            threshold = thresholds.pick(scores[index], hits[index], min_precision=target)
        above = [
            hit for score, hit in zip(scores[index], hits[index], strict=True) if score >= threshold
        ]
        reasons = []
        if truth < thresholds.MIN_TRUTH:
            reasons.append(f"{truth} validation truth boxes, fewer than {thresholds.MIN_TRUTH}")
        if target is None:
            fell_back = truth < thresholds.MIN_TRUTH or not any(above)
            if not any(above):
                reasons.append(f"{threshold:.2f} is the fallback, no true positive above it")
        else:
            # the same test pick ran: nothing above the threshold reached the target
            fell_back = not above or sum(above) / len(above) < target
            if fell_back:
                reasons.append(
                    f"{threshold:.2f} is the fallback, no threshold reached its {target:.2f} "
                    "precision target"
                )
        if fell_back:
            rule = thresholds.RULE_CEILING
        picked[name] = threshold
        validation[name] = {
            "truth": truth,
            "certified": not reasons,
            "detections": len(above),
            "precision_at_threshold": round(sum(above) / len(above), 4) if above else 0.0,
            "min_precision": target,
            "rule": rule,
        }
        if rule == thresholds.RULE_BEST_F1:
            validation[name]["f1_at_threshold"] = round(2 * sum(above) / (len(above) + truth), 4)
        if reasons:
            uncertified.append((name, reasons))
        basis = f"target {target:.2f}" if target is not None else f"rule {rule}"
        print(f"  {name} threshold {threshold:.2f} {basis} truth {truth} detections {len(above)}")
    if uncertified:
        print("")
        for name, reasons in uncertified:
            print(f"WARNING: {name} is not certified: {'; '.join(reasons)}")
        print(
            "WARNING: an uncertified threshold is a guess, not a measurement, so the model is not"
        )
        print(
            "WARNING: proven on those classes; play.json records them as certified false. Collect"
        )
        print("WARNING: more frames that show them.")
        print("")
    return picked, validation


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args(argv)

    root = args.root if args.root is not None else default_root()
    if args.run is None:
        run = newest_run(root / "runs")
        if run is None:
            print(f"no run under {root / 'runs'}: train a model first")
            return 1
    else:
        run = Path(args.run)
        if not run.is_dir():
            run = root / "runs" / args.run
    checkpoint = run / CHECKPOINT
    if not checkpoint.is_file():
        print(f"no {checkpoint}: that run has no best checkpoint")
        return 1

    split_file = root / "coco" / "split.json"
    valid = root / "coco" / "valid"
    train_annotations = root / "coco" / "train" / "_annotations.coco.json"
    for needed in (split_file, train_annotations, valid / "_annotations.coco.json"):
        if not needed.is_file():
            print(f"no {needed}: run tools/play/coco.py on a Label Studio export first")
            return 1
    split = json.loads(split_file.read_text(encoding="utf-8"))
    names, _train_truth = read_coco(train_annotations)
    _valid_names, truth = read_coco(valid / "_annotations.coco.json")

    import cv2
    import onnxruntime
    from rfdetr import RFDETRNano

    print(f"loading {checkpoint}")
    model = RFDETRNano(pretrain_weights=str(checkpoint))
    # verbose=False: the default prints the whole ONNX graph over the run's own output.
    onnx_file = exported_onnx(run, model.export(output_dir=str(run), verbose=False))
    if onnx_file is None:
        print(f"the export left no single ONNX file in {run}")
        return 1
    print(f"exported {onnx_file}")

    session = onnxruntime.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    if not {"dets", "labels"} <= set(output_names):
        print(f"unexpected ONNX outputs {output_names}: this kit decodes dets and labels")
        return 1

    scores: dict[int, list[float]] = defaultdict(list)
    hits: dict[int, list[bool]] = defaultdict(list)
    truth_counts: dict[int, int] = defaultdict(int)
    checked = matched = 0
    images = sorted(path for path in valid.iterdir() if path.name in truth)
    if not images:
        print(f"no validation image under {valid}")
        return 1
    for path in images:
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"cannot read {path}")
            return 1
        height, width = bgr.shape[:2]
        outputs = dict(
            zip(
                output_names,
                session.run(None, {input_name: thresholds.preprocess(bgr)}),
                strict=True,
            )
        )
        found = thresholds.decode(outputs["dets"], outputs["labels"], width, height)

        reference = library_boxes(
            model.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), threshold=PREDICT_THRESHOLD)
        )
        checked += len(reference)
        matched += sum(1 for box in reference if agrees(box, found))

        here = truth[path.name]
        for box in here:
            truth_counts[box["class_index"]] += 1
        for box, hit in zip(found, thresholds.match_detections(found, here), strict=True):
            scores[box["class_index"]].append(box["score"])
            hits[box["class_index"]].append(hit)

    share = matched / checked if checked else 1.0
    print(f"{matched} of {checked} library detections reproduced by the ONNX pass ({share:.1%})")
    if share < AGREE_MIN:
        print(f"the ONNX output does not match the library: under {AGREE_MIN:.0%} agreement.")
        print("The column mapping in thresholds.decode is wrong for this rfdetr build.")
        return 1

    picked, validation = pick_thresholds(
        names, scores, hits, truth_counts, args.min_precision_other, best_f1=args.best_f1_other
    )

    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_file, models / "play.onnx")
    (models / "play.json").write_text(
        json.dumps(
            {
                "model": "rfdetr-nano",
                "rfdetr": RFDETR_VERSION,
                "input": {
                    "name": input_name,
                    "size": INPUT_SIZE,
                    "layout": "NCHW",
                    "color": "RGB",
                    "mean": list(thresholds.MEAN),
                    "std": list(thresholds.STD),
                },
                "outputs": {
                    "boxes": "dets",
                    "logits": "labels",
                    "box_format": "cxcywh_normalised",
                },
                "classes": names,
                "thresholds": picked,
                "training_set_hash": split["training_set_hash"],
                "smoke": split["smoke"],
                "validation": validation,
                "exported": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"wrote {models / 'play.onnx'} and {models / 'play.json'}")
    if split["smoke"]:
        print("WARNING: trained on a smoke set. These thresholds mean nothing; do not ship it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
