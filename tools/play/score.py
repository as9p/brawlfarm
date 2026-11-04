"""Score an exported model on the held-out test frames against what the farm's templates say.

The templates are the only truth this kit has outside the labelled set, and they are the same
truth the farm loop already trusts, so a model is judged by them before it is ever allowed to
drive: per template class and per frame, the model and the template matcher either agree or
they do not. The frames are the test split of the COCO build the model was trained from, never
the frames it trained on, because a model scored on its own training set is marking its own
homework.

The report ends in one of three verdicts, and the exit code says which:

  PASS (0)          every tap anchor with a template was measured and found often enough,
                    and none was detected where the templates see none.
  FAIL (1)          a tap anchor detected where the templates see none, or beside the one they
                    see: a tap on the wrong pixel spends gems or leaves a queue. Or a tap anchor
                    with at least --min-truth template boxes whose recall is under --min-recall.
  INCONCLUSIVE (2)  nothing here proves the model safe: a tap anchor with too few template
                    boxes to measure, no frame scored, no COCO build, a smoke model, or a scan
                    of whole folders that may hold training frames.

FAIL wins over INCONCLUSIVE: a false positive is real wherever it was seen.

Three tap anchors have no template behind them (skull_star, team_up_panel, event_tab), so
nothing here can tell a good detection of one from a bad one, or count one as missed. Those are
listed as having no label source and decide nothing; they are for the owner to look at.

    uv run python tools/play/score.py
    uv run python tools/play/score.py --cuda --min-truth 50
    uv run python tools/play/score.py --frames D:/shots
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from brawlfarm import settings
from brawlfarm.core import vision
from tools.play import classes, dataset, prelabel, thresholds

SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
EXIT_CODES = {PASS: 0, FAIL: 1, INCONCLUSIVE: 2}
# Below this many template boxes a recall swings on a frame or two, so it proves nothing: the
# same number export.py certifies a threshold by.
MIN_TRUTH = thresholds.MIN_TRUTH
MIN_RECALL = 0.5
# tap anchors a template labels: the only ones whose misses this report can count
LABELLED: frozenset[str] = classes.TAP_ANCHORS & set(classes.TEMPLATE_CLASS.values())
# tap anchors the templates cannot judge: a detection of one is reported, never counted
UNVERIFIABLE: frozenset[str] = classes.TAP_ANCHORS - set(classes.TEMPLATE_CLASS.values())
CUDA = "CUDAExecutionProvider"
SCAN = "a folder scan, not the held-out test split: the model may have trained on these frames"


def load_model(models: Path, *, cuda: bool = False) -> tuple[ort.InferenceSession, dict]:
    """play.onnx and play.json out of a models folder, on the CPU provider unless cuda is asked.

    The providers the session really got are printed rather than assumed, and a cuda request
    that did not get CUDA stops here: onnxruntime falls back to the CPU without a word when the
    CUDA libraries are missing, and a report that took an hour on the CPU while the owner
    thought it was on the GPU is a report nobody trusts twice.
    """
    meta = json.loads((models / "play.json").read_text(encoding="utf-8"))
    providers = ["CPUExecutionProvider"]
    if cuda:
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception as exc:  # an old or CPU-only build: say so and carry on
                print(f"preload_dlls failed ({exc}); trying CUDA anyway")
        providers = [CUDA, "CPUExecutionProvider"]
    session = ort.InferenceSession(str(models / "play.onnx"), providers=providers)
    active = session.get_providers()
    print(f"providers: {', '.join(active)}")
    if cuda and CUDA not in active:
        raise SystemExit(
            f"--cuda was asked for, but onnxruntime gave only {', '.join(active)}. Install the "
            "play-gpu extra, see docs/install.md; if it is installed already, the CPU "
            "onnxruntime package won the install order, and docs/install.md has the fix."
        )
    return session, meta


def detect(sess, meta: dict, frame: np.ndarray) -> list[dict]:
    """One frame through the model: the detections at or above their own class threshold.

    The two outputs are taken by the names play.json recorded, looked up in the session, so an
    export that reordered them cannot silently swap the boxes with the logits. A class the
    export left out of the thresholds has not earned the right to be believed at all, so it
    gets the ceiling and is effectively off.
    """
    height, width = frame.shape[:2]
    names = [output.name for output in sess.get_outputs()]
    outputs = dict(
        zip(names, sess.run(None, {meta["input"]["name"]: preprocessed(meta, frame)}), strict=True)
    )
    found = thresholds.decode(
        outputs[meta["outputs"]["boxes"]], outputs[meta["outputs"]["logits"]], width, height
    )

    kept = []
    for box in found:
        if not 0 <= box["class_index"] < len(meta["classes"]):
            continue
        name = meta["classes"][box["class_index"]]
        if box["score"] < meta["thresholds"].get(name, thresholds.CEILING):
            continue
        kept.append({**box, "class": name})
    return kept


def preprocessed(meta: dict, frame: np.ndarray) -> np.ndarray:
    """The model's input tensor, at whatever size play.json says the export used."""
    return thresholds.preprocess(frame, meta["input"]["size"])


def template_truth(frame: np.ndarray, *, find=vision.find_with_score) -> dict[str, dict | None]:
    """What the templates say about a frame: a box per template class, or None for that class.

    Every template class is a key even when nothing matched, because an absent box is the half
    of the truth that catches a false positive.
    """
    boxes = {box["class"]: box for box in prelabel.boxes_for(frame, find=find)}
    return {cls: boxes.get(cls) for cls in classes.TEMPLATE_CLASS.values()}


def compare(detections: list[dict], truth: dict[str, dict | None], iou_min: float = 0.5) -> dict:
    """One verdict per template class for one frame: tp, tp+extra, fn, fp or tn.

    A detection of the right class in the wrong place is an fn and not also an fp: the template
    box proves there is something of that class on the frame, so the model is wrong about where
    it is, not about whether it is there. Finding the control and also pointing somewhere else
    is tp+extra, which counts as both: the extra box is a tappable pixel the templates do not
    see, which is the never-tap failure however right the other box was.
    """
    verdicts = {}
    for cls, real in truth.items():
        here = [box for box in detections if box["class"] == cls]
        if real is None:
            verdicts[cls] = "fp" if here else "tn"
            continue
        box = (real["x"], real["y"], real["w"], real["h"])
        hits = [
            thresholds.iou(box, (det["x"], det["y"], det["w"], det["h"])) >= iou_min for det in here
        ]
        if not any(hits):
            verdicts[cls] = "fn"
        else:
            # Two detections on the one template box are one control seen twice, not an extra.
            verdicts[cls] = "tp" if all(hits) else "tp+extra"
    return verdicts


def report(
    rows: list[dict],
    *,
    min_truth: int = MIN_TRUTH,
    min_recall: float = MIN_RECALL,
    caveats: list[str] | None = None,
) -> dict:
    """The verdicts of every frame summed per class and per tap anchor, then the one verdict.

    A precision or a recall with nothing under it is 0.0, the way export.py reports one: a class
    the model never detected has not shown anything, and printing 1.0 there would read as perfect.
    A tp+extra is one of each: the control was found, and a second box was invented beside it.

    The truth of a tap anchor is its template boxes, tp plus fn. Fewer than min_truth of them
    and its recall proves nothing, which makes the report INCONCLUSIVE, as does no frame at all
    or any caveat the caller passes in (a smoke model, frames that may be training ones). A
    never-tap false positive, or a measured recall under min_recall, is a FAIL however little
    else could be measured. The reasons say why the verdict is not PASS.
    """
    totals: dict[str, Counter[str]] = {}
    for row in rows:
        for cls, verdict in row.items():
            totals.setdefault(cls, Counter())[verdict] += 1

    per_class = {}
    for cls, count in totals.items():
        extra = count["tp+extra"]
        tp, fp, fn, tn = count["tp"] + extra, count["fp"] + extra, count["fn"], count["tn"]
        per_class[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        }
    bad = sum(
        count["fp"] + count["tp+extra"]
        for cls, count in totals.items()
        if cls in classes.TAP_ANCHORS
    )

    failed: list[str] = []
    unsure = list(caveats or [])
    if bad:
        failed.append(f"{bad} tap anchor detection(s) where the templates see none")
    if not rows:
        unsure.append("no frame was scored, so nothing is proven")
    anchors = {}
    for cls in sorted(LABELLED):
        count = per_class.get(cls, {"tp": 0, "fp": 0, "fn": 0, "recall": 0.0})
        truth = count["tp"] + count["fn"]
        anchors[cls] = {
            "truth": truth,
            "tp": count["tp"],
            "fn": count["fn"],
            "fp": count["fp"],
            "recall": count["recall"],
        }
        if truth < min_truth:
            unsure.append(f"{cls} has {truth} template box(es), fewer than {min_truth} to measure")
        elif count["tp"] / truth < min_recall:
            failed.append(f"{cls} recall {count['recall']} on {truth} is under {min_recall}")
    verdict = FAIL if failed else INCONCLUSIVE if unsure else PASS
    return {
        "verdict": verdict,
        "pass": verdict == PASS,
        "fail_reasons": failed,
        "inconclusive_reasons": unsure,
        "classes": per_class,
        "tap_anchors": anchors,
        "no_label_source": sorted(UNVERIFIABLE),
        "tap_anchor_false_positives": bad,
    }


def held_out(root: Path, meta: dict) -> tuple[list[tuple[str, Path]], list[str]]:
    """The test split of the COCO build as (name, path) pairs, and why it cannot prove a pass.

    coco.py copies every frame of a split byte for byte, under its own file name, into
    `coco/<split>/` and lists it in that folder's `_annotations.coco.json`, so the copy in
    `coco/test/` is the frame from `frames/<source>/` that the split held out. The build has to
    be the one the model trained on, which the training set hash in split.json and play.json
    proves: a later build can have moved a training frame into the test folder.
    """
    coco = root / "coco"
    listing = coco / "test" / "_annotations.coco.json"
    if not (coco / "split.json").is_file() or not listing.is_file():
        return [], ["no COCO build under the dataset root: run tools/play/coco.py first"]
    split = json.loads((coco / "split.json").read_text(encoding="utf-8"))
    caveats = []
    if split.get("training_set_hash") != meta["training_set_hash"]:
        caveats.append(
            "coco/split.json has another training set hash than the model, so its test split "
            "may hold frames the model trained on"
        )
    images = json.loads(listing.read_text(encoding="utf-8")).get("images", [])
    names = sorted({image["file_name"] for image in images})
    return [(name, coco / "test" / name) for name in names], caveats


def scanned(folders: list[Path]) -> tuple[list[tuple[str, Path]], list[str]]:
    """Every picture under the folders as (name, path) pairs, and why that cannot prove a pass."""
    caveats = [SCAN]
    folders = [folder for folder in folders if folder.is_dir()]
    if not folders:
        caveats.append("no frame folder to scan: pass --frames")
    pictures = [
        # The name is relative to its folder: an absolute path carries the user profile.
        (path.relative_to(folder).as_posix(), path)
        for folder in folders
        for path in _pictures(folder)
    ]
    return pictures, caveats


def truth_bar(text: str) -> int:
    """An argparse type for a truth minimum: a whole number, at least 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} is below 1")
    return value


def recall_bar(text: str) -> float:
    """An argparse type for a recall bar: a number in (0, 1]."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError(f"{value} is outside (0, 1]")
    return value


def _read_frame(path: Path) -> np.ndarray | None:
    """imdecode, not imread: OpenCV cannot read a non-ASCII path on Windows."""
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def _pictures(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in SUFFIXES)


def main(argv: list[str] | None = None, *, find=vision.find_with_score) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=None, help="dataset root")
    ap.add_argument("--models", type=Path, default=None, help="folder holding play.onnx")
    ap.add_argument(
        "--min-truth",
        type=truth_bar,
        default=MIN_TRUTH,
        help="template boxes a tap anchor needs in the scored frames before its recall counts "
        f"(default {MIN_TRUTH})",
    )
    ap.add_argument(
        "--min-recall",
        type=recall_bar,
        default=MIN_RECALL,
        help=f"recall a measured tap anchor must reach (default {MIN_RECALL})",
    )
    ap.add_argument(
        "--scan-folders",
        action="store_true",
        help="score every picture under the dataset's frames and the recordings instead of the "
        "held-out test split; a scan is never a PASS",
    )
    ap.add_argument(
        "--frames",
        action="append",
        default=None,
        metavar="FOLDER",
        help="scan the pictures under this folder instead; repeat for more (implies "
        "--scan-folders)",
    )
    ap.add_argument(
        "--cuda", action="store_true", help="run on the CUDA provider, and stop if it is missing"
    )
    args = ap.parse_args(argv)

    root = args.root if args.root is not None else dataset.default_root()
    models = args.models if args.models is not None else root / "models"
    if not (models / "play.onnx").is_file() or not (models / "play.json").is_file():
        print(f"no play.onnx and play.json under {models}: export a model first")
        return 1

    sess, meta = load_model(models, cuda=args.cuda)
    scan = args.scan_folders or args.frames is not None
    if args.frames is not None:
        pictures, caveats = scanned([Path(name) for name in args.frames])
    elif scan:
        recordings = settings.default_home() / "calibration" / "recordings"
        pictures, caveats = scanned([root / "frames", recordings])
    else:
        pictures, caveats = held_out(root, meta)
    if meta["smoke"]:
        caveats.append("the model was trained on a smoke set, so the report means nothing")

    rows: list[dict] = []
    failures: list[dict] = []
    unverified: list[dict] = []
    scored = skipped = 0
    for name, path in pictures:
        frame = _read_frame(path) if path.is_file() else None
        if frame is None:
            skipped += 1
            continue
        height, width = frame.shape[:2]
        if (width, height) != (dataset.FRAME_W, dataset.FRAME_H):
            skipped += 1
            continue
        scored += 1
        found = detect(sess, meta, frame)
        verdicts = compare(found, template_truth(frame, find=find))
        rows.append(verdicts)
        failures.extend(
            {"file": name, "class": cls}
            for cls, verdict in sorted(verdicts.items())
            if verdict in ("fp", "tp+extra") and cls in classes.TAP_ANCHORS
        )
        unverified.extend(
            {"file": name, "class": box["class"]} for box in found if box["class"] in UNVERIFIABLE
        )

    summary = report(rows, min_truth=args.min_truth, min_recall=args.min_recall, caveats=caveats)
    summary |= {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "training_set_hash": meta["training_set_hash"],
        "smoke": meta["smoke"],
        "held_out": not scan,
        "min_truth": args.min_truth,
        "min_recall": args.min_recall,
        "frames": {"scored": scored, "skipped": skipped},
        "tap_anchor_failures": failures,
        "unverified_tap_anchor_detections": unverified,
    }
    models.mkdir(parents=True, exist_ok=True)
    (models / "score.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    where = "frames scanned" if scan else "held-out test frames scored"
    print(f"{scored} {where}, {skipped} skipped (unreadable or not 1600 x 900)")
    if scored:
        head = f"{'class':<18} {'truth':>5} {'tp':>5} {'fp':>5} {'fn':>5} {'tn':>5}"
        print(f"  {head} {'prec':>6} {'rec':>6}")
        for cls, count in summary["classes"].items():
            anchor = " tap anchor" if cls in classes.TAP_ANCHORS else ""
            print(
                f"  {cls:<18} {count['tp'] + count['fn']:>5} {count['tp']:>5} {count['fp']:>5} "
                f"{count['fn']:>5} {count['tn']:>5} {count['precision']:>6.3f} "
                f"{count['recall']:>6.3f}{anchor}"
            )
    for bad in failures:
        print(f"  never-tap failure: {bad['class']} on {bad['file']}, the templates see none there")
    print(
        f"no label source for {', '.join(summary['no_label_source'])}: these tap anchors have no "
        "template, so they are not measured and do not change the verdict"
    )
    if unverified:
        print(
            f"{len(unverified)} detections of them, which nothing here can call right or wrong. "
            "Check them by hand."
        )
        for box in unverified:
            print(f"  unverified: {box['class']} on {box['file']}")
    print(f"wrote {models / 'score.json'}")
    if summary["verdict"] == PASS:
        print(
            "PASS: every tap anchor with a template was found often enough, and none was "
            "detected where the templates see none"
        )
    else:
        print(summary["verdict"])
    for reason in summary["fail_reasons"]:
        print(f"  fail: {reason}")
    for reason in summary["inconclusive_reasons"]:
        print(f"  inconclusive: {reason}")
    return EXIT_CODES[summary["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
