"""Per-class confidence thresholds, and the ONNX in and out the exporter shares with them.

A detector that ships to the farm loop is allowed to miss things and is not allowed to invent
them: a false positive on a tap anchor is a tap on the wrong pixel. So a threshold is picked per
class as the lowest one that still buys its precision target on the validation set, never a
single number for the whole model, and it never drops below FLOOR however clean a class looks.
A tap anchor (classes.TAP_ANCHORS) is held to MIN_PRECISION; every other class is only ever
watched, never tapped, and is held to the looser MIN_PRECISION_OTHER.

A precision target can price a watched class out of being seen at all: it lands so high that
nearly every real box is dropped. So the exporter can instead be asked for best F1 on the classes
that are only watched (pick_best_f1). rule_for is the one place that decides, and it never hands
best F1 to a tap anchor, which stays on MIN_PRECISION whatever was asked for.

This module is imported by train.py and export.py, which run in their own isolated environment
with rfdetr and torch in it and no brawlfarm. It therefore imports numpy and the standard
library only, pulls cv2 in inside preprocess where a resize is unavoidable, and pulls the
standard-library-only tools.play.classes in inside target_for.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# a tap anchor's false positive is a tap on the wrong pixel, which can spend gems
MIN_PRECISION = 0.98
# a class the controller only watches: a false positive costs a wrong guess, not a tap
MIN_PRECISION_OTHER = 0.90
FLOOR, CEILING = 0.30, 0.95
# what play.json records as the rule behind a threshold; RULE_CEILING is the fallback, whichever
# rule was tried first
RULE_PRECISION, RULE_BEST_F1, RULE_CEILING = "precision", "best_f1", "ceiling"
# Below this many validation truth boxes a class's precision swings on a box or two, so export.py
# does not certify its threshold, and coco.py warns about a valid or test split that thin.
MIN_TRUTH = 30
# the ImageNet statistics RF-DETR normalises with, in RGB order
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two boxes given as (x, y, w, h) with x, y the top left."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - overlap
    return float(overlap / union) if union > 0 else 0.0


def match_detections(dets: list[dict], truth: list[dict], iou_min: float = 0.5) -> list[bool]:
    """Greedy one-to-one matching, best box first: True where a detection is a true positive.

    `dets` is expected in descending score order, which is what decode returns, so the
    confident detection gets first pick of the truth boxes. A truth box is consumed once: a
    second detection of the same object is a false positive, the same way the farm loop would
    see it. Classes never match each other, whatever the overlap.
    """
    taken = [False] * len(truth)
    hits = []
    for det in dets:
        best, best_iou = -1, iou_min
        box = (det["x"], det["y"], det["w"], det["h"])
        for i, real in enumerate(truth):
            if taken[i] or real["class_index"] != det["class_index"]:
                continue
            score = iou(box, (real["x"], real["y"], real["w"], real["h"]))
            if score >= best_iou:
                best, best_iou = i, score
        if best >= 0:
            taken[best] = True
        hits.append(best >= 0)
    return hits


def target_for(name: str, *, other: float = MIN_PRECISION_OTHER) -> float:
    """The precision a class has to reach: MIN_PRECISION for a tap anchor, `other` otherwise."""
    from tools.play import classes  # inside the body: the import test allows numpy at top level

    return MIN_PRECISION if name in classes.TAP_ANCHORS else other


def rule_for(name: str, *, best_f1: bool = False) -> str:
    """RULE_BEST_F1 if it was asked for and `name` is not a tap anchor, else RULE_PRECISION."""
    from tools.play import classes  # inside the body: the import test allows numpy at top level

    return RULE_BEST_F1 if best_f1 and name not in classes.TAP_ANCHORS else RULE_PRECISION


def pick(
    scores: Sequence[float], is_tp: Sequence[bool], *, min_precision: float = MIN_PRECISION
) -> float:
    """The lowest threshold in [FLOOR, CEILING] whose precision reaches min_precision.

    The search walks a one hundredth grid upwards and stops at the first threshold that is
    clean enough, so the answer is already rounded to two decimals. A class with no detection
    at all, or one that never reaches the precision at any threshold, gets CEILING: the model
    has not earned the right to be believed about it.
    """
    keep = np.asarray(scores, dtype=np.float64)
    true = np.asarray(is_tp, dtype=bool)
    steps = int(round((CEILING - FLOOR) * 100))
    for step in range(steps + 1):
        threshold = round(FLOOR + step / 100, 2)
        above = keep >= threshold
        total = int(above.sum())
        if total and int(true[above].sum()) / total >= min_precision:
            return threshold
    return CEILING


def pick_best_f1(name: str, scores: Sequence[float], is_tp: Sequence[bool], truth: int) -> float:
    """The threshold in [FLOOR, CEILING] with the best F1 against `truth` real boxes.

    Walks the same grid as pick, and a tie goes to the higher threshold, the cleaner of the two.
    A class with fewer than MIN_TRUTH truth boxes, or with no true positive at any threshold,
    gets CEILING: there is nothing to balance. A tap anchor is refused outright, because a false
    tap costs far more than a miss and F1 prices the two the same.
    """
    if rule_for(name, best_f1=True) != RULE_BEST_F1:
        raise ValueError(f"{name} is a tap anchor: its threshold is never picked by F1")
    if truth < MIN_TRUTH:
        return CEILING
    keep = np.asarray(scores, dtype=np.float64)
    true = np.asarray(is_tp, dtype=bool)
    best, best_f1 = CEILING, 0.0
    steps = int(round((CEILING - FLOOR) * 100))
    for step in range(steps + 1):
        threshold = round(FLOOR + step / 100, 2)
        above = keep >= threshold
        f1 = 2 * int(true[above].sum()) / (int(above.sum()) + truth)
        if f1 > 0 and f1 >= best_f1:
            best, best_f1 = threshold, f1
    return best


def decode(
    dets: np.ndarray, logits: np.ndarray, width: int, height: int, floor: float = 0.05
) -> list[dict]:
    """RF-DETR's two ONNX outputs turned into pixel boxes, highest score first.

    `dets` is (queries, 4) as centre x, centre y, width, height in 0..1, `logits` is
    (queries, classes + 1) of unnormalised scores; a leading batch dimension of one is
    accepted for both. Column k is class index k, which is COCO category id k + 1, and the
    last column is the background the library never reports. Every query and class above
    `floor` becomes a box, so the caller can pick its own threshold per class afterwards.
    """
    boxes = np.asarray(dets, dtype=np.float64).reshape(-1, np.asarray(dets).shape[-1])
    scores = np.asarray(logits, dtype=np.float64)
    scores = scores.reshape(-1, scores.shape[-1])[:, :-1]
    scores = 1.0 / (1.0 + np.exp(-scores))

    found = []
    for query, cls in zip(*np.nonzero(scores >= floor), strict=True):
        cx, cy, w, h = boxes[query]
        found.append(
            {
                "class_index": int(cls),
                "score": float(scores[query, cls]),
                "x": float((cx - w / 2) * width),
                "y": float((cy - h / 2) * height),
                "w": float(w * width),
                "h": float(h * height),
            }
        )
    found.sort(key=lambda box: box["score"], reverse=True)
    return found


def preprocess(bgr: np.ndarray, size: int = 384) -> np.ndarray:
    """A BGR screen as the ONNX model's input: RGB, size x size, normalised, NCHW, batch of one.

    The resize is a plain squash with no letterbox, which is what the library does, so a box
    decoded from the output maps back to the original picture by scaling alone.
    """
    import cv2  # inside the body: this module has to import wherever numpy alone is installed

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    scaled = small.astype(np.float32) / 255.0
    scaled = (scaled - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return np.ascontiguousarray(scaled.transpose(2, 0, 1)[None], dtype=np.float32)
