"""Offline coverage of the HUD reader over a folder of frames. Reads images, writes one JSON file
outside the repository, prints the coverage per field and the two checks that need no labels:
bar fill against the digits on the same frame, and teams left never rising within a match. Runs
on the CPU, one process, at most two OpenCV threads.

    uv run python tools/play/reader_check.py FRAMES --out OUT.json
    uv run python tools/play/reader_check.py FRAMES --boxes det.jsonl --out OUT.json

--boxes holds one {"file", "boxes": [[cls, score, x, y, w, h], ...]} per line. Only the frames it
names are read, with those boxes.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from brawlfarm.play import reader
from brawlfarm.play.detect import Box

REPO = Path(__file__).resolve().parents[2]
# A match is an unbroken run of frames with a teams left value. On a full folder a gap of more
# than this many frame numbers is the end screens and the lobby between two matches.
MATCH_GAP = 3
# Maximum health a cube adds: full bars on the capture step by this, for both brawlers in it.
CUBE_HEALTH = 400


def _number(name: str) -> int:
    found = re.findall(r"\d+", Path(name).stem)
    return int(found[-1]) if found else 0


def _load_boxes(path: Path) -> dict[str, list[Box]]:
    out: dict[str, list[Box]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out[row["file"]] = [Box(*b) for b in row["boxes"]]
    return out


def split_matches(rows: list[dict[str, Any]], gap: int = MATCH_GAP) -> list[list[dict[str, Any]]]:
    """Rows in frame order, cut where the frames with a teams left value are far apart."""
    matches: list[list[dict[str, Any]]] = []
    last: int | None = None
    for row in rows:
        if row["teams_left"] is None:
            if matches:
                matches[-1].append(row)
            continue
        if last is None or row["n"] - last > gap:
            matches.append([])
        matches[-1].append(row)
        last = row["n"]
    return matches


def teams_rises(matches: list[list[dict[str, Any]]]) -> list[tuple[int, int, int]]:
    """(frame, before, after) wherever teams left went up inside one match."""
    rises = []
    for match in matches:
        seen = [r for r in match if r["teams_left"] is not None]
        for a, b in zip(seen, seen[1:], strict=False):
            if b["teams_left"] > a["teams_left"]:
                rises.append((b["n"], a["teams_left"], b["teams_left"]))
    return rises


def fill_errors(matches: list[list[dict[str, Any]]]) -> list[float]:
    """For a part bar, how far its fill is from health over maximum health on that same frame.
    Maximum health is the match's base plus CUBE_HEALTH per cube; the base is the commonest full
    bar health less its cubes. No cube reading counts as zero, the badge hides at zero."""
    errors = []
    for match in matches:
        read = [r for r in match if r["self_health"] and r["self_health_fraction"] is not None]
        bases = Counter(
            r["self_health"] - CUBE_HEALTH * (r["cube_count"] or 0)
            for r in read
            if r["self_health_fraction"] == 1.0
        )
        if not bases:
            continue
        base = bases.most_common(1)[0][0]
        for r in read:
            if r["self_health_fraction"] < 1.0:
                top = base + CUBE_HEALTH * (r["cube_count"] or 0)
                errors.append(abs(r["self_health_fraction"] - r["self_health"] / top))
    return errors


def _report(rows: list[dict[str, Any]], gap: int) -> None:
    total = len(rows)
    for field in (
        "teams_left",
        "self_health",
        "self_health_fraction",
        "self_ammo",
        "cube_count",
        "teammates_alive",
    ):
        got = sum(r[field] is not None for r in rows)
        print(f"{field:22s} {got:5d} of {total} ({100 * got / max(total, 1):.1f}%)")
    got = sum(bool(r["enemy_health"]) for r in rows)
    print(f"{'enemy_health':22s} {got:5d} of {total} ({100 * got / max(total, 1):.1f}%)")
    matches = split_matches(rows, gap)
    rises = teams_rises(matches)
    print(f"matches {len(matches)}, teams left rises {len(rises)}: {rises[:10]}")
    errors = fill_errors(matches)
    if errors:
        errors.sort()
        p90 = errors[int(len(errors) * 0.9)]
        print(
            f"fill against digits on {len(errors)} part bars: "
            f"median error {statistics.median(errors):.3f}, p90 {p90:.3f}"
        )
    else:
        print("fill against digits: no part bar after a full bar")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames", type=Path)
    ap.add_argument("--boxes", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gap", type=int, default=MATCH_GAP)
    args = ap.parse_args()
    if args.out.resolve().is_relative_to(REPO):
        print("--out must be outside the repository: frames never enter it", file=sys.stderr)
        return 2
    cv2.setNumThreads(2)
    boxes = _load_boxes(args.boxes) if args.boxes else None
    names = sorted(boxes) if boxes else sorted(p.name for p in args.frames.glob("*.jpg"))
    rows: list[dict[str, Any]] = []
    for name in names:
        frame = cv2.imread(str(args.frames / name))
        if frame is None:
            continue
        reading = reader.read(frame, boxes[name] if boxes else ())
        rows.append({"file": name, "n": _number(name), **asdict(reading)})
    _report(rows, args.gap)
    args.out.write_text(json.dumps(rows), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
