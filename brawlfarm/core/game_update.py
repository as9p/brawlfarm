"""Game client auto-update through the Play Store (2026-11-04).

When a new Brawl Stars client ships, the game opens to an update prompt and never plays.
``try_update`` opens the store page, taps "Update" if the page shows one, and waits for
the installed versionName to change. Every effect comes in through ``Effects``, so the
flow is tested on fakes. The controller decides when to call it (once per run).

Safety rails: the only tap is an OCR line that equals ``config.STORE_UPDATE_NEEDLE``.
A line holding a ``config.STORE_NEVER_TAP`` word is never a candidate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from brawlfarm.core import adb, config


class UpdateResult(Enum):
    UPDATED = "updated"
    NO_UPDATE = "no_update"
    FAILED = "failed"


@dataclass(frozen=True)
class Outcome:
    result: UpdateResult
    reason: str
    old: str | None = None
    new: str | None = None


@dataclass(frozen=True)
class Effects:
    open_store: Callable[[], None]
    foreground: Callable[[], str]
    screenshot: Callable[[], Any]
    read_boxes: Callable[[Any], list[tuple[str, float, tuple[int, int]]]]
    tap: Callable[[tuple[int, int]], None]
    version: Callable[[], str | None]
    sleep: Callable[[float], None]
    clock: Callable[[], float]


def pick_update_button(
    lines: list[tuple[str, float, tuple[int, int]]],
) -> tuple[int, int] | None:
    """Centre of the OCR line that reads exactly the update needle, else None.
    Lines containing a never-tap word are skipped before the needle is compared."""
    needle = config.STORE_UPDATE_NEEDLE.lower()
    for text, _conf, centre in lines:
        word = text.strip().lower()
        if any(bad in word for bad in config.STORE_NEVER_TAP):
            continue
        if word == needle:
            return centre
    return None


def _wait_for_store(fx: Effects) -> bool:
    deadline = fx.clock() + config.STORE_OPEN_TIMEOUT_S
    while True:
        if fx.foreground() == config.PLAY_STORE_PACKAGE:
            return True
        if fx.clock() >= deadline:
            return False
        fx.sleep(1.0)


def try_update(fx: Effects) -> Outcome:
    """Run the store flow once. Never raises on adb trouble: that is a FAILED outcome."""
    try:
        old = fx.version()
        if old is None:
            return Outcome(UpdateResult.FAILED, "version_unreadable")
        fx.open_store()
        if not _wait_for_store(fx):
            return Outcome(UpdateResult.FAILED, "store_not_foreground", old=old)
        fx.sleep(config.STORE_SETTLE_S)
        point = pick_update_button(fx.read_boxes(fx.screenshot()))
        if point is None:
            return Outcome(UpdateResult.NO_UPDATE, "no_update_button", old=old)
        if fx.foreground() != config.PLAY_STORE_PACKAGE:
            return Outcome(UpdateResult.FAILED, "store_not_foreground", old=old)
        fx.tap(point)
        deadline = fx.clock() + config.GAME_UPDATE_TIMEOUT_S
        while fx.clock() < deadline:
            fx.sleep(config.GAME_UPDATE_POLL_S)
            new = fx.version()
            if new is not None and new != old:
                return Outcome(UpdateResult.UPDATED, "installed", old=old, new=new)
        return Outcome(UpdateResult.FAILED, "timeout", old=old)
    except adb.AdbError as e:
        return Outcome(UpdateResult.FAILED, f"adb: {e}")
