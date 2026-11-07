"""The Play Store update flow, run entirely on fakes: no device, no OCR model."""

from __future__ import annotations

import numpy as np
import pytest

from brawlfarm.core import adb, game_update
from brawlfarm.core.game_update import Effects, UpdateResult, pick_update_button

STORE = "com.android.vending"
UPDATE = ("Update", 0.99, (1200, 520))
UNINSTALL = ("Uninstall", 0.98, (900, 520))


def _fx(*, fg=STORE, lines=(UNINSTALL, UPDATE), versions=("1.0", "1.0", "2.0"), open_raises=None):
    calls: list = []
    t = [0.0]
    vs = list(versions)
    fgs = [fg] if isinstance(fg, str) else list(fg)

    def version():
        return vs.pop(0) if len(vs) > 1 else vs[0]

    def foreground():
        return fgs.pop(0) if len(fgs) > 1 else fgs[0]

    def open_store():
        calls.append("open")
        if open_raises is not None:
            raise open_raises

    def sleep(s):
        t[0] += s

    fx = Effects(
        open_store=open_store,
        foreground=foreground,
        screenshot=lambda: np.zeros((900, 1600, 3), np.uint8),
        read_boxes=lambda img: list(lines),
        tap=lambda pt: calls.append(("tap", pt)),
        version=version,
        sleep=sleep,
        clock=lambda: t[0],
    )
    return fx, calls, t


def test_update_found_then_version_changes_is_updated() -> None:
    fx, calls, _ = _fx()
    out = game_update.try_update(fx)
    assert out.result is UpdateResult.UPDATED
    assert (out.old, out.new, out.reason) == ("1.0", "2.0", "installed")
    assert calls == ["open", ("tap", (1200, 520))]


def test_no_update_word_is_no_update_and_never_taps() -> None:
    fx, calls, _ = _fx(lines=(UNINSTALL, ("Open", 0.99, (1200, 520))))
    out = game_update.try_update(fx)
    assert out.result is UpdateResult.NO_UPDATE
    assert out.reason == "no_update_button"
    assert calls == ["open"]


def test_uninstall_alone_is_never_tapped() -> None:
    fx, calls, _ = _fx(lines=(UNINSTALL,))
    assert game_update.try_update(fx).result is UpdateResult.NO_UPDATE
    assert not any(isinstance(c, tuple) for c in calls)


def test_version_never_changes_fails_after_timeout() -> None:
    fx, calls, t = _fx(versions=("1.0",))
    out = game_update.try_update(fx)
    assert out.result is UpdateResult.FAILED
    assert out.reason == "timeout"
    assert ("tap", (1200, 520)) in calls
    assert t[0] >= 600


def test_store_never_reaches_foreground_fails_without_tapping() -> None:
    fx, calls, _ = _fx(fg="com.supercell.brawlstars")
    out = game_update.try_update(fx)
    assert out.result is UpdateResult.FAILED
    assert out.reason == "store_not_foreground"
    assert calls == ["open"]


def test_store_leaves_foreground_before_tap_fails_without_tapping() -> None:
    fx, calls, _ = _fx(fg=(STORE, "com.supercell.brawlstars"))
    out = game_update.try_update(fx)
    assert out.result is UpdateResult.FAILED
    assert out.reason == "store_not_foreground"
    assert calls == ["open"]


def test_unreadable_version_is_inconclusive_before_opening_store() -> None:
    fx, calls, _ = _fx(versions=(None,))
    out = game_update.try_update(fx)
    assert out.result is UpdateResult.NO_UPDATE
    assert out.reason == "version_unreadable"
    assert calls == []


def test_adb_error_is_failed_not_raised() -> None:
    fx, _, _ = _fx(open_raises=adb.AdbError("device offline"))
    out = game_update.try_update(fx)
    assert out.result is UpdateResult.FAILED
    assert out.reason.startswith("adb: ")


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        ([UPDATE], (1200, 520)),
        ([("  UPDATE ", 0.9, (5, 6))], (5, 6)),
        ([UNINSTALL, UPDATE], (1200, 520)),
        ([("Update all", 0.9, (1, 1))], None),
        ([("Uninstall update", 0.9, (1, 1))], None),
        ([("Updated recently", 0.9, (1, 1))], None),
        ([], None),
    ],
)
def test_pick_update_button(lines, expected) -> None:
    assert pick_update_button(lines) == expected
