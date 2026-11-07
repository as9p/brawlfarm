"""
Select the lowest-trophy brawler — a once-per-session startup action (not part of the
per-frame farm loop). The farm plays whatever brawler is selected, so to grind
efficiently we keep the lowest-trophy one active (most headroom before the 1000 cap).

Flow (open from menu -> act -> exit to menu via taps):
  menu -> BRAWLERS button (left side) -> sort by "Least Trophies" -> make sure the Quest
  and Heart/Favorites filter toggles are OFF -> tap the TOP-LEFT card (now the lowest-
  trophy brawler) -> tap SELECT on the detail screen -> back to the menu.

Why the toggles must be OFF: when the Quest or Heart filter is ON the grid is reduced to
a subset, so the top-left card is no longer the global lowest. An ON toggle shows an
ORANGE check badge; the Quest clipboard icon's own check is BLUE (always there), so we
gate on ORANGE specifically.

Safety: TAPS only, never the Android BACK key (project rule); exit via the top-left back
arrow and close any stray popup via close_x. Idempotent — safe to re-run (re-selecting the
same sort/brawler is a no-op, toggles are only tapped when ON).

A later, separate feature will reuse select_lowest_trophy_brawler() when a brawler hits
1000 trophies (prestige) to rotate to the new lowest — see the legacy planning notes (not ported).
"""

from __future__ import annotations

import time

from brawlfarm.core import adb, config, recalib, states, vision


def _on_brawlers_screen(screen) -> bool:
    """True if the BRAWLERS screen is open (its 'BRAWLERS' header sits top-centre)."""
    return vision.find_text(screen, "BRAWLERS", region=config.BRAWLERS_HEADER_REGION) is not None


def _toggle_is_on(screen, center) -> bool:
    """True if the filter toggle at `center` is ON (shows an ORANGE check badge). The
    Quest clipboard's own check is blue, so an orange gate ignores it."""
    cx, cy = center
    h = config.BRAWLER_TOGGLE_HALF
    frac = vision.color_fraction(
        screen,
        (cx - h, cy - h, cx + h, cy + h),
        config.BRAWLER_TOGGLE_ORANGE_LO,
        config.BRAWLER_TOGGLE_ORANGE_HI,
    )
    return frac >= config.BRAWLER_TOGGLE_ON_FRAC


def _ensure_toggle_off(center, name, log) -> None:
    """If the toggle at `center` is ON, tap it off (one verify + retry)."""
    for _ in range(2):
        if not _toggle_is_on(adb.screencap(), center):
            return
        adb.tap(*center)
        log(f"[brawlers] turned {name} filter OFF")
        time.sleep(1.0)


def _set_sort(item_label: str, log) -> bool:
    """Open the sort dropdown and pick ``item_label`` (idempotent — re-picking the
    active sort is harmless). Returns whether the item was found+tapped."""
    adb.tap(*config.BRAWLER_SORT_LABEL)
    time.sleep(1.2)
    item = vision.find_text(adb.screencap(), item_label, region=config.BRAWLER_SORT_DROPDOWN_REGION)
    if item is None:
        log(f"[brawlers] {item_label!r} not found in the sort dropdown — leaving sort as is")
        return False
    adb.tap(*item)
    time.sleep(1.2)
    log(f"[brawlers] sort set to {item_label}")
    return True


def _set_least_trophies(log) -> bool:
    return _set_sort("Least Trophies", log)


def _exit_to_menu() -> None:
    """Return to the main menu using TAPS only (no BACK key): tap the top-left back arrow
    until states.classify sees the MENU; close a stray popup via its red close_x."""
    for _ in range(6):
        screen = adb.screencap()
        state = states.classify(screen)
        if state is states.State.MENU:
            return
        if state is states.State.POPUP:
            x = vision.find(screen, "close_x")
            adb.tap(*(x.center if x is not None else config.CLOSE_X_BUTTON))
        else:
            adb.tap(*config.QUESTS_CLOSE_BUTTON)  # top-left back arrow (shared menu-screen exit)
        time.sleep(1.2)


def select_lowest_trophy_brawler(log=print) -> str | None:
    """Open BRAWLERS, sort by Least Trophies, force the Quest/Heart filters OFF, then
    select the top-left (lowest-trophy) brawler. Returns the brawler's name (OCR'd from the
    detail screen), or None if the select failed and the game still plays what it had: the
    sort didn't take or the name didn't read, so SELECT was never tapped. Leaves the game on
    the main menu. Assumes we start at/near it."""
    adb.tap(*config.BRAWLERS_BUTTON)
    # Poll for the screen instead of a flat 2s wait: proceed the instant it's up (usually
    # <1s), but still allow a slow open before bailing. This trims the dead gap after the
    # shop step without risking a premature "didn't open" bail — which wouldn't retry this
    # session and would leave the farm grinding the wrong brawler.
    opened = False
    for _ in range(8):
        time.sleep(0.3)
        if _on_brawlers_screen(adb.screencap()):
            opened = True
            break
    if not opened:
        log("[brawlers] BRAWLERS screen didn't open — leaving")
        _exit_to_menu()
        return None

    if not _set_least_trophies(log):
        log("[brawlers] sort didn't take, leaving")
        adb.tap(*config.BRAWLER_SORT_LABEL)  # close the stray dropdown
        time.sleep(0.8)
        _exit_to_menu()
        return None
    _ensure_toggle_off(config.BRAWLER_QUEST_TOGGLE, "Quest", log)
    _ensure_toggle_off(config.BRAWLER_HEART_TOGGLE, "Heart", log)

    # Top-left card is the lowest-trophy brawler now that the list is sorted + unfiltered.
    adb.tap(*config.BRAWLER_FIRST_CARD)
    time.sleep(1.8)
    screen = adb.screencap()

    # OCR the brawler name (largest non-chrome line near the top). No name means nothing
    # says what the card tap opened, so SELECT would be a blind tap.
    name = _read_brawler_name(screen)
    if not name:
        log("[brawlers] brawler name didn't read on the detail screen, leaving")
        _exit_to_menu()
        return None

    # Tap the fixed SELECT button (OCR mangles the styled label, so we don't OCR it).
    # Idempotent: a greyed "SELECTED" on an already-active brawler is a harmless no-op.
    adb.tap(*config.BRAWLER_SELECT_BUTTON)
    time.sleep(1.2)
    log(f"[brawlers] selected {name}")

    _exit_to_menu()
    return name


# --- select BY NAME (farm plan) ------------------------------------------------
# The farm plan (core/farmplan.py) can name an explicit brawler to grind. Flow:
# BRAWLERS screen -> sort "Name" (alphabetical, digits first) -> filters OFF ->
# name-directed scan (OCR the visible card names, compare the target with the names
# on screen, swipe one screen sideways toward it, re-check) -> tap the card -> VERIFY
# the name on the detail screen (config.BRAWLER_NAME_REGION) -> SELECT.
# Geometry lives in config (BRAWLER_GRID_* / BRAWLER_SCROLL_*).


def _norm(name: str | None) -> str:
    """Collapse a brawler name for OCR-proof comparison: uppercase, alnum only
    ("EL PRIMO"→"ELPRIMO", "8-BIT"→"8BIT", "LARRY & LAWRIE"→"LARRYLAWRIE")."""
    return "".join(c for c in (name or "").upper() if c.isalnum())


def _visible_cards(screen, owned_norm: set[str]) -> dict[str, tuple[int, int]]:
    """OCR the visible grid and return {normalized name: card tap point}. The name label
    sits bottom-right IN the card (BRAWLER_NAME_LABEL_DX right of and
    BRAWLER_NAME_LABEL_DY below the card center), so subtracting that offset lands near
    the center, which is then snapped to the nearest column center (COL0_X + k * COL_W)
    and the nearest of the 3 row centers. A column the grid region cuts in half is
    dropped: only part of that card is on screen, so its center is not a tap point."""
    region_left, _top, region_right, _bottom = config.BRAWLER_GRID_REGION
    half_w = config.BRAWLER_CARD_W // 2
    cards: dict[str, tuple[int, int]] = {}
    for text, _conf, (lx, ly) in vision.read_lines_boxes(screen, region=config.BRAWLER_GRID_REGION):
        n = _norm(text)
        if n not in owned_norm:
            continue
        cx, cy = lx - config.BRAWLER_NAME_LABEL_DX, ly - config.BRAWLER_NAME_LABEL_DY
        col = round((cx - config.BRAWLER_GRID_COL0_X) / config.BRAWLER_GRID_COL_W)
        card_x = config.BRAWLER_GRID_COL0_X + col * config.BRAWLER_GRID_COL_W
        if card_x - half_w < region_left or card_x + half_w > region_right:
            continue  # half-cut column: tapping it would land on the card's edge or past it
        card_y = min(config.BRAWLER_GRID_ROWS_Y, key=lambda ry: abs(ry - cy))
        cards[n] = (card_x, card_y)
    return cards


def _scroll_grid(direction: str) -> None:
    """Swipe the column-major grid one screen sideways along the middle row. Dragging
    "left" pulls the content left and reveals the columns further RIGHT (the later names
    under the Name sort); "right" is the reverse. Directions only, never a pixel count:
    the grid snaps back to whole columns, so a short swipe just undoes itself."""
    if direction == "left":
        x1, x2 = config.BRAWLER_SCROLL_X_RIGHT, config.BRAWLER_SCROLL_X_LEFT
    elif direction == "right":
        x1, x2 = config.BRAWLER_SCROLL_X_LEFT, config.BRAWLER_SCROLL_X_RIGHT
    else:
        raise ValueError(f"_scroll_grid direction must be 'left' or 'right', got {direction!r}")
    y = config.BRAWLER_SCROLL_Y
    adb.swipe(x1, y, x2, y, config.BRAWLER_SCROLL_MS)
    time.sleep(1.0)  # let the fling settle before re-reading the grid


def select_brawler_by_name_checked(
    target: str, owned_names: list[str], log=print
) -> tuple[str | None, str | None]:
    """Open BRAWLERS, sort alphabetically, scan sideways for ``target`` (one of the
    API's OWNED brawler names) and select it, verifying the detail-screen name by OCR
    before tapping SELECT, and bailing safely (back to menu, return None) on any
    mismatch. Returns ``(name, suspicion)``: the verified name or None, plus a short
    detail string when the season-rollover tripwire saw the BRAWLERS screen verify
    fine yet the grid OCR read 0 owned names across the FULL scroll budget (None
    otherwise; see ops-resilience.md §B — the grid layout/label style may have
    shifted). Assumes we start at/near the menu."""
    target_n = _norm(target)
    owned_norm = {_norm(n) for n in owned_names}
    if target_n not in owned_norm:
        log(f"[brawlers] target {target!r} is not in the owned list — leaving")
        return None, recalib.SKIP  # never navigated: no observation (review #56)

    adb.tap(*config.BRAWLERS_BUTTON)
    opened = False
    for _ in range(8):
        time.sleep(0.3)
        if _on_brawlers_screen(adb.screencap()):
            opened = True
            break
    if not opened:
        log("[brawlers] BRAWLERS screen didn't open — leaving")
        _exit_to_menu()
        return None, recalib.SKIP  # screen never verified: no observation (review #56)

    if not _set_sort(config.BRAWLER_SORT_NAME_ITEM, log):
        adb.tap(*config.BRAWLER_SORT_LABEL)  # close the stray dropdown
        time.sleep(0.8)
        _exit_to_menu()
        return None, recalib.SKIP  # grid never read under the right sort (review #56)
    _ensure_toggle_off(config.BRAWLER_QUEST_TOGGLE, "Quest", log)
    _ensure_toggle_off(config.BRAWLER_HEART_TOGGLE, "Heart", log)

    # Which way the next swipe goes. The Name sort runs the roster left to right, so the
    # names after the ones on screen are further right, which is where a blind scan starts.
    direction = "left"
    saw_names = False  # tripwire: did the grid OCR EVER read an owned name?
    for attempt in range(config.BRAWLER_SELECT_MAX_SWIPES):
        screen = adb.screencap()
        cards = _visible_cards(screen, owned_norm)
        saw_names = saw_names or bool(cards)
        if target_n in cards:
            adb.tap(*cards[target_n])
            time.sleep(1.8)
            detail = adb.screencap()
            ok, shown = _detail_name_matches(detail, target_n, owned_norm)
            if ok:
                adb.tap(*config.BRAWLER_SELECT_BUTTON)
                time.sleep(1.2)
                log(f"[brawlers] selected {target} (detail showed {shown!r})")
                _exit_to_menu()
                return target, None
            log(f"[brawlers] detail screen shows {shown!r}, wanted {target!r} — backing out")
            _exit_to_menu()
            return None, None
        # Not visible: the names on screen say which way the target is. Before the
        # smallest one it's in the columns to the left, after the largest one in the
        # columns to the right. Between the two it's a gap in the grid (not owned on
        # this account, or a misread), so keep going the way we were and never bounce.
        if cards:
            if target_n < min(cards):
                direction = "right"
            elif target_n > max(cards):
                direction = "left"
        _scroll_grid(direction)
        log(f"[brawlers] scrolling {direction} toward {target} (attempt {attempt + 1})")

    log(f"[brawlers] couldn't reach {target!r} after scroll budget — leaving")
    _exit_to_menu()
    # Season-rollover tripwire: the screen verified (we got past the open check)
    # but a FULL self-correcting scroll never OCR'd a single owned name — that's
    # not "target unreachable", that's "the grid reads as blank". Pure bookkeeping
    # on values already computed, so nothing here can disturb the select flow.
    suspicion = None
    if config.RECALIB_TRIPWIRE and not saw_names:
        suspicion = (
            f"BRAWLERS screen verified but grid OCR read 0 owned names "
            f"across {config.BRAWLER_SELECT_MAX_SWIPES} swipes"
        )
        log(f"[brawlers] tripwire: {suspicion} — possible UI reskin")
    return None, suspicion


def _detail_name_matches(screen, target_n: str, owned_norm: set[str]) -> tuple[bool, str | None]:
    """Verify the detail screen belongs to the target brawler. Gotcha (measured live):
    the header shows the EQUIPPED SKIN's name when one is set — BO read as
    "WARRIOR"+"BO", EL PRIMO's skin OCR'd as the space-collapsed "VAMPRIMO". So:
      1. exact: any line normalizes to the target;
      2. skin pattern: a line's last 1-2 words, or its suffix (len>=4 to keep
         "TURBO" from passing for "BO"), normalize to the target;
      3. fuzzy: the target is the best difflib match (>=0.6) among ALL owned names
         across the lines — catches collapsed skin names while still flagging a
         mis-tap, which would resemble a DIFFERENT brawler's name more.
    Returns (matched, first plausible line for the log)."""
    import difflib

    lines = [t for t, _ in vision.read_lines(screen, region=config.BRAWLER_NAME_REGION)]
    for t in lines:
        n = _norm(t)
        if n == target_n:
            return True, t
        if len(target_n) >= 4 and n.endswith(target_n):
            return True, t
        words = t.replace("-", " ").split()
        for k in (1, 2):
            if len(words) >= k and _norm("".join(words[-k:])) == target_n:
                return True, t
    # Fuzzy disambiguation: which owned brawler do the lines resemble most?
    best_name, best_score = None, 0.0
    for t in lines:
        n = _norm(t)
        if len(n) < 2:
            continue
        for cand in owned_norm:
            score = difflib.SequenceMatcher(None, n, cand).ratio()
            if score > best_score:
                best_name, best_score = cand, score
    if best_name == target_n and best_score >= 0.6:
        return True, next((t for t in lines if _norm(t)), None)
    return False, next((t for t in lines if _norm(t)), None)


def _read_brawler_name(screen) -> str | None:
    """Read the brawler's name off the detail screen, None if no line reads as one. Skips
    obvious UI chrome so a button label never passes for the brawler."""
    chrome = {
        "SELECT",
        "SELECTED",
        "BRAWLERS",
        "INFO",
        "SKINS",
        "POWER",
        "GADGET",
        "TRY",
    }
    for text, _conf in vision.read_lines(screen, region=config.BRAWLER_NAME_REGION):
        t = text.strip()
        # Brawler names are alphabetic; skip badges/counters like "2/24" or "0/250".
        if len(t) >= 3 and t.upper() not in chrome and t.replace(" ", "").isalpha():
            return t
    return None
