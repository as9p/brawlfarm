"""
Per-frame screen classification.

We detect reliable ANCHOR screens via template matching. The controller's state
machine (core/controller.py) uses these; UNKNOWN is resolved by phase context.

Validated separations (own screen vs others) at the thresholds below:
  play / playagain / proceed / exit / trophy_screen -> crisp (~1.0 vs <=0.5)
  matchmaking -> clean;  teams_left (in-match) -> 0.98-1.0 vs <=0.47 off-match
  reload (disconnect) -> 1.0 on both disconnect modals; other_device distinguishes them
"""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from brawlfarm.core import config, vision


class State(Enum):
    DISCONNECT = "disconnect"  # "Idle Disconnect" / "Another device" modal (RELOAD)
    MENU = "menu"  # main menu (PLAY button visible)
    MATCHMAKING = "matchmaking"  # "Players found N/12" lobby
    RESULTS = "results"  # post-match screens (PLAY AGAIN / PROCEED / EXIT)
    IN_MATCH = "in_match"  # actually in a match ("Teams left:" HUD present)
    TROPHY_SCREEN = "trophy_screen"  # individual brawler trophy / Trophy Road screen
    POPUP = "popup"  # promo/news/offer popup with a red ✕ close button
    UNKNOWN = "unknown"  # loading or a transient screen — resolved by context


# Anchor template -> the State it proves. Notes on the individual anchors:
#   * playagain/proceed/exit are the three RESULTS buttons (two results screens).
#   * teams_left is the positive in-match signal (the "Teams left:" HUD), per user
#     guidance — so we only move/attack when actually in a match, never blind-swiping
#     a loading screen.
#   * trophy_screen (black "TROPHIES" — Trophy Road) and trophy_brawler (white
#     "TROPHIES" header — the brawler rank/trophy detail auto-shown at rank-up
#     milestones) are two variants of the same never-linger screen; the returning
#     handler taps HOME to leave. RESULTS is checked first, so the post-match
#     trophy-progress screen still classifies as RESULTS, not TROPHY_SCREEN.
_ANCHOR_STATE = {
    "play": State.MENU,
    "playagain": State.RESULTS,
    "proceed": State.RESULTS,
    "exit": State.RESULTS,
    "matchmaking": State.MATCHMAKING,
    "teams_left": State.IN_MATCH,
    "trophy_screen": State.TROPHY_SCREEN,
    "trophy_brawler": State.TROPHY_SCREEN,
}

# The original (pre-phase-hint) check order. This order is PRIORITY, not just cost:
# e.g. RESULTS before TROPHY_SCREEN keeps the post-match trophy screen as RESULTS.
_BASE_ORDER = (
    "play",
    "playagain",
    "proceed",
    "exit",
    "matchmaking",
    "teams_left",
    "trophy_screen",
    "trophy_brawler",
)

# Phase-aware check order (config.PHASE_CLASSIFY): the controller passes its phase so
# classify can test the EXPECTED anchor first (find returns on first hit, so order =
# expected-case latency). Two hard rules, both corpus-validated (562 real frames,
# tools/validate_gray_thresholds.py — zero decision changes vs _BASE_ORDER):
#
#   1. REORDER ONLY, never a subset. An earlier draft skipped `matchmaking` in
#      `playing`; wrong — phase_queuing hands off to `playing` on an UNKNOWN flicker
#      once matchmaking has been seen, so a MATCHMAKING frame can still arrive under
#      the `playing` hint, and the run loop treats UNKNOWN specially (daily-streak/
#      team-invite OCR + the unknown-streak foreground probe).
#   2. `matchmaking` must stay BEHIND play + the results anchors. Its template
#      co-matches at 0.90–0.97 on many MENU and RESULTS frames (164/562 in the
#      corpus, both color and gray) — only the play/results-first priority masks
#      that. A `queuing` order with matchmaking first misclassified 40+ menu frames
#      as MATCHMAKING, so queuing keeps _BASE_ORDER (the matchmaking hit then costs
#      4 cheap gray misses first — fine).
#
#   * playing: teams_left first (the common case — every in-match frame otherwise
#     pays for 5 misses before it). Safe to move ahead of everything: teams_left
#     co-matches nothing in the anchor set (its high off-state scores occur only
#     under reload/close_x overlays, which are checked before any anchor).
#   * queuing / returning / at_menu: base order (menu/results anchors already lead).
PHASE_ORDER = {
    "playing": (
        "teams_left",
        "play",
        "playagain",
        "proceed",
        "exit",
        "matchmaking",
        "trophy_screen",
        "trophy_brawler",
    ),
}

# Guard the reorder-only invariant above (import-time, zero runtime cost).
assert all(set(order) == set(_BASE_ORDER) for order in PHASE_ORDER.values()), (
    "PHASE_ORDER entries must be pure reorders of _BASE_ORDER, never subsets"
)


def _anchor_threshold(name: str) -> float:
    """Per-anchor match threshold; the single source is vision.threshold_for."""
    return vision.threshold_for(name)


def classify(screen: np.ndarray, phase: str | None = None) -> State:
    """Classify one frame. `phase` is an optional hint (the controller's current phase
    name) that only re-orders/trims the anchor checks for speed — it never changes the
    resulting State for a frame (see PHASE_ORDER). Callers without phase context (and
    everything when config.PHASE_CLASSIFY is off) get the original full chain."""
    # DISCONNECT first: this modal overlays match/results, so those templates still
    # match behind it — catch the modal first.
    if vision.find(screen, "reload", threshold=0.9) is not None:
        return State.DISCONNECT
    # Promo/offer popups overlay the menu (hiding PLAY), so check the close ✕ before MENU.
    if vision.find(screen, "close_x") is not None:
        return State.POPUP
    order = _BASE_ORDER
    if phase is not None and config.PHASE_CLASSIFY:
        order = PHASE_ORDER.get(phase, _BASE_ORDER)
    for name in order:
        if vision.find(screen, name, threshold=_anchor_threshold(name)) is not None:
            return _ANCHOR_STATE[name]
    return State.UNKNOWN


def is_trio_showdown_selected(screen: np.ndarray) -> bool:
    """True if the menu's mode banner shows Trio Showdown."""
    return vision.find(screen, "trio_showdown") is not None


def trio_showdown_score(screen: np.ndarray) -> tuple[bool, float]:
    """Like :func:`is_trio_showdown_selected` but also returns the template match
    score (0..1) — for honest wrong-mode diagnostics: the controller logs the score
    on a confirmed miss so a recurring false alarm is debuggable from the event log."""
    m, sc = vision.find_with_score(screen, "trio_showdown")
    return m is not None, sc


def is_other_device(screen: np.ndarray) -> bool:
    """True if the disconnect modal is the 'Another device is connecting' variant
    (vs the benign idle/bad-network one). Lets the controller treat them differently."""
    return vision.find(screen, "other_device", threshold=0.85) is not None


def is_daily_streak(screen: np.ndarray) -> bool:
    """True if the DAILY STREAK login-reward popup is showing — a daily, menu-blocking
    popup whose ✕ the close_x template misses (so it would otherwise stall the bot into a
    match_timeout loop). OCR its title banner for "STREAK" — NOT "DAILY", which also appears
    in the "DAILY WINS" banner on Star/Angel drops. Called only on UNKNOWN frames (the
    controller gates it) so this OCR stays off the hot path."""
    return vision.find_text(screen, "STREAK", region=config.DAILY_STREAK_TITLE_REGION) is not None


def is_team_invite(screen: np.ndarray) -> bool:
    """True if a TEAM INVITE modal is showing ("<player> wants to team up with you!").
    It blocks the menu and eats PLAY taps until it times out, so the controller declines
    it (MUTE + REJECT). OCR'd in a tight title band; like is_daily_streak this is only
    called off the hot path (UNKNOWN frames, or a menu that stopped accepting PLAY)."""
    return (
        vision.find_text(screen, "TEAM INVITE", region=config.TEAM_INVITE_TITLE_REGION) is not None
    )


def is_skin_popup(screen: np.ndarray) -> bool:
    """True if a skin-reward popup is showing ("NEW <rarity> SKIN!" with the green
    EQUIP NOW button). Appears mid-drop when a Starr/Chaos drop awards a skin — the
    drop tap-through must press CONTINUE here, not blind-tap (which could hit EQUIP)."""
    return vision.find(screen, "skin_popup") is not None


# --- Self-heal scenarios (r8): brawler-unlock ceremonies + in-match modals -------
#
# These are CLASSIFIED BY COLOR/STRUCTURE, not templates: the ceremony screens use
# stylized banner text that OCR mangles ("CHOOSE A BRAWLER" -> "ER"; "LET'S GO" /
# "GOT IT" unreadable), so the house rule (CONTRIBUTING.md) says prefer a color/structure
# gate. The reliable signal across every ceremony frame is the distinct GREEN call-
# to-action button (LET'S GO / GOT IT / CHOOSE) in the bottom band — a flat green
# H≈61, S/V high — that does NOT appear on the menu (PLAY is a different green),
# results screens, or in-match frames. OCR is the SECONDARY confirm only (the
# CHOOSE-a-brawler subtitle "Switch your choice anytime!" reads cleanly).
#
# Calibrated on tonight's real 1600x900 frames (saved ceremony screenshots):
#   green CTA HSV (50,120,120)-(70,255,255); bottom band y 780-870 split into thirds.
#   KAZE unlock -> right third 0.70 (LET'S GO); ULTRA TRAIT -> center 0.77 (GOT IT);
#   CHOOSE-selected -> center 0.57 (CHOOSE); CHOOSE-unselected/menu/results/match 0.00.

_CTA_BAND_Y = (780, 870)  # bottom band where the green CTA sits
_CTA_THIRDS = {  # name -> (x1, x2) and the tap point for that CTA
    "left": ((0, 533), (266, 822)),
    "center": ((533, 1066), (800, 822)),
    "right": ((1066, 1600), (1380, 822)),
}
_CTA_GREEN_LO = (50, 120, 120)
_CTA_GREEN_HI = (70, 255, 255)
_CTA_MIN_FRACTION = 0.30  # green fraction in a third above this => a CTA is there
# CHOOSE-A-BRAWLER title/subtitle band. The stylized title mangles, but the subtitle
# reads cleanly — either needle confirms the screen.
_CHOOSE_TITLE_REGION = (300, 10, 1300, 160)


def green_cta(screen: np.ndarray) -> tuple[int, int] | None:
    """Return the tap point of the single green call-to-action button in the bottom
    band (LET'S GO / GOT IT / CHOOSE), or None if no green CTA is present. Picks the
    third with the most green so a CHOOSE button beside a (blue) TRY still resolves to
    CHOOSE. Color-only — never matches the menu PLAY / results buttons (different hue)."""
    y1, y2 = _CTA_BAND_Y
    best_name, best_frac = None, 0.0
    for name, ((x1, x2), _tap) in _CTA_THIRDS.items():
        frac = vision.color_fraction(screen, (x1, y1, x2, y2), _CTA_GREEN_LO, _CTA_GREEN_HI)
        if frac > best_frac:
            best_name, best_frac = name, frac
    if best_name is None or best_frac < _CTA_MIN_FRACTION:
        return None
    return _CTA_THIRDS[best_name][1]


def is_choose_a_brawler(screen: np.ndarray) -> bool:
    """True if the "CHOOSE A BRAWLER" screen is up (the 3-card chooser shown after a
    brawler unlock / at the prestige reselect). OCR the title band for the stylized
    "CHOOSE" or the clean subtitle word "SWITCH" — either is a reliable, collapse-proof
    needle."""
    return (
        vision.find_text(screen, "CHOOSE", region=_CHOOSE_TITLE_REGION) is not None
        or vision.find_text(screen, "SWITCH", region=_CHOOSE_TITLE_REGION) is not None
    )


# In-match server-error modal (e.g. "Server error: 43") — a flat neutral-GRAY box
# centered over the match. The PLAYING phase otherwise never scans for modals, so it
# sat on this forever on a live farm. Gate on the low-saturation gray mass in the
# modal's body region — it scored 0.92 on the real frame vs ~0.16 on drops/match
# scenes. NOTE the disconnect/RELOAD modal is ALSO a gray box, but it carries the
# `reload` template and is classified DISCONNECT (and handled) BEFORE phase_playing
# is ever reached, so this gate only ever sees the no-reload error variant.
_MODAL_BODY_REGION = (560, 320, 1040, 480)
_MODAL_GRAY_SMAX = 40  # modal body is near-zero saturation
_MODAL_GRAY_VLO = 40  # exclude the pure-black HUD / map
_MODAL_GRAY_VHI = 110  # the modal's mid-gray fill
_MODAL_MIN_FRACTION = 0.50  # gray fraction above this => a modal is up (0.92 vs 0.16)


def in_match_modal(screen: np.ndarray) -> bool:
    """True if a gray dismissable modal (server-error etc.) is sitting over an
    in-match frame. Cheap structure gate (one masked region mean); see _MODAL_*."""
    x1, y1, x2, y2 = _MODAL_BODY_REGION
    crop = screen[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = (s <= _MODAL_GRAY_SMAX) & (v >= _MODAL_GRAY_VLO) & (v <= _MODAL_GRAY_VHI)
    return float(mask.mean()) >= _MODAL_MIN_FRACTION


# Forced shooting gallery (live on Pie64, 1600x900, 2026-11-05): a fullscreen screen a launch
# can open, cleared only by tapping its two cannons. Classified by color, not a template: the
# `exit` template is the results button and peaks at 0.48 here against 0.43 on the menu, so it
# has no valid threshold. Both gates must pass, so neither alone can fire:
#   wall: bright red across the top band. 0.82-1.00 on 12 gallery frames, 0.19 at most on 27
#   other frames and 574 captures. The dimmed arena behind CHOOSE A BRAWLER scores 0.00.
#   sign: the gold EXIT plate in the top left, in a box that holds it across the camera pan
#   (the plate moves about 30 px). 0.23-0.27 on the gallery. The results EXIT sits bottom right.
_GALLERY_WALL_REGION = (0, 0, 1600, 80)
_GALLERY_WALL_LO = (0, 150, 225)
_GALLERY_WALL_HI = (12, 255, 255)
_GALLERY_WALL_MIN_FRACTION = 0.50
_GALLERY_SIGN_REGION = (270, 70, 430, 170)
_GALLERY_SIGN_LO = (18, 100, 200)
_GALLERY_SIGN_HI = (32, 255, 255)
_GALLERY_SIGN_MIN_FRACTION = 0.15


def is_forced_gallery(screen: np.ndarray) -> bool:
    """True if the forced shooting gallery is showing. Color-only, so it is cheap enough for
    every UNKNOWN frame on the way back to the menu. See _GALLERY_*."""
    wall = vision.color_fraction(screen, _GALLERY_WALL_REGION, _GALLERY_WALL_LO, _GALLERY_WALL_HI)
    if wall < _GALLERY_WALL_MIN_FRACTION:
        return False
    sign = vision.color_fraction(screen, _GALLERY_SIGN_REGION, _GALLERY_SIGN_LO, _GALLERY_SIGN_HI)
    return sign >= _GALLERY_SIGN_MIN_FRACTION
