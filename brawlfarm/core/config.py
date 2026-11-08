"""
Central configuration for the Brawl Stars automation bot.

Everything that might change between machines, BlueStacks setups, or game updates
lives here so the rest of the code never hard-codes a path, port, or coordinate.

Coordinates assume the LOCKED display: 1600x900, DPI 240. The bot asserts this at
startup (see core/controller.py's run()) so a changed resolution can't silently
misfire taps.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -------------------------------------------------------------------
# HOME_DIR is where runtime state lives (data, captures, .env). The control panel sets
# BRAWLFARM_HOME to the user data directory; a bare checkout defaults to the current
# working directory so `uv run python -m brawlfarm.worker` behaves like the legacy layout.
# set_home() re-points every derived path at runtime (the supervisor and the test suite
# call it); modules must read these names at call time, never cache them at import.
PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"

HOME_DIR = Path(os.environ.get("BRAWLFARM_HOME", "") or Path.cwd()).resolve()
CAPTURES_DIR = HOME_DIR / "captures"
ONBOARD_SHOTS_DIR = CAPTURES_DIR / "onboard"  # step-by-step evidence screenshots
# Data dir is env-overridable so concurrent workers on different instances write to
# separate folders (one shared games.csv would race and corrupt). Pairs with
# BRAWL_ADB_PORT and BRAWL_PLAYER_TAG for multi-instance runs.
DATA_DIR = HOME_DIR / os.environ.get("BRAWL_DATA_DIR", "data")


def set_home(home: Path) -> None:
    """Point every home-derived path at ``home`` (resolved). DATA_DIR keeps honouring
    BRAWL_DATA_DIR so a worker process still lands in its own instance folder."""
    global HOME_DIR, CAPTURES_DIR, ONBOARD_SHOTS_DIR, DATA_DIR
    HOME_DIR = Path(home).resolve()
    CAPTURES_DIR = HOME_DIR / "captures"
    ONBOARD_SHOTS_DIR = CAPTURES_DIR / "onboard"
    DATA_DIR = HOME_DIR / os.environ.get("BRAWL_DATA_DIR", "data")
    global CALIBRATION_FILE, CALIBRATION
    CALIBRATION_FILE = HOME_DIR / "calibration" / "calibration.toml"
    CALIBRATION = _calibration.apply(globals(), CALIBRATION_FILE)


# BlueStacks ships its own adb; use it directly. Overridable for non-default installs.
ADB_PATH = os.environ.get("BRAWL_ADB_PATH", r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe")

# Brawl Stars Android package (for relaunch / freeze recovery).
BS_PACKAGE = "com.supercell.brawlstars"

# --- game client auto-update (2026-11-04) -------------------------------------------
# When a stuck run's first force-stop + relaunch does not help, recover() opens the Brawl
# Stars page in the Play Store ONCE per run. An "Update" button means a new client is out:
# tap it, wait for the installed versionName to change, relaunch. OCR-driven tap outside
# the game (safety rail: verify then act). Lines containing a STORE_NEVER_TAP word are
# never tapped; "Uninstall" sits next to "Update" on the store page.
PLAY_STORE_PACKAGE = "com.android.vending"
STORE_UPDATE_NEEDLE = "update"  # whole OCR line, case-insensitive
STORE_NEVER_TAP = ("uninstall",)
STORE_OPEN_TIMEOUT_S = 15  # wait for the store to reach the foreground
STORE_SETTLE_S = 3.0  # let the store page render before the screenshot
GAME_UPDATE_TIMEOUT_S = 600  # wait this long for the new version to install
GAME_UPDATE_POLL_S = 15
GAME_UPDATE_ON_ATTEMPT = 2  # recovery attempt that triggers the once-per-run check

load_dotenv(HOME_DIR / ".env")


# --- Farm instances ------------------------------------------------------------
# Populated at runtime by the supervisor from the user's settings (phase 2). Shape per
# entry: {"port": "5555", "tag": "#XXXXXXX" or "", "data": "data/<name>"}. The module
# ships EMPTY so the package never carries anyone's account identifiers.
INSTANCES: dict[str, dict[str, str]] = {}


def set_instances(mapping: dict[str, dict[str, str]]) -> None:
    """Replace the runtime instance table in place (callers hold references to the dict)."""
    INSTANCES.clear()
    INSTANCES.update({name: dict(entry) for name, entry in mapping.items()})


# --- ADB connection ----------------------------------------------------------

# Ports come from bluestacks.conf, one per instance, usually 5555, 5565, 5575, ...
# Port is env-overridable so a second process/instance can target another emulator
# (e.g. BRAWL_ADB_PORT=5565 for the second one) — needed for multi-instance work.
ADB_HOST = os.environ.get("BRAWL_ADB_HOST", "127.0.0.1")
ADB_PORT = int(os.environ.get("BRAWL_ADB_PORT", "5555"))
ADB_SERIAL = f"{ADB_HOST}:{ADB_PORT}"


# --- Display lock ------------------------------------------------------------

SCREEN_W = 1600
SCREEN_H = 900
SCREEN_DPI = 240


# --- Game launch -------------------------------------------------------------
# After a fresh emulator boot the instance sits on the BlueStacks launcher home,
# not in the game. The bot opens it the human way: OCR the app's text label and TAP
# the icon centred a little above it (the icon sits above its label). If the label
# isn't visible it falls back to `adb.launch_app()` (monkey, coordinate-free).
GAME_LABEL = "Brawl Stars"  # the launcher-home label to OCR + tap
GAME_ICON_LABEL_DY = 80  # icon centre is ~this many px above its label
GAME_LAUNCH_TIMEOUT = 45  # seconds to wait for the game to foreground


# --- Fast input --------------------------------------------------------------
# `adb shell input tap` spawns a JVM per call (~700 ms + CPU churn that causes adb
# timeouts under multi-instance load). Writing the raw touch events straight to the
# input device node in ONE shell call is ~70 ms and uses no JVM — a ~10x speedup
# (measured on BlueStacks). core/adb.py uses this when FAST_INPUT is on and silently
# falls back to `adb shell input` if a raw write fails. Verified device: BlueStacks
# "Virtual Touch" — a type-A multitouch, position-only node (ABS_MT_POSITION_X/Y only),
# coordinate range 0..TOUCH_MAX. If a different emulator/device misbehaves (taps do
# nothing), set BRAWL_FAST_INPUT=0 to force the old path.
FAST_INPUT = os.environ.get("BRAWL_FAST_INPUT", "1") != "0"
TOUCH_DEVICE = "/dev/input/event4"  # the "BlueStacks Virtual Touch" node
TOUCH_MAX_X = 32767  # ABS_MT_POSITION_X max (taps scale pixel_x/SCREEN_W*this)
TOUCH_MAX_Y = 32767  # ABS_MT_POSITION_Y max


# --- Fast capture --------------------------------------------------------------
# `exec-out screencap -p` makes the DEVICE encode a PNG (~790 KB) that the HOST then
# cv2.imdecode's (~290 ms of pure CPU). Dropping `-p` streams the raw RGBA framebuffer
# (16-byte header + w*h*4 bytes) — a bigger transfer but NO encode and NO decode, ~200 ms
# faster per capture (569 -> 370 ms median, measured on this machine; see the legacy
# research note "performance optimization", item 4, not ported). core/adb.py uses this
# when RAW_CAP is on and silently falls back to the PNG path on any failure (and
# PERMANENTLY for the session if the device reports a non-RGBA_8888 raw format we can't
# parse). Set BRAWL_RAW_CAP=0 to force the old PNG path.
# Re-measured 2026-09-10 on the development machine: 172 ms median with one instance and
# no worker running; see docs/notes/2026-09-10-frame-capture-assessment.md.
RAW_CAP = os.environ.get("BRAWL_RAW_CAP", "1") != "0"


# --- Brawl Stars API ---------------------------------------------------------

API_BASE = "https://api.brawlstars.com/v1"
PLAYER_TAG = os.environ.get("BRAWL_PLAYER_TAG", "")
API_TOKEN = os.environ.get("BRAWL_API_TOKEN", "")  # loaded from .env; never hard-code

# How often (seconds) we're willing to hit the API for trophy snapshots / battlelog.
API_TROPHY_MIN_INTERVAL = 60.0
API_BATTLELOG_MIN_INTERVAL = 30.0

# Substrings that identify our farming mode in battlelog (matched case-insensitively).
SHOWDOWN_MODE_KEYS = ("showdown",)


# --- Timings (seconds) -------------------------------------------------------

ATTACK_INTERVAL = 2.75  # BASE seconds between attack taps (also used by tools/harvest_match)
# Anti-detection: a perfectly periodic tap is an obvious bot tell, so we jitter each attack
# gap by +/- this many seconds (so the real gap is uniform in ATTACK_INTERVAL +/- jitter).
ATTACK_JITTER = 0.9  # 2.75 +/- 0.9 => ~1.85-3.65 s between attacks
TAP_JITTER_PX = 7  # nudge in-match tap coords by +/- this many px (don't hit the same pixel)
LOOP_POLL_INTERVAL = 0.20  # base delay between capture->classify iterations
TAP_SETTLE = 0.6  # small wait after a navigation tap before re-capturing
# Drops/reward reveals open on RAPID taps; clear them fast so more time is spent in
# matches (the real goal is trophies, not watching reward animations). Each returning
# iteration fires a BURST of taps (instead of one) so the rate isn't capped by the
# screencap+poll cost. Tunable DOWN until the inputs blur through the animation; back UP
# only if drops start getting skipped/mis-handled.
DROP_TAP_INTERVAL = 0.06  # delay between taps within a burst (was 0.12; halved)
DROP_TAP_BURST = 4  # taps per returning iteration while clearing a drop
# Hold-drops (Angel / Demon / Nova) ONLY open on a press-and-HOLD — tap-only loops forever
# on them — but they announce themselves with "TAP AND HOLD!" centred near the bottom
# (measured ~800,802). So while clearing a drop we OCR this band for "HOLD" and press-and-
# hold when we see it (signal-driven, beats blindly holding on a fixed cadence). Monster
# Eggs still need a SWIPE (handled on an occasional step).
DROP_HOLD_TEXT_REGION = (300, 720, 1300, 880)  # band around the "TAP AND HOLD!" prompt
# (kept generous: a tight crop garbles OCR — the detector needs breathing room around it)
DROP_HOLD_MS = 2500  # press-and-hold long enough to pop a hold-drop
DROP_OCR_EVERY = 3  # OCR for the hold prompt every Nth drop-clear step (OCR is costly)

# Freeze detection: if the frame is ~identical this long while we expect motion.
FREEZE_TIMEOUT = 45.0


# --- Vision tuning -----------------------------------------------------------

# Grayscale template matching: TM_CCOEFF_NORMED on 3-channel BGR does the correlation
# per channel — ~3x the work for no benefit on high-luminance-contrast UI anchors.
# Matching template + screen in grayscale is ~4.7x faster per find (99 -> 21 ms measured;
# the legacy research note "performance optimization", item 1, not ported). Set
# BRAWL_GRAY_MATCH=0 to force color.
#
# Decision-equivalence was validated on the full capture corpus (562 real 1600x900
# frames, tools/validate_gray_thresholds.py): with the COLOR_ONLY_TEMPLATES carve-out
# below, every frame classifies IDENTICALLY to the original color classifier, under
# every phase hint. Per-template at the thresholds in use, gray vs color flipped the
# >=threshold decision on ZERO frames for: reload, play*, playagain*, proceed, exit,
# teams_left*, trophy_screen, trophy_brawler, trio_showdown, other_device, skin_popup.
# (* = score crossed the threshold on 1-4 frames, but only where an earlier check
# (reload/close_x/another RESULTS anchor) already decides the State — gray scored
# HIGHER there, e.g. playagain behind a disconnect modal 0.83 color -> 1.00 gray, so
# nothing is ever missed.)
GRAY_MATCH = os.environ.get("BRAWL_GRAY_MATCH", "1") != "0"

# Templates that must KEEP matching in color even when GRAY_MATCH is on — for these
# the corpus shows grayscale has NO valid threshold (true/false score ranges overlap),
# so going gray would change real decisions:
#   close_x      color separates by a hair at 0.85 (min-true 0.851 / max-false 0.845);
#                gray INVERTS it (min-true 0.876 / max-false 0.897) -> 15/562 frames flip.
# matchmaking left this set in phase 8: the re-shot "Players found N/12" template
# separates cleanly in gray (see MATCHMAKING_THRESHOLD).
COLOR_ONLY_TEMPLATES = frozenset({"close_x"})

# Phase-aware classification: the controller passes its phase to states.classify so the
# EXPECTED anchor is template-matched first (find returns on first hit, so order =
# expected-case latency). Orders are pure REORDERS of the full anchor chain — never
# subsets — and were corpus-validated to classify every frame identically (see
# states.PHASE_ORDER for why subsets/matchmaking-first are unsafe). Set
# BRAWL_PHASE_CLASSIFY=0 to ignore the hint and always run the base order.
PHASE_CLASSIFY = os.environ.get("BRAWL_PHASE_CLASSIFY", "1") != "0"

MATCH_THRESHOLD = 0.85  # default template-match confidence (0..1)
IN_MATCH_THRESHOLD = 0.70  # "Teams left:" HUD match (lower: text over varying maps)
# "Players found N/12" line in the queue screen (276x45), gray-matched. Measured on the
# 2026-09 corpus: queue frames at or above 0.99, every other frame at or below 0.51.
MATCHMAKING_THRESHOLD = 0.85


# --- Behavior flags ----------------------------------------------------------

# Close the game on every graceful session stop (scheduler sleep/caps, /stop,
# stop.flag) so the account goes genuinely OFFLINE between sessions — Supercell
# sees online/session time, and "menu-parked 20 h a day" defeats the scheduler's
# human-shaped play-days (owner, 2026-06-10). The next worker reopens the game
# via ensure_game_open(). BRAWL_CLOSE_GAME_ON_STOP=0 restores leave-at-menu.
CLOSE_GAME_ON_STOP = os.environ.get("BRAWL_CLOSE_GAME_ON_STOP", "1") != "0"

# "Another device is connecting to this game" modal. NOW (single account, dev,
# you have PC access): reload and keep farming. LATER (multi-account / remote
# control): set False so it ABORTS + alerts instead of reconnecting.
RECONNECT_ON_OTHER_DEVICE = True

# Mid-session brawler switch: the farm grinds the lowest-trophy brawler, but once it climbs
# past this many trophies it's no longer the most efficient pick (and hits the 1000 rank
# milestone), so we re-select the new lowest brawler. Trigger is API-driven (we watch the
# farmed brawler's per-brawler trophies via get_player), independent of fragile screen reads.
# NOTE: this is only the controller's STARTING goal — every trophy snapshot re-evaluates
# the farm plan's live goal (ladder's moving step floor; see core/farmplan.choose_target).
RESELECT_TROPHY_THRESHOLD = 1000

# --- ladder rate-based rotation (r6: folded in from the old "optimal" mode) -----
# These names predate r6 (calibration constants, kept stable) but now drive LADDER
# rotation: selection = ladder's lowest-first within the rising floor, PLUS rotate
# OFF the farmed brawler for the rest of the session when its trailing net trophy
# rate is subnormal AND it is not on a winstreak ("if it's been running a while
# with little improvement it should switch — but never mid-winstreak", r6 owner
# rule; the winstreak guard lives in farmplan.recent_winstreak). Calibrated
# 2026-06-10 against three live farms' games.csv (~1,800 showdown games; the three
# accounts are labelled A, B and C in the notes below — A and B sit in the low
# ladder bands, C is near-maxed):
#   * mean trophyChange/game: A +6.6, B +6.2 (ladder bands), C +2.3 (high bands)
#   * a 10-game trailing mean < 0 occurred in 0% (B) / 1.3% (A) of windows —
#     normal variance essentially never sustains a NET LOSS over 10 games
#   * C's genuinely stagnant high-band brawlers (GIGI/OTIS/FINX) sat below 0 for
#     5-8 consecutive windows — exactly the stagnation the rotation must catch
# So: rotate when the last OPTIMAL_RATE_WINDOW games on the farmed brawler average
# below OPTIMAL_RATE_MIN trophies/game ("the brawler is net-losing"). Fewer logged
# games than the window = not enough evidence, never rotate. Rotation state lives
# in controller memory (session-scoped, reset on worker restart) — a transient
# cold streak never sticks to the plan file across sessions.
OPTIMAL_RATE_WINDOW = 10  # trailing games.csv games to judge the farmed brawler on
OPTIMAL_RATE_MIN = 0.0  # mean trophyChange/game below this => rotate it out

# Opportunity-cost rotation kill-switch (owner decision 2026-06-14). The reactive
# rotation has two triggers (farmplan.rotation_decision): "absolute_floor" (the
# farmed brawler is net-LOSING — a genuine stagnation signal, kept ON) and
# "opportunity_cost" (a marginally-hotter in-band alternative exists). On the
# near-maxed account C the latter fired on EVERY freshly-selected brawler —
# often within the same second, before it played a single game — because the
# lowest brawlers (all ~800-1000) have near-identical rates, so something always
# scores marginally higher. Result: the worker thrashed in the brawler menu
# (32 reselects / 10 h, 30 of them opportunity_cost) and made no progress. The
# owner's call: switch OFF opportunity_cost, keep absolute_floor. Default OFF;
# re-enable with BRAWL_WINRATE_OPPORTUNITY_COST=1 (no deploy needed).
WINRATE_OPPORTUNITY_COST = os.environ.get("BRAWL_WINRATE_OPPORTUNITY_COST", "0") != "0"

# --- Win-rate-aware selection (Q2, legacy planning note "winrate-aware farming", not ported) ---
# PROACTIVE companion to the reactive rotation above: at ladder selection
# points, farmplan re-orders the LEGAL candidates (owned brawlers below the current
# step goal — so the ladder floor invariant is untouched by construction) by their
# trailing net trophy rate, and promotes the top one only when it beats the
# roster-minimum brawler's score by >= WINRATE_MARGIN trophies/game. Margin +
# min-sample + neutral-prior are the anti-churn rails: a brawler with fewer than
# WINRATE_MIN_GAMES logged showdown games scores the account-wide trailing mean
# (neither favored nor punished), so a fresh/missing/garbled games.csv degrades
# bit-identically to today's lowest-trophy flow. Prestige mode never consults any
# of this (its pick is a goal statement, never overridden).
# Offline replay vs the live farms' games.csv + live API rosters (2026-06-10,
# table in PR evidence): account C (ladder, step 700, 9 in-band candidates) promotes
# OTIS (+3.25/game over 20 logged games) over roster-min DOUG (8 games -> the
# +1.90 prior), margin +1.35; accounts A and B change NOTHING — every in-band
# candidate there is under the 10-game min-sample (fresh 0-100 bands), all
# score the prior, and the default lowest-trophy flow holds. Exactly the
# designed split: promote on real evidence, fall through on cold start.
# Kill-switch: BRAWL_WINRATE_AWARE=0 pins today's selection without a deploy
# (FAST_INPUT-style convention). Ships ON: the margin gate keeps "on"
# conservative until the data clearly favors a candidate.
WINRATE_AWARE = os.environ.get("BRAWL_WINRATE_AWARE", "1") != "0"
WINRATE_WINDOW = 20  # v1 trailing window (kept for the v1-vs-v2 replay/back-compat)
WINRATE_MIN_GAMES = 10  # v1 min-sample (kept for the replay); v2 uses shrinkage
WINRATE_MARGIN = 1.0  # promote only when top beats the roster-minimum by this

# --- Win-rate model v2 (r8, legacy research note "winrate model", not ported) --
# v2 keeps the surface (WINRATE_AWARE kill-switch, WINRATE_MARGIN gate, the
# ladder-only by-name promotion) but upgrades the STATISTICS the score is built on:
#
#   (a) Empirical-Bayes SHRINKAGE — a brawler's score is pulled toward the
#       account's own overall mean trophyChange/game (mu0, computed from games.csv)
#       with prior strength WINRATE_PRIOR_K games:  (n*ewma + k*mu0)/(n+k). This
#       replaces v1's hard min-sample/neutral-prior CLIFF (9 games -> the prior,
#       10 games -> the raw mean) with a smooth pull, killing small-sample
#       overreaction in BOTH the pick bias and the rotation trigger. k=10 means a
#       brawler needs ~10 effective games before its own rate outweighs the prior.
#   (b) Recency-weighted EWMA — an exponentially-weighted mean (half-life
#       WINRATE_HALFLIFE games) over the brawler's history instead of a hard
#       window, so form changes register without a cliff at the window edge. The
#       effective sample size (sum of EWMA weights) saturates near
#       HALFLIFE/ln2 (~36 games at hl=25), so a long cold tail can't dominate.
#
# Constants are unitless trophies/game (same scale as OPTIMAL_RATE_*). Calibrated
# by reasoning + an offline replay against the live farms' games.csv (table in the
# PR / the model doc): k=10 and hl=25 keep "on" conservative — A/B (fresh low
# bands) still fall through to the lowest-trophy default; C's well-sampled
# high-band gaps (OTIS/GIGI/FINX) are the cases the shrinkage lets through on real
# evidence. NOT YET LIVE-VALIDATED — watch the first sessions; BRAWL_WINRATE_AWARE=0
# pins v1-free behavior (the whole feature) off without a deploy.
WINRATE_PRIOR_K = float(
    os.environ.get("BRAWL_WINRATE_PRIOR_K", "10")
)  # empirical-Bayes prior strength, in games
WINRATE_HALFLIFE = float(os.environ.get("BRAWL_WINRATE_HALFLIFE", "25"))  # EWMA half-life, in games
WINRATE_HISTORY_MAX = 200  # per-brawler games kept for the EWMA (older tail negligible)


# --- Coordinates -------------------------------------------------------------
# (x, y) tap targets, verified against live 1600x900 captures.

# Main menu
PLAY_BUTTON = (1434, 830)  # verified, confidence 1.0
# "Trio Showdown selected?" is checked via templates/trio_showdown.png (the banner).
# The mode banner (left of PLAY) opens the event/mode selector when tapped. Used to
# self-correct the mode (navigate to Trio Showdown) instead of just stopping.
MODE_BANNER = (950, 818)
# Round 6 (legacy owner-instructions note, not ported): a first "Trio not selected"
# miss is re-checked after this wait + a fresh capture — the banner is often just
# mid-animation, so a single transient miss must NOT raise a wrong_mode alarm. Only
# BOTH frames missing is a confirmed wrong-mode.
MODE_VERIFY_RECHECK_S = 1.7
# The event picker gained category tabs (measured on Pie64, 1600x900, 2026-09-17):
# it opens on SPECIAL EVENTS, and Showdown now lives under the TROPHIES tab. Cards the
# account has never opened wear a one-time "NEW!" cover that hides the card name from
# OCR until it is tapped once (peeling it selects nothing). Every value here is a
# needle, a y cut-off or a wait: the new taps are all centers of labels OCR just found.
MODE_TAB_TROPHIES = "TROPHIES"  # bottom tab bar, exact OCR match
MODE_NEW_COVER = "NEW!"  # a card's one-time cover; the tab-bar badges read "NEW"
MODE_CARD_AREA_MAX_Y = 800  # covers live above this; below it is the tab bar
MODE_TAB_WAIT_S = 3.0  # the trophy mode cards slide in after the tab tap
MODE_COVER_WAIT_S = 1.2  # between cover peels
MODE_CARD_WAIT_S = 2.5  # card expands in place / returns to the menu (1.8 was too short)

# In-match. This account has NO gadget/hypercharge unlocked on the farm brawler,
# and Super rarely charges from passive attacking, so in practice this is just the
# attack tap. We also tap Super (harmless no-op when uncharged) in case it charges.
ATTACK_POINT = (1177, 637)  # red crosshair attack button (auto-aims nearest)
SUPER_BUTTON = (1252, 734)  # skull super button

# In-match movement: swipe the left-side joystick so we register activity (attack
# taps alone don't prevent the idle-disconnect) and roam instead of standing still.
# Anti-detection: instead of a fixed-cadence CLOCKWISE octagon (a dead-giveaway pattern),
# we wander a continuously-drifting heading on a randomized cadence — no repeating period,
# no repeating direction sequence.
MOVE_ORIGIN = (230, 650)  # joystick centre; the swipe vector sets the move direction
MOVE_RADIUS = 180  # joystick swipe magnitude (jittered 0.75-1.0x per move)
MOVE_DURATION_MS = 480  # base swipe duration (jittered 0.8-1.25x per move)
# Each move the heading turns by a random amount up to +/- this (radians, ~92°) — small
# turns = smooth human-ish wandering, big turns = a course change, never a fixed cycle.
MOVE_TURN_MAX = 1.6
# Randomized gap between moves. Kept WELL under the ~30 s idle-disconnect timer at both ends
# so a long gap can't get us kicked, but spread enough that the cadence isn't periodic.
MOVE_INTERVAL_MIN = 1.0
MOVE_INTERVAL_MAX = 2.4

# Disconnect / "kicked for inactivity" modal: RELOAD button = rejoin the battle.
RELOAD_BUTTON = (459, 524)

# Promo / news / offer popups (e.g. "Formula Brawl starts now!") — close via the
# red ✕ top-right of the popup card. The `close_x` template detects this whole class.
CLOSE_X_BUTTON = (1400, 80)

# Results screens. There are TWO: screen 1 has PLAY AGAIN + PROCEED (battle stats);
# tapping PROCEED leads to screen 2 with PLAY AGAIN + EXIT (trophy progress).
# We advance PROCEED -> EXIT -> menu.
PLAY_AGAIN_BUTTON = (1090, 832)
PROCEED_BUTTON = (1426, 832)
EXIT_BUTTON = (1404, 832)

# Skin reward popup ("NEW <rarity> SKIN!" with CONTINUE / EQUIP NOW) — appears mid-drop
# when a Starr/Chaos drop awards a skin. Tap CONTINUE (left, blue) to proceed; NEVER
# EQUIP NOW (right, green), which would change the active skin.
CONTINUE_BUTTON = (929, 772)

# Generic popup handling (clear reward / Star Drop popups on the way back to menu)
SAFE_DISMISS_POINT = (800, 200)  # top-center tap to continue (avoids buttons)
SAFE_HOLD_POINT = (800, 450)  # screen center, for hold-to-open Star Drops

# --- Self-heal scenarios (r8) --------------------------------------------------
# Brawler-unlock CEREMONY screens (live incident on a farm account, 2026-06-11: a fullscreen KAZE
# unlock recover-looped for hours). Classified by the bottom-band GREEN call-to-action
# button + the CHOOSE-A-BRAWLER title (see states.green_cta / is_choose_a_brawler).
# CHOOSE A BRAWLER: tap the CENTER card to pick, then the green CHOOSE button confirms.
# NEVER tap the blue TRY button beside CHOOSE (it enters a match preview).
CHOOSE_BRAWLER_CENTER_CARD = (800, 480)  # center brawler card on the 3-card chooser
CHOOSE_BRAWLER_CONFIRM = (984, 822)  # green CHOOSE button (left of the blue TRY)
CEREMONY_MAX_SCREENS = 4  # max ceremony screens cleared per recovery cycle, then fall through

# In-match server-error modal (live incident on a farm account, 2026-06-11: a gray
# "Server error: 43" box sat over the match — the PLAYING phase never scanned for modals).
# Detect via the gray-box gate (states.in_match_modal) and tap OK (bottom-left of the
# modal) to dismiss; the game returns to menu/launcher and the existing recovery takes over.
INGAME_MODAL_OK_BUTTON = (440, 520)  # OK button on the server-error modal
INGAME_MODAL_CHECK_EVERY = 12  # scan for the modal every Nth in-match playing iteration

# Spectate camera (Trio Showdown, the whole team is out): an "Exit" button sits where the
# controls were. Not the results screen's EXIT_BUTTON. The point is the centre of the play
# reader's EXIT_LABEL_BOX (777, 822, 822, 843). Read by brawlfarm.play.reader.spectating,
# which is False while the respawn "Back in:" countdown is up (a teammate still lives).
SPECTATE_EXIT_BUTTON = (799, 832)
SPECTATE_CONFIRM_TICKS = 2  # consecutive spectating in-match ticks before the first tap
SPECTATE_RETAP_TICKS = 3  # in-match ticks a tap gets to move off the spectate screen
SPECTATE_MAX_TAPS = 3  # taps per match (the first plus two retries), then wait it out

# Top-right HOME (white house) button. Returns to the main menu from sub-screens like the
# brawler TROPHIES detail / Trophy Road (auto-shown at rank-up milestones). We exit those
# via HOME — never a center-tap (re-opens the brawler) or BACK (project rule: taps only).
HOME_BUTTON = (1525, 47)  # calibrated on the brawler trophy screen (white house icon)

# DAILY STREAK login-reward popup. Auto-appears once a day, BLOCKS the menu, and its ✕ is
# NOT matched by the close_x template — worse, it survives the game-relaunch recovery, so an
# unhandled one becomes a match_timeout loop (~6 min each). We CLAIM it (free reward; claiming
# clears it for the day, whereas ✕ can re-pop) — a Starr-Drop reveal follows, cleared by the
# normal returning drop flow. Detect by OCR'ing "STREAK" (NOT "DAILY" — drops show a "DAILY
# WINS" banner that would false-match). Calibrated live on a farm account.
DAILY_STREAK_TITLE_REGION = (380, 90, 1080, 270)  # banner band — OCR "STREAK" here
DAILY_STREAK_CLAIM = (1331, 703)  # green CLAIM button
# Not every variant has a CLAIM button (2026-11-04: the day-1 "It's time to Brawl!" popup
# has none, and a blind tap on DAILY_STREAK_CLAIM hit empty background 1442 times). So the
# handler verifies first: OCR "CLAIM" around the button, else the popup's own red ✕ (which
# scored 0.853 on a capture, too close to the global 0.85 to trust, hence the local
# threshold, accepted only in the top-right area so a weak match elsewhere is never
# tapped), else one Android BACK.
DAILY_STREAK_CLAIM_REGION = (1130, 620, 1530, 790)  # OCR "CLAIM" here (around the button)
DAILY_STREAK_CLOSE_X_REGION = (1420, 100, 1600, 240)  # the ✕ centre must land in here
DAILY_STREAK_CLOSE_X_THRESHOLD = 0.80  # local close_x threshold, this popup only

# Forced shooting gallery (live on Pie64, 1600x900, 2026-11-05): a launch opened a fullscreen
# gallery with two "TAP!" prompts on cannons. Its EXIT sign is dead while the screen is up
# (pixel diff 0.0 after a tap). About 10 taps on the cannons cleared it: a purple beam fired,
# a gem shard dropped, and it did not return on the next launch. See states.is_forced_gallery.
GALLERY_CANNON_LEFT = (200, 620)
GALLERY_CANNON_RIGHT = (1400, 620)
GALLERY_MAX_ROUNDS = 30  # cannon taps per session, three times the live count, then fall through

# Repeat guard for the controller's UNKNOWN-path handlers (daily streak, team invite):
# this many consecutive loop iterations taking the SAME handler branch escalate to
# recover("<name>_stuck") instead of running the handler again.
HANDLER_REPEAT_MAX = 4

# --- Brawler selection -------------------------------------------------------
# Open the BRAWLERS screen from the menu (left-side button), sort by "Least Trophies",
# turn the Quest + Heart/Favorites filter toggles OFF (when ON they hide most brawlers, so
# the top-left card would be the wrong one), then select the top-left (lowest-trophy) brawler
# so the farm plays it. Like the shop/pass actions: open from menu -> act -> exit via taps.
# ⚠️ Coordinates calibrated live on 5585 (1600x900); the brawler UI can shift between updates.
BRAWLERS_BUTTON = (110, 415)  # left-side menu BRAWLERS button (label OCR'd ~85,429)
# was (267, 93, 550, 146), which cut the word down to "RS108/108"; "BRAWLERS 108/108" text
# about (150,105,420,135) on capture-20261107 frames 01779 and 01780
BRAWLERS_HEADER_REGION = (100, 90, 560, 150)  # OCR "BRAWLERS" here to confirm the screen is open

# was (1010, 42); "Least Trophies" text box (627,32,795,56) on 0006, "Name" centres on the
# same x on 0066
BRAWLER_SORT_LABEL = (711, 44)  # current-sort label, top-centre; tap to open the dropdown
# was (820, 80, 1200, 720); open panel on 0014 and 0077; items centre x 705, pitch 67;
# "Least Trophies" (705,359), "Name" (705,628)
BRAWLER_SORT_DROPDOWN_REGION = (311, 98, 1109, 815)  # OCR the sort items here

# Quest + Heart/Favorites filter toggles (top-right). ON shows an ORANGE check badge; OFF
# none. The Quest clipboard's own check is BLUE (always present), so we gate on ORANGE only.
# was (1215, 55); clipboard icon box (934,14,986,64) on 0006
BRAWLER_QUEST_TOGGLE = (960, 39)  # clipboard toggle (tap point also covers its badge)
# was (1360, 55); heart glyph box (1095,19,1148,61) on 0006
BRAWLER_HEART_TOGGLE = (1121, 40)  # heart toggle (between its orange badge and the icon)
# new, never-tap landmark; field box about (1160,18,1432,72) on 0006
BRAWLER_SEARCH_FIELD = (1296, 45)  # top-bar search field: a landmark only, never tapped
# UNVERIFIED: neither toggle was ON in session 1 (20260916-212232) or session 2
# (20260916-223234); the clipboard's own blue tick is icon art, never the ON test
BRAWLER_TOGGLE_ORANGE_LO = (5, 140, 140)
BRAWLER_TOGGLE_ORANGE_HI = (22, 255, 255)
BRAWLER_TOGGLE_ON_FRAC = 0.07  # orange fraction in a toggle's box above this => it's ON
BRAWLER_TOGGLE_HALF = 45  # half-size of the box sampled around each toggle's center

# was (457, 255), now the green strip beside the art; top-left card art at the resting
# scroll offset, x 127 to 420, y 155 to 335 on capture-20261107 frame 01779
BRAWLER_FIRST_CARD = (273, 255)  # top-left grid card art center = lowest after the sort
# Brawler detail screen: the blue SELECT button sits at a FIXED bottom-left spot, far from
# the TRY button and the bottom-right UPGRADE button (Gems/Coins, never tapped). OCR mangles
# the styled label ("SELET"), so we tap the fixed coordinate; tapping is idempotent (an
# already-active brawler shows a greyed "SELECTED" that's a harmless no-op).
BRAWLER_SELECT_BUTTON = (213, 818)  # confirmed on session 2 (measured (216, 816) on 0148)
BRAWLER_NAME_REGION = (
    0,
    0,
    520,
    250,
)  # OCR the brawler name here for the log/return value

# Select-BY-NAME (farm plan): sort the grid alphabetically ("Name" in the sort dropdown,
# where digits sort first, e.g. 8-BIT leads), then scroll to the target and tap it. The
# grid is HORIZONTAL and column-major: 3 rows per column, cards read top to bottom down a
# column and then on to the next one, so walking forward through the names means swiping
# LEFT. Geometry measured on the session-2 observe recording (Pie64/20260916-223234,
# 1600x900, frames 0015 and 0020). Swipes carry fling inertia, so selection runs a
# SELF-CORRECTING loop (OCR the visible card names, estimate the column offset, swipe,
# re-check) instead of trusting one big blind scroll. Owned brawlers list alphabetically
# BEFORE unowned (BELLE/BERRY, unowned, were skipped between BEA and BIBI on a 51/103
# account).
# The red shop offer panel holds x 0 to 125 (frames 0006 and 0015, red to x 116) with a live
# buy control: never-tap, and no swipe endpoint may land below x 146.
BRAWLER_SORT_NAME_ITEM = "Name"  # the alphabetical entry in the sort dropdown
# replaces BRAWLER_GRID_COLS_X; column centres 457, 837, 1217, 1597 (the 4th is half cut)
BRAWLER_GRID_COL0_X = 457  # card-center x of the leftmost column
BRAWLER_GRID_COL_W = 380  # column pitch; column lefts 277, 657, 1037, 1417 on 0015
# replaces BRAWLER_GRID_ROW0_Y and BRAWLER_GRID_ROW_H; row bands (147,364), (392,610),
# (637,853) on 0015
BRAWLER_GRID_ROWS_Y = (255, 501, 745)  # card-center y of the 3 rows
# new; card spans 657 to 1017 on 0015; a column counts as visible only when its centre plus
# half this fits inside BRAWLER_GRID_REGION
BRAWLER_CARD_W = 361
BRAWLER_CARD_H = 217  # new; row 1 spans 147 to 364 on 0015
# was (150, 80, 1430, 880); OCR band past the offer panel (red ends at x 116 on 0006) and
# below the top bar (ends y 72)
BRAWLER_GRID_REGION = (146, 140, 1600, 860)  # OCR band: the visible grid
# was absent; the label is right-aligned in the card, centre is 53 (ANGELO) to 79 (BEA)
# right of the card centre on 0020, so snap to the nearest column after subtracting this
BRAWLER_NAME_LABEL_DX = 66
# was 63; label band is card top + 144 to card top + 182, centre 55 below the card centre
# on 0020
BRAWLER_NAME_LABEL_DY = 55  # name label sits ~this far BELOW its card's center
# new; middle row centre, the lane x 500 to 1400 crosses card art and gaps only on 0015;
# y 378 (the gap between rows 1 and 2) is the control-free alternative
BRAWLER_SCROLL_Y = 501
BRAWLER_SCROLL_X_RIGHT = 1400  # new; swipe-left start, swipe-right end
# new; swipe-left end, swipe-right start; the offer panel ends at x 116 on 0006, never
# swipe below x 146
BRAWLER_SCROLL_X_LEFT = 500
BRAWLER_SCROLL_MS = 600  # new; duration of one grid swipe
# was 14; 105 owned brawlers make 35 columns and a 900 px swipe moves about 2.4 columns
BRAWLER_SELECT_MAX_SWIPES = 20  # bail-out budget for the find-the-card loop

# --- Mega quests -------------------------------------------------------------
# The QUESTS button sits in the bottom menu bar. When a NEW MEGA QUEST is available to
# activate it turns GOLD; once one is active (or none remain) it's gray — so a gold
# fraction over its face is the "should I activate a mega quest?" gate (measured live:
# ~0.36 gold on the low-band accounts = available, 0.00 on the near-maxed one = none).
# Activating = open QUESTS, tap the yellow NEW MEGA QUEST card on the left (one tap
# activates it directly, no confirm), then leave via the top-left back arrow. Activating
# clears the gold, so it won't re-fire.
QUESTS_BUTTON = (340, 852)  # bottom-bar QUESTS clipboard (tap to open)
QUESTS_NEW_REGION = (285, 762, 395, 872)  # button face — gold here => a new mega quest
QUESTS_NEW_FRAC = 0.15  # gold fraction above this => "new mega quest available"
QUESTS_GOLD_HSV_LO = (18, 90, 120)  # same gold as the pass tiles
QUESTS_GOLD_HSV_HI = (38, 255, 255)
QUESTS_TITLE_REGION = (90, 0, 520, 80)  # OCR "QUESTS" title here to confirm the screen
QUESTS_MEGA_CARD = (312, 540)  # the "NEW MEGA QUEST" card on the left (tap to activate)
QUESTS_MEGA_CARD_REGION = (
    180,
    430,
    450,
    660,
)  # gold here => NEW MEGA QUEST card present
QUESTS_CLOSE_BUTTON = (55, 52)  # top-left back arrow -> menu (NO BACK key)

# --- Quest grid --------------------------------------------------------------
# Where the three rows of quest cards sit, so questpick.group_cards can bucket one OCR
# pass into cards: a line's cy picks its row band, its cx picks its card in the row.
# Measured on the full-size observe recording Pie64 20260916-223234 (1600x900), frames
# 0118 and 0121 (the left edge), 0126 and 0133 (season quests) and 0140 (placeholder
# cards). They replace the provisional values doubled from the half-size recording.
# row bands (235,424), (448,638), (661,851) on 0118 and 0126, 10 px margin
QUEST_LIST_REGION = (0, 225, 1600, 860)
QUEST_ROW_PITCH = 213  # row tops 235, 448, 661 on 0118
# row 1 title text runs y 262 to 345 on 0126 (one line 295-311, two 273-325, three 267-339)
QUEST_TITLE_BAND0 = (250, 356)
# row 1 progress text y 371 to 412 on 0118 and 0126, the bar y 380 to 411
QUEST_PROGRESS_BAND0 = (362, 420)
# card lefts 597 and 1067 on 0126, card width 440, in-group gap 33; groups sit 142 further apart
QUEST_CARD_PITCH_X = 471
QUEST_CARD_X_TOL = 150  # one card's title lines share a centre within a few px
# on 0118 the cut season card starts at x 1432, centre about 1652, so anything past 1420
# is a cut column
QUEST_EDGE_MARGIN_X = 180

# The sweep that pages the grid sideways to read every card: swipes ONLY, never a tap
# inside the grid (a stationary tap opens a quest, a drag doesn't).
# 22 px below the row 2 card top; crosses card faces and gaps only on 0118 and 0126
QUEST_SWIPE_Y = 470
# inside the daily card face on 0118 (x 858 to 1290); the lane is card face or gap from
# x 250 to 1450 on 0126
QUEST_SWIPE_X_START = 1250
QUEST_SWIPE_X_END = 650  # a moving drag never opens a quest; only a stationary tap does
QUEST_SWIPE_MS = 600
# was 8 in the spec; about eight columns at 471 px and a 600 px swipe moves about 1.3
# columns, and the repeated-page stop ends the sweep early anyway
QUEST_SWEEP_MAX = 10
# Landmark only, so a swipe lane can be checked against it: the reroll button box is
# (1297, 137, 1570, 197) on 0126 and 0140, and it is absent at the left edge (0118, 0121).
QUEST_REROLL_BUTTON = (1433, 167)  # NEVER tapped: a reroll throws a quest away

# --- Social / Do-Not-Disturb ---------------------------------------------------
# Team/friend invites pop a modal that BLOCKS the menu and can interrupt the farm, so at
# startup we set the game's own mute settings (in-game DND) once per session. Calibrated
# live on a farm account (1600x900, 2026-06-09):
#   team-invite mutes:  menu -> translucent "+" team slot -> TEAM UP panel -> gear ->
#      SOCIAL SETTINGS. Set MUTE FRIENDS = 24h and MUTE RECENT TEAMMATES = 30 days.
# (Round 6, legacy owner-instructions note, not ported: the separate "block online push
# notifications" leg was removed — unnecessary; its coordinates/HSV constants are
# deleted below.)
# Every hop VERIFIES the expected screen via OCR and bails back to the menu if it isn't
# there — so a stale coordinate degrades to a logged no-op, never a misfire.

# How long each panel gets to animate in before the verify gives up. Probe-measured
# live (1600x900, 2026-09-17): the TEAM UP title reads 4.9 s after the "+" slot tap and
# MUTE FRIENDS reads 2.5 s after the gear tap, so the old fixed 4 x 1.0 s poll gave up
# first and the whole leg skipped. Budgets are the measured times roughly doubled, for
# cold-booted clients; settings.py polls them every 0.5 s. Waits only, no tap points.
DND_PANEL_WAIT_S = 8.0  # TEAM UP title after the "+" slot tap (measured 4.9 s)
DND_SETTINGS_WAIT_S = 6.0  # MUTE FRIENDS after the gear tap (measured 2.5 s)

# The translucent "+" team slot beside the menu brawler (left slot). Calibrated from a
# menu capture (zoomed crop of a saved menu screenshot): the slot card spans ~(357-445,
# 443-515), plus icon centred at (402,482). (685,380) is the brawler body — never tap it.
# Confirmed live by the 2026-09-17 probe: this tap opens the TEAM UP panel.
DND_TEAM_SLOT = (402, 482)

# TEAM UP panel (opens from the "+"): gear sits top-right just LEFT of the red ✕ — they're
# only ~65 px apart, so these two must stay distinct (a 1485,30 tap closed the panel).
DND_TEAMUP_GEAR = (1420, 30)  # confirmed 2026-09-17: opens SOCIAL SETTINGS
DND_TEAMUP_CLOSE_X = (1490, 30)  # red ✕ -> back to menu

# SOCIAL SETTINGS (team invites). The WHOLE ROW is tappable, so we tap the OCR'd label
# centers; the radio circles (left of each label) carry the selected-state colour.
DND_MUTE_FRIENDS_24H = (408, 427)  # row label; its radio ≈ (278, 427)
DND_MUTE_FRIENDS_24H_RADIO = (278, 427)
DND_MUTE_RECENT_30D = (998, 505)  # row label; its radio ≈ (868, 505)
DND_MUTE_RECENT_30D_RADIO = (868, 505)
DND_MUTES_CONFIRM = (1229, 720)  # green CONFIRM (applies + closes the panel)
# Selected radio = bright ORANGE fill; unselected = dark blue. Same orange as the brawler
# filter badges, so we reuse that HSV window with a box sampled around the radio center.
DND_RADIO_ORANGE_LO = BRAWLER_TOGGLE_ORANGE_LO
DND_RADIO_ORANGE_HI = BRAWLER_TOGGLE_ORANGE_HI
DND_RADIO_ON_FRAC = 0.12  # orange fraction above this => that option is selected
DND_RADIO_HALF = 22  # half-size of the sampled box (radio circle is small)
# The full mute-duration radio grid (probe-measured live, 2026-06-10): rows
# 60min/24h/30d at y 349/427/505; radios at x 278 (MUTE FRIENDS column) / 868
# (MUTE RECENT TEAMMATES column). There is NO "Don't mute" row in the current
# UI — unmuting = tap the SELECTED (orange) radio, which CLEARS it (verified
# live: the "Muted friends for…" status line is gone after CONFIRM + reopen).
DND_MUTE_RADIO_GRID = {
    "friends": ((278, 349), (278, 427), (278, 505)),
    "recent": ((868, 349), (868, 427), (868, 505)),
}

# Stop etiquette: a USER-initiated stop (the stop.flag path — a stop from the control
# panel, or the supervisor retiring a worker) hands the account back to a human, so the
# worker best-effort turns the in-game invite mutes OFF at the menu before closing the
# game. Scheduled cap stops (max_games / max_minutes) keep DND on — the account comes
# right back. BRAWL_DND_OFF_ON_STOP=0 disables the whole behavior without a deploy.
DND_OFF_ON_STOP = os.environ.get("BRAWL_DND_OFF_ON_STOP", "1") != "0"

# --- Team-invite popup (reactive decline) --------------------------------------
# Backstop to the DND mutes above: a TEAM INVITE modal (e.g. from a non-friend) still
# BLOCKS the menu and eats PLAY taps until it times out. Detect it by OCR'ing its title
# band ("TEAM INVITE", measured ~800,223), then tap MUTE and REJECT. Calibrated live on
# a farm account, 2026-06-09.
TEAM_INVITE_TITLE_REGION = (450, 150, 1150, 300)  # band around the "TEAM INVITE" title
INVITE_MUTE_BUTTON = (796, 654)  # MUTE label (also OCR-findable; this is the fallback)
# ⚠️ REJECT's stylized label OCRs as "REJEGT" — never OCR it. It sits at the fixed
# mirror of the green ACCEPT button (947,565), which must NEVER be tapped.
INVITE_REJECT_BUTTON = (653, 565)

# --- Supercell ID / onboarding --------------------------------------------------
# Navigation for the new-account onboarding flow (core/onboarding.py). Everything in
# this block was calibrated READ-ONLY on a farm account (1600x900, 2026-06-09): the menu
# hamburger opens a side menu (SETTINGS at top, SUPERCELL ID at the bottom); the
# SUPERCELL ID entry opens a right-side overlay (gear top-left, ✕ top-right); the
# gear opens the SCID SETTINGS panel that holds the logged-in email ("Logged in with
# <email>"), "Switch account" and "Log Out".
# ⚠️ SCID_SWITCH_ACCOUNT is ACCOUNT-CHANGING: only core/onboarding.py may tap it, and
# only on an explicitly-allowed non-farm instance. SCID_LOG_OUT is documented purely
# so nothing ever taps near it.
MENU_BURGER = (1522, 43)  # ≡ button, top-right of the main menu
SIDE_MENU_REGION = (1150, 80, 1600, 900)  # OCR band: the side-menu entries
SIDE_MENU_SETTINGS = (1399, 128)  # SETTINGS entry
SIDE_MENU_SUPERCELL_ID = (1352, 837)  # SUPERCELL ID entry (bottom)
SCID_PANEL_REGION = (990, 0, 1600, 900)  # the white SCID overlay
SCID_GEAR = (1052, 53)  # gear inside the SCID overlay -> SCID SETTINGS
SCID_CLOSE_X = (1546, 54)  # ✕ (same spot on the overlay and its settings panel)
SCID_EMAIL_REGION = (1000, 510, 1600, 610)  # "Logged in with" + the email line
SCID_SWITCH_ACCOUNT = (1224, 663)  # ⚠️ account-changing — onboarding only
SCID_LOG_OUT = (1177, 797)  # ☠️ NEVER tapped — listed to keep taps away from it
SETTINGS_SUPERCELL_ID_BTN = (327, 780)  # green ID button on the SETTINGS screen


# --- Season-rollover recalibration tripwire (ops-resilience.md §B) -------------
# The brawler grid detector above is calibrated per season skin and silently
# stops matching when the game reskins — the failure smell is "things are visibly
# present but the sweep claims 0", which otherwise reads identically to a healthy
# "nothing to claim" day. The surface computes a cheap DISAGREEMENT signal (an
# independent, coarser check says "something IS there" while the calibrated detector
# says 0); core/controller.py streaks it in data/<acct>/recalib.json and emits a
# "recalibrate" event at the threshold (one alert per surface per account per day;
# the streak keeps counting while cooled down). ADVISORY ONLY: it never pauses
# farming or skips a sweep, and an exception in any tripwire codepath can never
# affect the sweep result. BRAWL_RECALIB_TRIPWIRE=0 turns it all off (no extra
# computation, no events). (r8: the shop-freebie surface was dropped with the shop.)
RECALIB_TRIPWIRE = os.environ.get("BRAWL_RECALIB_TRIPWIRE", "1") != "0"
RECALIB_BRAWLERS_STREAK = 2  # SESSIONS in a row the grid OCR reads 0 owned names


# --- In-match intelligence (P1: gas-aware heading + ability buttons) -----------
# See the legacy planning note "in-match intelligence", not ported (Phases A/D), and the
# measured calibration in the legacy research note "in-match vision" (not ported). Both
# features are pure color CV on frames the loop already captures (a handful of small-ROI
# color_fraction calls, negligible against the ~370 ms capture cost).

# Phase A — gas-aware heading. Kill-switch: BRAWL_GAS_AWARE=0 reverts the wander
# to today's pure random drift without a deploy (same convention as FAST_INPUT etc.).
GAS_AWARE = os.environ.get("BRAWL_GAS_AWARE", "1") != "0"
# Validated gas HSV window (gas body H55 S108 V229, padded for cloud fringes) and
# the 4 screen-edge bands for the LOCKED 1600x900 display, clipped inside the HUD
# (top-left "Teams left", top-right portraits, bottom-left joystick, bottom-right
# attack/super cluster, right-mid chat bubble). Keep in sync with tools/probe_gas.py.
GAS_HSV_LO = (48, 80, 150)
GAS_HSV_HI = (62, 140, 255)
GAS_BANDS = {
    "top": (280, 8, 1280, 88),
    "bottom": (430, 800, 1060, 880),
    "left": (8, 130, 88, 560),
    "right": (1512, 150, 1592, 540),
}
# Hysteresis (validated): an edge becomes gassed at >= ON, un-gasses below OFF. The
# corpus-measured fraction distribution is bimodal with a near-empty [0.005, 0.04)
# valley ("gas arriving"), so this only delays the signal a frame while killing flicker.
GAS_BAND_ON = 0.04
GAS_BAND_OFF = 0.02
# Repulsion strength: each gassed edge pushes the heading away with weight
# fraction * gain (the fraction rises as gas penetrates the band, so it doubles as
# an urgency signal). At gain 2.5 a freshly-gassed edge (0.04) nudges ~10% of the
# drift vector; a fully-gassed band (~1.0) dominates it ~2.5:1.
GAS_REPULSION_GAIN = 2.5

# Phase D — tap ability buttons when they light their ready color (gadget=green,
# super=yellow, hypercharge=purple). Kill-switch: BRAWL_ABILITY_BUTTONS=0.
# Calibrated live on a near-maxed account playing a maxed TARA (gadget + hypercharge unlocked),
# 704 read-only frames over 6 matches, 2026-06-09/10 — see the legacy research note
# "in-match vision", section "Phase D" (not ported). Each entry:
#   region       (x1, y1, x2, y2) ROI inside the button face (small = cheap + specific)
#   hsv_lo/hi    the button's READY color window
#   min_fraction ready-color fraction in the ROI at/above which the button reads lit
#   tap          where to tap it (button center)
# GADGET (measured): ready face is flat green H 59-60 / S 225-255; unlit face is
# dark gray (ZERO pixels above S 180). Corpus split: 404 frames < 0.02, 296 >= 0.54,
# only 4 in [0.02, 0.40) — so 0.40 sits in an empty valley. S >= 180 also hard-excludes
# the poison gas (S ~101-111), which can drift over the button cluster.
# SUPER / HYPERCHARGE: NOT calibrated yet — the ready states (yellow / bright purple)
# never appeared in the corpus (the blind super-tap in phase_playing fires the super
# within seconds of charging, and hypercharge never charged at the bot's damage
# output). Entries get added here once a ready frame is captured; the unlit faces
# measured so far: super skull = blue H 105 / S ~170, hypercharge swirl = dim purple
# H ~127 / S ~150 / V ~165 (these are what a window must NOT match).
ABILITY_BUTTONS_ENABLED = os.environ.get("BRAWL_ABILITY_BUTTONS", "1") != "0"
ABILITY_TAP_COOLDOWN = 5.0  # min seconds between taps of the SAME button (no spam)
ABILITY_BUTTONS: dict[str, dict] = {
    "gadget": {
        "region": (1340, 794, 1396, 850),
        "hsv_lo": (55, 180, 80),
        "hsv_hi": (66, 255, 255),
        "min_fraction": 0.40,
        "tap": (1368, 822),
    },
}

# --- Phase B — bush-hide + gas-aware repositioning (r8) ------------------------
# Steer the idle wander toward the nearest BUSH and hide; when the poison gas closes
# in, relocate to a bush nearer the MAP CENTER. Kill-switch: BRAWL_BUSH_HIDE=0 reverts
# to the existing pure random/gas-repelled wander without a deploy. DEFAULT ON, but
# EVERY failure path (no bush detected, detector exception, gate not met) degrades to
# that wander — see core/match_vision.bush_clusters / bush_decision and the playing
# phase in core/controller.py.
#
# ⚠️ CALIBRATED BY REASONING, NOT LIVE-VALIDATED. The P9 corpus had only RED bushes
# (Kroket); the green window was measured from ONE real green-biome frame
# (captures/live_inmatch.png — bush masses H 60-95 / S>=140 / V~60-200; gas-window
# overlap negligible). Map themes vary, so the gate is CONSERVATIVE: a saturated-green
# window + a high min-cluster-area, so "no bush" (-> wander) is the safe common
# outcome. The orchestrator/owner must watch the first sessions.
BUSH_HIDE = os.environ.get("BRAWL_BUSH_HIDE", "1") != "0"
# Saturated-green foliage window (OpenCV HSV: H 0-179, S/V 0-255). Measured-green is
# H~73 S~230 V~140; widened to H 40-95 / S>=120 / V 50-210 to span biome variation
# while staying clear of the gas body (H55 S108) by requiring S>=120 (gas S<=140 but
# the bush mass is far more saturated) and excluding bright/pale greens (V<=210).
BUSH_HSV_LO = (40, 120, 50)
BUSH_HSV_HI = (95, 255, 210)
BUSH_DOWNSCALE = 8  # mask/cluster on a 1/8 frame (200x112) — cheap; centroids re-scaled
# Min connected-component area IN THE DOWNSCALED MASK to count as a bush cluster.
# At 1/8 scale the big left bush mass in the calibration frame measured ~1117 px and
# small green patches ~100-400; 60 keeps real masses while dropping single-tile specks
# and the player's green aim-glow. Conservative — raise it if false bushes appear live.
BUSH_MIN_AREA = 60
# Self anchors at ~(690, 440) on the 1600x900 frame (research §3 — it drifts with the
# camera, so treat these distances as slack, not exact).
BUSH_SELF_POS = (690, 440)
BUSH_AT_TARGET_PX = 90  # within this of the target bush center => already hidden -> hold
# Gas "closing in" gate for relocation: relocate when gassed edges are present on
# >= this many sides, OR any gassed edge's band fraction is this deep (gas deep in a
# band ≈ gas near us). 0.5 means a band half-saturated by gas.
BUSH_GAS_RELOCATE_EDGES = 2
BUSH_GAS_PROXIMITY = 0.5
# Cap on relocations per gas event (controller-enforced) so a flickering gas read
# can't turn into relocate-spam; reset when the gassed-edge set clears.
BUSH_MAX_RELOCATIONS = 3
# Micro-jitter while hidden in a bush: a small periodic joystick nudge so the worker
# never looks TRUE-AFK (idle-disconnect / AFK-detection) while concealed.
BUSH_JITTER_RADIUS = 45  # small joystick swipe magnitude for the in-bush micro-jitter
BUSH_JITTER_MIN_INTERVAL = 4.0  # seconds between micro-jitters (well under idle-kick)
BUSH_JITTER_MAX_INTERVAL = 8.0

# Play mode's detector runs beside an untouched farm loop and logs its boxes; it sends
# no input. OFF by default: with the flag absent the worker behaves exactly as before.
PLAY_SHADOW = os.environ.get("BRAWL_PLAY_SHADOW", "0") == "1"

# --- Calibration overrides (phase 8): <home>/calibration/calibration.toml ----------
# Floats and two-int tap tuples above may be overridden from that file. The
# module below only reads; nothing in the app writes calibration.toml.
from brawlfarm.core import calibration as _calibration  # noqa: E402

CALIBRATION_FILE = HOME_DIR / "calibration" / "calibration.toml"
CALIBRATION = _calibration.apply(globals(), CALIBRATION_FILE)
