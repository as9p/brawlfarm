"""
The bot's brain: a contextual state machine that farms Trio Showdown matches.

Phases (distinct from the per-frame State in states.py):
  returning  -> get back to a known MAIN MENU, clearing any reward/Star-Drop popups
  at_menu    -> log trophies + battlelog, verify Trio Showdown, press PLAY
  queuing    -> wait through matchmaking (no taps near the Exit button)
  playing    -> tap the attack point (and Super, harmless) on cadence until results
(There is no "results" phase — RESULTS frames are a per-frame State handled inside
the returning/queuing/playing phases via advance_results.)

Why contextual instead of pure per-frame detection: in-match scenes can't be
template-matched reliably, but the EXIT conditions (menu, results, matchmaking)
can. So we anchor on those and treat UNKNOWN according to the phase we're in.

Safety rails (for unattended overnight running):
  - assert 1600x900 at startup
  - freeze detection -> relaunch the game
  - bounded recoveries -> hard stop with a clear log if we can't reach a known state
  - --max-games / --max-minutes caps; Ctrl-C exits cleanly
  - never taps the matchmaking Exit; never blindly mass-taps an unknown dialog forever
"""

from __future__ import annotations

import math
import os
import random
import time

import cv2

from brawlfarm.core import (
    adb,
    brawlers,
    captures,
    config,
    farmplan,
    game_update,
    match_vision,
    preview,
    questpick,
    quests,
    recalib,
    rewards,
    settings,
    states,
    status,
    vision,
)
from brawlfarm.core.api import ApiClient, ApiError
from brawlfarm.core.datalog import DataLog
from brawlfarm.core.recorder import Recorder
from brawlfarm.core.states import State

# --- tunable timeouts (seconds / counts) ---
QUEUE_REPRESS = 8.0  # if still on MENU this long after PLAY, press again
QUEUE_TIMEOUT = 45.0  # no matchmaking seen this long -> recover
MATCH_TIMEOUT = 360.0  # playing this long without results -> recover
RETURN_PATIENCE = 4  # short: let menu-load transitions resolve to MENU before tapping
DROP_TAPS = 30  # tap-through budget for drops/reward reveals (chaos can split 2/4/8)
BACK_TRIES = 4  # BACK attempts if tapping never reaches a known screen
MAX_RECOVERY = 4  # consecutive recoveries (without reaching menu) before stop
MEGA_QUEST_MAX_STREAK = 2  # backstop: don't re-enter QUESTS this many times w/o a game played
# (guards against a mis-read gold indicator that never clears -> quests<->menu thrash)
# Network-stuck detection (owner-reported 2026-06-10): a short disconnect can wedge
# the game on the END of a Showdown match FOREVER with dead buttons. Frames keep
# ANIMATING, so freeze detection never fires, and the results branch used to reset
# the stuck counter on every tap — an unbounded loop with healthy heartbeats. The
# only fix is closing the game (recover()'s force-stop): a plain HOME is NOT enough,
# the app resumes straight back into the stuck screen (verified live on a farm account).
# The mid-match variant of the same bug ("Teams left: 1" forever) is already
# bounded by MATCH_TIMEOUT -> recover(): after the force-stop relaunch the server-
# side match has long expired, so the game loads to the menu. No banner OCR — the
# "Teams left" banner shows during NORMAL play too, so presence is not a signal.
RESULTS_STUCK_TAPS = 10  # consecutive results-advance taps without leaving RESULTS
# (real results clear in 2-3 taps; 10 means the UI is dead)


class Controller:
    def __init__(
        self,
        max_games=None,
        max_minutes=None,
        debug_shots=False,
        select_brawler=False,
        dnd=False,
    ):
        self.api = ApiClient()
        self.dl = DataLog()
        self.max_games = max_games
        self.max_minutes = max_minutes
        self.debug_shots = debug_shots
        self.select_brawler = select_brawler  # select lowest-trophy brawler once at start
        self.dnd = dnd  # set the in-game invite mutes (DND) once at start

        self.running = True
        self.phase = "returning"
        self.games_played = 0
        self._select_brawler_done = False
        self._dnd_done = False
        # Mega-quest activation is ALWAYS ON (no flag): a recurring menu check, not a
        # once-per-session step. This counter is only a thrash backstop (reset on PLAY).
        self._mega_quest_streak = 0
        # The quest-aware pick (plan flag, ladder only): whether this session's plan asked
        # for it and the goal it caps candidates at (both decided at the first menu), the
        # gate for the one quests visit that reads the cards, and the cards themselves
        # (None until read, and None again if the visit failed).
        self._quest_aware = False
        self._quest_goal: int | None = None
        self._quest_visit_done = False
        self._quest_cards: list[str] | None = None

        # Session-recap bookkeeping (printed on stop). Trophies come from API snapshots;
        # skins from skin-reward events. (Currency gain-tracking was removed in round 7
        # — trophy-only minimalism, legacy owner note, not ported.)
        self.rewards = rewards.SessionRewards()
        self._start_trophies = None
        self._last_trophies = None
        self._recap_done = False
        self._adb_errors = 0  # consecutive adb failures -> recover, don't crash

        now = time.monotonic()
        self.start = now
        self.phase_started = now
        self.last_attack = 0.0
        self.last_change = now
        self.last_sig = None

        self.matchmaking_seen = False
        self.unknown_clears = 0
        self.recovery_attempts = 0
        self.disconnect_count = 0
        # Graceful-stop plumbing (scheduler/control panel): a `stop.flag` file in DATA_DIR
        # asks this worker to stop CLEANLY at the next menu (between games), like the
        # --max-games/--max-minutes caps. Polled on the %25 cadence (cheap); honored
        # in phase_at_menu before PLAY; the in-loop hard stop only fires if the soft
        # stop is >10 min overdue (e.g. wedged mid-recovery).
        self._stop_flag_seen = False
        self._soft_stop_since = None  # monotonic ts when a soft-stop reason appeared
        # Labeled frame recorder (calibration corpora): off unless a `record.flag`
        # file sits next to stop.flag in DATA_DIR. Observation only — it never taps,
        # never changes a State, and swallows its own errors.
        self.recorder = Recorder(
            config.HOME_DIR / "calibration", config.DATA_DIR.name, config.DATA_DIR / "record.flag"
        )
        # Shadow play session (the model watching the farm play): built on the first tick
        # that needs it, so a worker with `shadow` off never imports the play package.
        # Observation only: it never taps. _play_off latches on the first failure.
        self.play = None
        self._play_off = False
        # Network-stuck tracker (see RESULTS_STUCK_TAPS):
        self._results_taps = 0  # consecutive advance_results taps without progress
        self.popup_count = 0
        self.last_move = 0.0
        # In-match anti-detection state: a continuously-wandering heading + per-action
        # randomized gaps, re-rolled each time we act so the attack/move cadence and the
        # movement direction never settle into a detectable repeating pattern.
        self._heading = random.uniform(0, 2 * math.pi)
        self._attack_gap = config.ATTACK_INTERVAL
        self._move_gap = config.MOVE_INTERVAL_MIN
        # Phase A (gas-aware heading): per-edge hysteresis tracker + the last set of
        # gassed edges (kept only to log on change). Reset at every match entry.
        self._gas = match_vision.GasTracker()
        self._gas_edges: set[str] = set()
        # Phase B (bush-hide + gas relocation, r8): bookkeeping reset each match.
        # _bush_jitter_at = monotonic ts of the last in-bush micro-jitter; _relocations
        # = relocations spent on the CURRENT gas event (capped, reset when gas clears);
        # _bush_logged = the last decision kind we logged (log only on change, sparse).
        self._bush_jitter_at = 0.0
        self._relocations = 0
        self._bush_logged: str | None = None
        # Phase D (ability buttons): per-button monotonic ts of the last tap, so a
        # button that stays lit (e.g. a mis-read) is tapped at most once per cooldown.
        self._ability_last: dict[str, float] = {}
        # Fast re-entry + status-heartbeat + mid-session brawler-switch bookkeeping.
        self._unknown_streak = 0  # consecutive UNKNOWN frames (detects the home screen fast)
        # UNKNOWN-path handler repeat guard (_unknown_modal): which handler branch the last
        # loop iteration took, and how many iterations in a row took it.
        self._repeat_handler: str | None = None
        self._repeat_count = 0
        self._loop_i = 0  # loop counter; throttles the status heartbeat write
        self._farm_brawler = None  # API name of the brawler we grind (for the goal switch)
        # Trophy goal for the farmed brawler: the farm plan (core/farmplan.py) can set a
        # per-target goal; default = the existing 1000 reselect threshold.
        self._farm_goal = config.RESELECT_TROPHY_THRESHOLD
        self._reselect_pending = False  # set when the farm brawler crosses _farm_goal
        # Prestige "account maxed" log/event de-dup latch (owner rule 2026-06-14:
        # "don't grind brawlers over 1000"). The actual hold behavior comes from NOT
        # setting _reselect_pending (and the early return in _do_select_brawler) — this
        # flag only suppresses re-logging the "account maxed" line/event each snapshot.
        # Deliberately never read for control flow, so a stuck latch can't wedge the
        # worker. Set when nothing is under the goal; reset when a reselect fires.
        self._account_maxed = False
        # ladder rotation state (r6: folded in from the old "optimal" mode):
        # brawlers rotated out for THIS session (subnormal trailing trophy rate and
        # not on a winstreak — see config.OPTIMAL_RATE_*). In memory on purpose: a
        # worker restart forgives a cold streak.
        self._rotated: set[str] = set()
        self.last_trophy_api = 0.0
        self.last_battlelog_api = 0.0
        self._prev_phase = None
        self._shot_n = 0
        # Self-heal scenarios (r8): bound the ceremony tap-through so a screen we
        # CAN'T clear escalates to recovery instead of looping forever, and throttle
        # the in-match modal scan so it costs ~nothing on the playing hot path.
        self._ceremony_count = 0  # ceremony screens cleared in the current run cycle
        self._playing_modal_i = 0  # in-match iteration counter for the modal scan
        self._dismiss_tried = False  # recover() soft dismiss-ladder used this cycle
        self._update_checked = False  # game auto-update store check used this run (never reset)

    # --- small helpers -------------------------------------------------------

    def log(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def tap(self, point, kind: str | None = None) -> None:
        adb.tap(*point)
        if kind:
            self.dl.event("tap", button=kind, x=point[0], y=point[1], phase=self.phase)

    def _jitter(self, point) -> tuple[int, int]:
        """Nudge a tap point by +/- TAP_JITTER_PX. Hitting the exact same pixel on every
        attack is an obvious automation signature; the in-match buttons (attack auto-aim,
        Super) are large, so a few px of jitter never misses."""
        j = config.TAP_JITTER_PX
        return (point[0] + random.randint(-j, j), point[1] + random.randint(-j, j))

    def set_phase(self, name: str) -> None:
        if name != self.phase:
            self.dl.event("phase", to=name, frm=self.phase, games=self.games_played)
            self.log(f"phase: {self.phase} -> {name}")
        self.phase = name
        self.phase_started = time.monotonic()
        self.unknown_clears = 0
        self._results_taps = 0  # results-stuck counter is per-visit (network-stuck)
        self._write_status()

    def _maybe_shot(self, screen, tag: str) -> None:
        if not self.debug_shots:
            return
        d = config.CAPTURES_DIR / "run"
        d.mkdir(parents=True, exist_ok=True)
        self._shot_n += 1
        cv2.imwrite(str(d / f"{self._shot_n:03d}_{tag}.png"), screen)

    def event_shot(self, screen, tag: str) -> None:
        """Always-on snapshot of a notable event (recover/disconnect/stuck) so we
        can diagnose anything novel in the morning, even without --debug-shots.
        Capped by captures.py: one frame per tag per 10 minutes, newest 100 kept."""
        captures.write_event(screen, tag)

    # --- API snapshots (throttled) ------------------------------------------

    def maybe_snapshot_trophies(self) -> None:
        now = time.monotonic()
        if now - self.last_trophy_api < config.API_TROPHY_MIN_INTERVAL:
            return
        self.last_trophy_api = now
        try:
            player = self.api.get_player()
            self.dl.log_menu_trophies(player)
            total = player.get("trophies")
            self.dl.event("trophies", total=total)
            self.log(f"trophies: {total}")
            if total is not None:  # track session start/last for the recap
                if self._start_trophies is None:
                    self._start_trophies = total
                self._last_trophies = total
            self._check_farm_brawler_trophies(player)
        except ApiError as e:
            self.dl.event("api_error", where="player", err=str(e))

    def maybe_log_battlelog(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_battlelog_api < config.API_BATTLELOG_MIN_INTERVAL:
            return
        self.last_battlelog_api = now
        try:
            items = self.api.get_battlelog()
            n = self.dl.log_games_from_battlelog(items)
            if n:
                self.log(f"logged {n} new game(s) from battlelog")
        except ApiError as e:
            self.dl.event("api_error", where="battlelog", err=str(e))

    # --- status heartbeat + mid-session brawler switch ----------------------

    def _write_status(self) -> None:
        """Write this instance's heartbeat (best-effort) so the control panel / watchdog
        can see what we're doing without touching the process. See core/status.py."""
        status.write_status(
            config.DATA_DIR,
            {
                "pid": os.getpid(),
                "running": self.running,
                "phase": self.phase,
                "games_played": self.games_played,
                "max_games": self.max_games,
                "minutes_elapsed": round((time.monotonic() - self.start) / 60, 1),
                "max_minutes": self.max_minutes,
                "last_trophies": self._last_trophies,
                "start_trophies": self._start_trophies,
                "port": config.ADB_PORT,
                "tag": config.PLAYER_TAG,
                "session": self.dl.session_path.name,
                "farm_brawler": self._farm_brawler,
                "recovery_attempts": self.recovery_attempts,
                "disconnect_count": self.disconnect_count,
            },
        )

    def _capture_farm_brawler(self) -> None:
        """Record the API name of the brawler we just selected to grind (the current
        lowest-trophy one) so we can watch its trophies and switch off it once it crosses
        RESELECT_TROPHY_THRESHOLD. Uses the API-canonical name for reliable later matching."""
        try:
            brawler_list = self.api.get_player().get("brawlers") or []
            low = min(brawler_list, key=lambda b: b.get("trophies", 0), default=None)
            if low:
                self._farm_brawler = low.get("name")
                self.log(f"farm brawler = {self._farm_brawler} ({low.get('trophies')} trophies)")
        except ApiError as e:
            self.dl.event("api_error", where="capture_farm_brawler", err=str(e))

    def _check_farm_brawler_trophies(self, player: dict) -> None:
        """If the brawler we're grinding has crossed its goal (the farm plan's
        CURRENT goal — re-evaluated against the live roster every snapshot), flag a
        mid-session re-select (handled at the next menu). Matches by the API name
        captured at selection; if it can't be matched we just don't switch (safe no-op).

        WHY re-evaluate instead of trusting the goal captured at selection: ladder's
        step goal is a function of the roster MINIMUM, which moves under us — a drop
        can unlock a brand-new brawler at 0 trophies mid-session, pulling the step
        down to 0-100 while we're grinding a 300-trophy brawler toward a goal frozen
        at 400 (the owner-observed "CROW 336/100" bug; a farmplan hiccup at selection
        even left the goal at the 1000 default). Owner's law: "if the current brawler
        is above or climbs above the current ladder step it should switch" — so the
        comparison must use the CURRENT step goal, not a snapshot of it."""
        if not self._farm_brawler or self._reselect_pending:
            return
        brawler_list = player.get("brawlers") or []
        plan = None
        try:  # cheap: a small json read + a pure min() over the roster we already have
            plan = farmplan.load_plan()
            _, goal = farmplan.choose_target(plan, brawler_list, exclude=self._rotated)
            self._farm_goal = goal
        except Exception as e:  # keep the last-known goal on a plan hiccup
            self.dl.event("farmplan_error", where="goal_refresh", err=repr(e))
        # Ladder rotation (r6: folded in from the old "optimal" mode; r8: the
        # statistics upgraded to the win-rate model v2 — legacy research note
        # "winrate model", not ported).
        # farmplan.rotation_decision rotates the farmed brawler out for the session
        # when (primary) a clearly-better in-band alternative exists AND we're below
        # the account mean on this one [opportunity_cost], or (secondary) its
        # empirical-Bayes-shrunk recency rate is net-losing [absolute_floor]. Both
        # triggers use the SAME shrunk EWMA the selection bias uses, so a 3-game cold
        # patch no longer overreacts. The owner's guard is UNCHANGED and applied HERE,
        # verbatim: NEVER rotate a brawler on a winstreak (farmplan.recent_winstreak) —
        # a hot brawler stays even if its rate dipped. The normal reselect machinery
        # then re-asks farmplan with it excluded. Prestige never rotates.
        if plan is not None and plan.get("mode") == "ladder":
            try:
                rotate, reason = farmplan.rotation_decision(self._farm_brawler, brawler_list)
                if rotate:
                    if farmplan.recent_winstreak(self._farm_brawler):
                        # Rotation triggered, but on a winstreak — keep it (the
                        # owner's rule). Don't return: the brawler may still cross
                        # its goal below, which should reselect normally.
                        self.log(
                            f"ladder: {self._farm_brawler} rotation ({reason}) "
                            "but on a winstreak — keeping"
                        )
                    else:
                        self._rotated.add(self._farm_brawler)
                        self._reselect_pending = True
                        self.dl.event(
                            "rotate_brawler",
                            brawler=self._farm_brawler,
                            reason=reason,
                            halflife=config.WINRATE_HALFLIFE,
                            prior_k=config.WINRATE_PRIOR_K,
                        )
                        self.log(
                            f"ladder: {self._farm_brawler} -> rotating out ({reason}; "
                            f"shrunk EWMA hl={config.WINRATE_HALFLIFE:g}, "
                            f"k={config.WINRATE_PRIOR_K:g})"
                        )
                        return
            except Exception as e:  # rate trouble must never block the snapshot
                self.dl.event("farmplan_error", where="rate_check", err=repr(e))
        for b in brawler_list:
            if b.get("name") == self._farm_brawler:
                if (b.get("trophies") or 0) >= self._farm_goal:
                    # Owner rule 2026-06-14: don't grind brawlers over the goal. In a
                    # prestige plan, if NOTHING is left under the goal the account is
                    # maxed — hold the current brawler (latch + log once) rather than
                    # reselect-loop into a >goal brawler. Ladder keeps reselecting (it
                    # raises the whole roster a tier at a time, goal moves with it).
                    is_prestige = bool(plan and plan.get("mode") == "prestige")
                    any_under = any(
                        (x.get("trophies") or 0) < self._farm_goal for x in brawler_list
                    )
                    if is_prestige and not any_under:
                        # Opt-in maxed_fallback (prestige-only): when prestige is
                        # EXHAUSTED, switch ONCE to the configured brawler and keep
                        # farming it (deliberate >goal grind for late-stage stats —
                        # owner-sanctioned, gated strictly to all-maxed). With it
                        # unset/unowned, OR once we're already ON the fallback, fall
                        # through to the legacy hold (latch + log-once) so no thrash.
                        # Gate on OWNERSHIP, not just "set": choose_target only returns
                        # the fallback when it's owned, so without this an unowned/typo'd
                        # name would fire the switch, get None back from the reselect,
                        # hold unchanged, and re-fire every snapshot forever (~60s
                        # reselect-thrash). Mirror choose_target's ownership guard.
                        fb = (plan or {}).get("maxed_fallback")
                        fb = fb.strip() if isinstance(fb, str) else ""
                        fb_owned = bool(fb) and any(
                            (x.get("name") or "").upper() == fb.upper() for x in brawler_list
                        )
                        if fb_owned and (self._farm_brawler or "").upper() != fb.upper():
                            self._account_maxed = False
                            self._reselect_pending = True
                            self.log(
                                f"all brawlers >= {self._farm_goal} — prestige "
                                f"exhausted; switching to maxed_fallback {fb}"
                            )
                            self.dl.event("maxed_fallback_switch", fallback=fb)
                            return
                        if not self._account_maxed:
                            self._account_maxed = True
                            self.log(
                                f"all brawlers >= {self._farm_goal} — account maxed; "
                                "holding current brawler (won't grind over the goal)"
                            )
                            self.dl.event("account_maxed", goal=self._farm_goal)
                    else:
                        self._account_maxed = False
                        self._reselect_pending = True
                        self.log(
                            f"farm brawler {self._farm_brawler} at {b.get('trophies')} "
                            f">= {self._farm_goal} -> will re-select per farm plan"
                        )
                return

    # --- phase handlers ------------------------------------------------------

    def phase_returning(self, screen, state) -> None:
        if state == State.MENU:
            self.recovery_attempts = 0
            self.disconnect_count = 0
            self.popup_count = 0
            self._ceremony_count = 0  # back at the menu -> fresh ceremony budget
            self._dismiss_tried = False  # fresh dismiss-ladder budget for next cycle
            self.set_phase("at_menu")
            return
        if state == State.MATCHMAKING:
            self.matchmaking_seen = True
            self.set_phase("queuing")
            return
        if state == State.IN_MATCH:
            # a PLAY AGAIN-style requeue dropped us straight into a match
            self._enter_match()
            return
        if state == State.RESULTS:
            # Network-stuck rail (owner 2026-06-10): a short disconnect can leave the
            # END-of-Showdown screen up FOREVER with dead buttons — every tap below
            # used to reset the stuck counter, so the loop tapped a corpse all day
            # with healthy heartbeats. Count consecutive results taps instead; real
            # results clear in 2-3 taps, so blowing this budget means the UI is dead
            # and only a game relaunch (recover's force-stop) un-wedges it.
            self._results_taps += 1
            if self._results_taps > RESULTS_STUCK_TAPS:
                self.recover("results_stuck")
                return
            self.advance_results(screen)
            self.unknown_clears = 0  # progressing through results, not stuck
            return
        if state == State.TROPHY_SCREEN:
            # The brawler TROPHIES detail / Trophy Road screen — auto-shown at rank-up
            # milestones (or if a stray tap opened it). The bot should never linger here.
            # Two variants: the auto rank-up "Reward Claimed!" celebration has only a
            # LET'S GO button (tap it), while the trophy detail has the top-right HOME
            # button (tap it — NOT a center-tap, which re-opens the brawler, and NOT BACK,
            # per the taps-only rule). Count attempts (don't reset to 0) so a screen we
            # genuinely can't clear escalates to recovery instead of looping forever.
            # "LET'S GO" (rank-up celebration): locate it by the green-CTA COLOR gate
            # (states.green_cta), not OCR — the stylized label mangles (house rule),
            # and the old case-insensitive "LET" substring needle also matched
            # unrelated text like "COMPLETED" ("compLETed").
            btn = states.green_cta(screen)
            if btn is not None:
                self.tap(btn, kind="lets_go")
            else:
                self.tap(config.HOME_BUTTON, kind="home")  # top-right HOME -> main menu
            self.unknown_clears += 1
            time.sleep(1.0)
            if self.unknown_clears > RETURN_PATIENCE + DROP_TAPS + BACK_TRIES:
                self.recover("stuck_trophy")
            return
        # A drop can reveal a SKIN -> a "NEW <rarity> SKIN!" popup with CONTINUE / EQUIP
        # NOW. Handle it explicitly (press CONTINUE, log the skin) rather than blind-tap,
        # which could hit EQUIP NOW. Cheap template check; OCR only runs when it matches.
        if self._handle_skin_popup(screen):
            self.unknown_clears = 0
            time.sleep(0.6)
            return
        # A brawler-unlock CEREMONY (single green CTA: LET'S GO / GOT IT) or the
        # CHOOSE-A-BRAWLER chooser (r8 self-heal, live incident). Handle it BEFORE
        # the blind drop tap-through — a blind safe-point tap can miss the CTA and stall
        # here for hours. Bounded by CEREMONY_MAX_SCREENS; once exhausted we fall through
        # to the drop/BACK/recover ladder below.
        if self._ceremony_count < config.CEREMONY_MAX_SCREENS and self._handle_ceremony(screen):
            self.unknown_clears = 0
            return
        # UNKNOWN. Could be: a menu-load transition (resolves to MENU on its own), or a
        # reward/Star/Chaos-drop flow that must be TAPPED through to claim. We can't
        # template every drop, so: wait briefly for transitions, then tap-to-advance and
        # RE-CHECK after each tap (the MENU branch above stops us the instant we're back).
        # Adaptive by design — chaos drops can split 2/4/8, needing a variable # of taps.
        self.unknown_clears += 1
        n = self.unknown_clears
        if n <= RETURN_PATIENCE:
            time.sleep(0.5)
            return
        if n == RETURN_PATIENCE + 1:
            self.event_shot(screen, "unknown")  # capture the persistent screen once
        if n <= RETURN_PATIENCE + DROP_TAPS:
            # Clear a drop / reward reveal. MOST drops (Starr / Chaos) and reward reveals
            # advance on TAPS, so we mainly burst-tap a safe point (NOT screen-centre, which
            # is the menu brawler). But hold-drops (Angel / Demon / Nova) only open on a
            # press-and-HOLD — tap-only loops here forever on those. They announce themselves
            # with "TAP AND HOLD!" near the bottom, so we OCR that band and hold ONLY when we
            # see it (signal-driven — much more reliable than holding on a blind cadence, and
            # it won't stray-hold the menu brawler). Monster Eggs need a SWIPE (occasional
            # step). The MENU branch up top stops us the instant we're back. Bursting beats
            # one tap/loop because the screencap+poll cost dominates; we re-check each pass.
            step = n - RETURN_PATIENCE
            if step % config.DROP_OCR_EVERY == 1 and self._drop_wants_hold(screen):
                adb.tap_hold(*config.SAFE_HOLD_POINT, config.DROP_HOLD_MS)  # Angel/Demon/Nova
                time.sleep(0.6)
            elif step % 7 == 0:
                adb.swipe(560, 450, 1040, 450, 250)  # slash a Monster Egg
                time.sleep(0.4)
            else:
                for _ in range(config.DROP_TAP_BURST):
                    adb.tap(*config.SAFE_DISMISS_POINT)
                    time.sleep(config.DROP_TAP_INTERVAL)
            return
        if n <= RETURN_PATIENCE + DROP_TAPS + BACK_TRIES:
            adb.keyevent(4)  # taps never reached menu -> BACK out of a stuck nav screen
            time.sleep(1.0)
            return
        self.recover("stuck_returning")

    def phase_at_menu(self, screen, state) -> None:
        if state != State.MENU:
            self.set_phase("returning")
            return
        # Graceful stop point: caps and the stop.flag are honored HERE, at the menu
        # between games — the session ends cleanly (recap logged), never mid-match.
        reason = self._soft_stop_reason()
        if reason is not None:
            self.stop(reason)
            return
        # Once per session, FIRST among the startup tasks: turn on the in-game invite
        # mutes (DND) so a team-invite modal can't interrupt the other startup tasks or
        # the farm. Runs here (the first confirmed menu) like the rest.
        if self.dnd and not self._dnd_done:
            self._dnd_done = True
            self._do_set_dnd()
            return  # re-evaluate the menu next loop
        # Once per session, ahead of the select below: with the plan's quest pick on, the
        # quests visit moves in front of the brawler select so the cards it reads can
        # steer it. It is the same single visit the mega-quest trigger further down makes
        # (it still activates a NEW MEGA QUEST if one is offered), not a second
        # navigation. With the flag off nothing is visited and the order stays as it was.
        if self.select_brawler and not self._quest_visit_done:
            self._quest_visit_done = True
            self._quest_goal = self._quest_pick_goal()
            self._quest_aware = self._quest_goal is not None
            if self._quest_aware:
                self._do_quest_visit()
                return  # re-evaluate the menu next loop
        # Once per session, before farming: select the lowest-trophy brawler so the
        # farm grinds it.
        if self.select_brawler and not self._select_brawler_done:
            self._select_brawler_done = True
            self._do_select_brawler()
            return  # re-evaluate the menu next loop
        # Mid-session: if the brawler we're grinding crossed the farm plan's CURRENT
        # goal (ladder = the live step floor, prestige = the plan goal, capped
        # by the 1000 rank milestone), re-select per the plan so we keep gaining fresh
        # trophies. Flag set API-side in maybe_snapshot_trophies / _check_farm_brawler_trophies.
        if self._reselect_pending:
            self._reselect_pending = False
            self.log("farm brawler crossed switch threshold -> re-selecting lowest")
            self.dl.event("reselect_brawler")
            self._do_select_brawler()
            return  # re-evaluate the menu next loop
        self.maybe_snapshot_trophies()
        self.maybe_log_battlelog()

        # Always-on, every menu visit (startup + between games): if a NEW MEGA QUEST is
        # available the QUESTS button is gold — go activate it. This is the ONLY menu action
        # taken between games. Activating clears the gold so it won't re-fire; the streak cap
        # backstops a mis-read gold that never clears (so we never thrash quests<->menu).
        if self._mega_quest_streak < MEGA_QUEST_MAX_STREAK and quests.quests_button_has_new(screen):
            self._mega_quest_streak += 1
            self._do_mega_quest()
            return  # re-evaluate the menu next loop

        # Mode verify (round 6, legacy owner-instructions note, not ported): the two live
        # "wrong_mode" firings today both RECOVERED — almost certainly the mode banner
        # caught mid-animation, NOT a real mode switch (the owner has never seen the farm
        # change modes). So DOUBLE-CONFIRM: a first miss waits ~1.7s and re-captures; a
        # transient miss (re-check hits) is just a debug line, nothing fires. Only a
        # confirmed miss (both fail) logs the event + a diagnostic shot WITH the match
        # score, attempts recovery, and — if nav fails — stops with the HONEST reason
        # "mode_verify_failed" (not "wrong_mode", which falsely implies a real switch).
        sel, miss_score = states.trio_showdown_score(screen)
        if not sel:
            # Keep the FIRST (failing) frame's score: the re-check below overwrites
            # miss_score, and on a transient miss the re-check HITS (score can be 1.00),
            # so logging the post-recheck value mislabeled a 1.00 hit as a "miss" score
            # (the owner's "transient miss (score=1.00)" oddity). Log the real miss.
            first_miss_score = miss_score
            time.sleep(config.MODE_VERIFY_RECHECK_S)
            screen = adb.screencap()
            sel, miss_score = states.trio_showdown_score(screen)
            if sel:
                self.log(
                    f"mode verify: transient miss (score={first_miss_score:.2f}) — re-checking"
                )
                return  # re-evaluate the menu next loop (Trio after all)
        if not sel:
            self.log(
                f"Trio Showdown not selected (confirmed, score={miss_score:.2f}) — "
                "attempting to navigate to it."
            )
            # Save a diagnostic shot on EVERY confirmed miss (even when recovery then
            # succeeds) so a recurring false alarm leaves ground truth to inspect.
            self.event_shot(screen, "wrong_mode")
            recovered = self.navigate_to_trio_showdown()
            self.dl.event("wrong_mode", score=round(miss_score, 3), recovered=recovered)
            if not recovered:
                self.log("WARNING: could not select Trio Showdown — stopping (mode_verify_failed).")
                self.stop("mode_verify_failed")
            return  # re-evaluate the menu next loop (now Trio, or stopped)

        self._mega_quest_streak = 0  # committing to a game clears the mega-quest backstop
        self.tap(config.PLAY_BUTTON, kind="play")
        self.matchmaking_seen = False
        self.set_phase("queuing")
        time.sleep(config.TAP_SETTLE)

    def _drop_wants_hold(self, screen) -> bool:
        """True if a hold-drop (Angel / Demon / Nova) is on screen. These show
        "TAP AND HOLD!" centred near the bottom and ONLY open on a press-and-hold —
        tap-only loops forever on them. OCR a tight band so the check stays cheap."""
        return vision.find_text(screen, "HOLD", region=config.DROP_HOLD_TEXT_REGION) is not None

    def _handle_skin_popup(self, screen) -> bool:
        """A drop awarded a SKIN -> "NEW <rarity> SKIN!" popup with CONTINUE / EQUIP NOW.
        Read the skin name + rarity (progression tracking), then tap CONTINUE (not EQUIP)
        so the drop flow continues. Returns True iff it handled one."""
        if not states.is_skin_popup(screen):
            return False
        rarity = "?"
        name_parts = []
        for text, _ in vision.read_lines(screen):
            u = text.upper()
            if "SKIN" in u and "NEW" in u:  # "NEW EPIC SKIN!"
                rarity = (u.replace("NEW", "").replace("SKIN", "").replace("!", "").strip()) or "?"
            elif (
                # "EQUIPNOW" = the documented OCR space-collapse of "EQUIP NOW"
                u in ("CONTINUE", "EQUIP NOW", "EQUIPNOW", "EQUIP", "NOW")
                or u.startswith("CURRENT")
                or "STORIES" in u
            ):
                continue  # UI chrome / collection
            else:
                name_parts.append(text)
        name = " ".join(name_parts).strip() or "?"
        self.dl.event("skin_reward", skin=name, rarity=rarity)
        self.rewards.add_event({"kind": "skin", "name": name, "rarity": rarity})
        self.log(f"skin reward: {name} ({rarity}) -> CONTINUE")
        self.tap(config.CONTINUE_BUTTON, kind="continue")
        return True

    def _handle_ceremony(self, screen) -> bool:
        """Self-heal a brawler-unlock CEREMONY screen (r8, live incident 2026-06-11:
        a fullscreen KAZE unlock recover-looped for HOURS). These are classified by the
        bottom-band green CTA + the CHOOSE-A-BRAWLER title (states), NOT templates — the
        stylized banner text OCR-mangles.

        Two shapes:
          * CHOOSE A BRAWLER (3-card chooser): pick the CENTER card, wait, then tap the
            green CHOOSE that appears. NEVER tap the blue TRY beside it (enters a preview).
          * single green CTA (KAZE unlock / ULTRA TRAIT): tap the detected green button.

        Bounded by CEREMONY_MAX_SCREENS per run cycle so a screen we can't clear escalates
        to the caller's recovery instead of looping. Returns True iff it acted. Saves an
        event shot + a session-log "ceremony_cleared" event each handled screen."""
        if states.is_choose_a_brawler(screen):
            cta = states.green_cta(screen)
            if cta is None:
                # No green CTA yet -> no card picked. Tap the CENTER card to select it;
                # the green CHOOSE then appears on the next pass.
                self.event_shot(screen, "ceremony_choose")
                self.dl.event("ceremony_cleared", kind="choose_card")
                self.log("ceremony: CHOOSE A BRAWLER -> tap center card")
                self.tap(config.CHOOSE_BRAWLER_CENTER_CARD, kind="choose_card")
                self._ceremony_count += 1
                time.sleep(config.TAP_SETTLE)
                return True
            # A card is selected -> the green CHOOSE is up. Confirm via the configured
            # CHOOSE coord (NOT the detected CTA third, to stay clear of the blue TRY).
            self.event_shot(screen, "ceremony_choose_confirm")
            self.dl.event("ceremony_cleared", kind="choose_confirm")
            self.log("ceremony: CHOOSE A BRAWLER -> CHOOSE (never TRY)")
            self.tap(config.CHOOSE_BRAWLER_CONFIRM, kind="choose_confirm")
            self._ceremony_count += 1
            time.sleep(config.TAP_SETTLE)
            return True
        cta = states.green_cta(screen)
        if cta is not None:  # single-CTA ceremony (LET'S GO / GOT IT)
            self.event_shot(screen, "ceremony")
            self.dl.event("ceremony_cleared", kind="cta", x=cta[0], y=cta[1])
            self.log(f"ceremony: green CTA at {cta} -> tap")
            self.tap(cta, kind="ceremony_cta")
            self._ceremony_count += 1
            time.sleep(config.TAP_SETTLE)
            return True
        return False

    def _try_dismiss_ladder(self, screen) -> bool:
        """ONE bounded pass of the known dismissers before recover() force-stops (r8
        self-heal): the new green-CTA ceremony check, then close_x, then proceed /
        CONTINUE. Returns True iff it acted (so the caller can re-loop instead of
        force-stopping). Cheap, idempotent, and bounded to a single pass per call —
        the caller decides whether to escalate."""
        if self._ceremony_count < config.CEREMONY_MAX_SCREENS and self._handle_ceremony(screen):
            return True
        m = vision.find(screen, "close_x")
        if m is not None:
            self.tap(m.center, kind="close_x")
            time.sleep(config.TAP_SETTLE)
            return True
        if vision.find(screen, "proceed") is not None:
            self.tap(config.PROCEED_BUTTON, kind="proceed")
            time.sleep(config.TAP_SETTLE)
            return True
        cont = vision.find_text(screen, "CONTINUE")
        if cont is not None:
            self.tap(cont, kind="continue")
            time.sleep(config.TAP_SETTLE)
            return True
        return False

    def navigate_to_trio_showdown(self) -> bool:
        """Open the event/mode selector and select Trio Showdown; return whether it is
        now selected. OCR-driven (vision.find_text) so it works regardless of where the
        SHOWDOWN card sits — event cards shuffle as events rotate, so a fixed coordinate
        would break. Self-validating: the Solo/Duo/Trio chooser only appears for the real
        Showdown card, so if we mis-tapped (e.g. the 'Loaded Duo Showdown' event) the TRIO
        lookup fails and we back out — phase_at_menu then stops on wrong_mode (unchanged
        safety net). Best-effort: if Trio's card needs horizontal scrolling, this returns
        False rather than guessing.

        The picker gained category tabs (measured 2026-09-17): it opens on SPECIAL EVENTS
        and Showdown sits under TROPHIES, behind a one-time "NEW!" cover on any card this
        account has never opened. So: tab (when there is one), peel the covers, then the
        card and TRIO as before. Every tap but the banner is a label OCR just found."""
        adb.tap(*config.MODE_BANNER)  # open the Events / mode selector
        time.sleep(2.5)  # let the event cards load in
        screen = adb.screencap()
        tab = vision.find_text(screen, config.MODE_TAB_TROPHIES, exact=True)
        if tab is not None:
            adb.tap(*tab)  # switch to the trophy game modes
            self.log(f"mode-selector navigate -> {config.MODE_TAB_TROPHIES} tab")
            time.sleep(config.MODE_TAB_WAIT_S)
            screen = adb.screencap()
        else:  # an older picker with no tab bar: the cards are already on screen
            self.log("mode-selector navigate -> no tab bar, using the open picker")
        peeled = 0
        for text, _conf, (cx, cy) in vision.read_lines_boxes(screen):
            # One OCR pass finds every cover at once. The tab-bar badges read "NEW"
            # (no bang) and sit below the cards, so BOTH tests have to pass.
            if text.strip().upper() != config.MODE_NEW_COVER:
                continue
            if cy >= config.MODE_CARD_AREA_MAX_Y:
                continue
            adb.tap(cx, cy)  # peel the cover: reveals the card name, selects nothing
            peeled += 1
            time.sleep(config.MODE_COVER_WAIT_S)
        if peeled:
            self.log(f"mode-selector navigate -> peeled {peeled} {config.MODE_NEW_COVER} cover(s)")
            time.sleep(config.MODE_CARD_WAIT_S)  # let the last card settle before we read it
            screen = adb.screencap()
        card = vision.find_text(screen, "SHOWDOWN", exact=True)  # the standard Showdown card
        if card is None:
            self.log("mode-selector navigate -> no SHOWDOWN card")
            adb.keyevent(4)
            time.sleep(1.0)  # back out to the menu
            return False
        adb.tap(*card)  # open the Solo / Duo / Trio chooser (the card expands in place)
        self.log("mode-selector navigate -> SHOWDOWN card")
        time.sleep(config.MODE_CARD_WAIT_S)
        screen = adb.screencap()
        trio = vision.find_text(screen, "TRIO", exact=True)
        if trio is None:
            self.log("mode-selector navigate -> no TRIO in the chooser")
            adb.keyevent(4)  # close the expanded card...
            time.sleep(1.0)
            adb.keyevent(4)  # ...and the picker under it
            time.sleep(1.0)
            return False
        adb.tap(*trio)  # select Trio Showdown -> back to menu
        time.sleep(config.MODE_CARD_WAIT_S)
        ok = states.is_trio_showdown_selected(adb.screencap())
        self.log(f"mode-selector navigate -> Trio selected: {ok}")
        return ok

    def _note_recalib(
        self, surface: str, suspicion: str | None, threshold: int, per_day: bool = False
    ) -> None:
        """Fold one surface observation into the season-rollover tripwire
        (ops-resilience.md §B): bump/reset the per-surface streak in
        data/<acct>/recalib.json and, at the threshold, emit the `recalibrate`
        event — which fans out through notify.maybe_alert (phone push) and the
        control panel's session tailer. One alert per surface per day
        (recalib.record's cooldown); the streak keeps counting meanwhile.
        ADVISORY ONLY and fully wrapped: a tripwire hiccup must never disturb the
        startup task that reported the observation."""
        if not config.RECALIB_TRIPWIRE:
            return
        if suspicion == recalib.SKIP:
            # Nav failure = the surface was never observed (review #56 MINOR):
            # neither bump NOR reset — a reskin that also breaks a nav needle
            # must not have its streak erased by the failed attempts.
            return
        try:
            fire, streak = recalib.record(
                config.DATA_DIR,
                surface,
                suspicion is not None,
                threshold,
                per_day=per_day,
            )
            if suspicion:
                self.log(f"recalib tripwire [{surface}]: {suspicion} (streak {streak}/{threshold})")
            if fire:
                self.dl.event(
                    "recalibrate",
                    surface=surface,
                    detail=f"{suspicion} — {streak} in a row; season reskin? "
                    f"re-probe and recalibrate config",
                )
        except Exception as e:
            try:  # even the error report is best-effort
                self.log(f"recalib tripwire error (ignored): {e!r}")
            except Exception:
                pass

    def _do_select_brawler(self) -> None:
        """Once-per-session (and at reselect points): select the brawler the FARM PLAN
        calls for (core/farmplan.py — set from the control panel). No planned target
        (ladder's default pick) keeps today's behavior: open BRAWLERS, sort by Least
        Trophies, filters OFF, select the top-left card. A planned target uses the
        alphabetical select-by-name flow and falls back to lowest-trophy if it can't
        be selected. Never taps Upgrade (Gems/Coins). Wrapped so a hiccup is logged,
        never crashes."""
        resolved_ok = False
        try:
            target, goal, owned = farmplan.resolve_target(self.api, exclude=self._rotated)
            self._farm_goal = goal
            resolved_ok = True
        except Exception as e:  # API/plan hiccup -> plain lowest-trophy fallback
            target, owned = None, []
            self.dl.event("farmplan_error", err=repr(e))
            self.log(f"farm plan error (falling back to lowest): {e!r}")
        # Prestige maxed (owner rule 2026-06-14): a SUCCESSFUL resolve with target=None
        # means nothing is under the goal. Do NOT fall through to the in-game
        # lowest-trophy sort — the visual prestige-reset fools it into selecting a
        # >goal brawler. Hold the current brawler instead. Guarded on resolved_ok so a
        # transient API error still takes the legacy lowest fallback (not a false
        # "maxed"); ladder's target=None is the legit in-game lowest selection, so the
        # mode check short-circuits prestige only.
        if resolved_ok and not target:
            try:
                if farmplan.load_plan().get("mode") == "prestige":
                    if not self._account_maxed:
                        self._account_maxed = True
                        self.log(
                            f"prestige: no brawler under {self._farm_goal} — account "
                            "maxed; not selecting (won't grind over the goal)"
                        )
                        self.dl.event("account_maxed", goal=self._farm_goal)
                    return
            except Exception:
                pass  # plan unreadable -> fall through to the legacy lowest flow
        # The quest-aware pick (plan flag, ladder only): the cards the session-start
        # quests visit read decide which owned brawler to farm, and the plan's own target
        # wins whenever it clears a quest itself. resolved_ok gates it because a failed
        # resolve leaves no roster to pick from, and the flag is cleared here because the
        # pick applies at session start only: a mid-session reselect is the plan's.
        if resolved_ok and self._quest_aware:
            self._quest_aware = False
            target = self._quest_pick(target, owned, self._quest_goal) or target
        if target:
            self.log(f"brawler select: planned -> {target} (goal {self._farm_goal})")
            try:
                name, suspicion = brawlers.select_brawler_by_name_checked(target, owned, self.log)
                # Tripwire (ops-resilience.md §B): screen verified but the grid OCR
                # read 0 owned names over a full scroll, two sessions in a row.
                # (The lowest-trophy fallback path has no grid OCR, so no signal.)
                self._note_recalib("brawlers", suspicion, config.RECALIB_BRAWLERS_STREAK)
                if name:
                    self.dl.event(
                        "select_brawler",
                        brawler=name,
                        planned=True,
                        goal=self._farm_goal,
                    )
                    self._farm_brawler = name
                    return
                self.log(f"planned select failed for {target} -> lowest-trophy fallback")
            except Exception as e:
                self.dl.event("select_brawler_error", err=repr(e))
                self.log(f"brawler select error: {e!r}")
        self.log("brawler select: lowest-trophy")
        try:
            ocr_name = brawlers.select_lowest_trophy_brawler(self.log)
        except Exception as e:
            self.dl.event("select_brawler_error", err=repr(e))
            self.log(f"brawler select error: {e!r}")
            self._capture_farm_brawler()
            return
        self._capture_farm_brawler()
        # v4 §12 (TANUKI fix): the detail screen header shows the EQUIPPED SKIN's
        # name ("TANUKI"), not necessarily the brawler ("SPROUT") — prefer the
        # API-canonical lowest-trophy name just captured for the narration event
        # (checklist label + farming record); the OCR text is the fallback only.
        name = self._farm_brawler or ocr_name
        self.dl.event("select_brawler", brawler=name)

    def _do_set_dnd(self) -> None:
        """Once-per-session: set the in-game invite mutes (MUTE FRIENDS 24h, MUTE RECENT
        TEAMMATES 30d) so team invites can't pop a farm-blocking modal. (Round 6 dropped
        the separate push-notification block — unnecessary.) The leg verifies its screen
        and bails safely if a coordinate is stale. Wrapped so a hiccup is logged but never
        crashes the farm session."""
        self.log("dnd: set invite mutes")
        try:
            result = settings.apply_dnd(self.log)
            self.dl.event("dnd", **result)
        except Exception as e:
            self.dl.event("dnd_error", err=repr(e))
            self.log(f"dnd error: {e!r}")

    def _quest_pick_goal(self) -> int | None:
        """The goal the quest pick caps its candidates at when this session's plan asks
        for the pick (the flag on AND ladder mode: prestige ignores it, the owner's task 9
        decision), else None for off.

        That goal is the PLAN's goal_trophies, never self._farm_goal: in ladder mode the
        farm goal is the live STEP goal (the next 100-line above the roster minimum), so
        capping at it would drop every owned brawler between the step and the goal -- most
        of the roster, most of the time. A plan that won't read counts as off, so the
        quests screen is never visited on a guess."""
        try:
            plan = farmplan.load_plan()
            goal = int(plan.get("goal_trophies") or farmplan.PRESTIGE_GOAL)
        except Exception as e:
            self.log(f"quest pick: plan unreadable ({e!r}) -> off")
            return None
        if not plan.get("quest_aware") or plan.get("mode") != "ladder":
            return None
        return goal

    def _do_quest_visit(self) -> None:
        """The session's ONE quests visit, run before the brawler select when the plan's
        quest pick is on: it activates a NEW MEGA QUEST if one is offered, exactly like
        the recurring trigger does, and reads the quest cards on the way out for
        _do_select_brawler. Same taps, no second navigation. A hiccup leaves the cards
        None (the select then reports "unreadable" and falls back to the plan), logged,
        never crashing the farm session."""
        self.log("quests: activate + read the cards for the brawler pick")
        try:
            self._quest_cards, activated = quests.visit(self.log)
            self.log(f"quests: read {len(self._quest_cards)} quest cards")
            if activated:
                # Activating consumed the gold badge, so the recurring trigger below will
                # never fire for it: log the row and take the thrash backstop's step here,
                # exactly as _do_mega_quest would have. Nothing offered logs nothing --
                # the trigger only ever fires on the badge, so a row at every session
                # start would be noise.
                self._mega_quest_streak += 1
                self.dl.event("mega_quest", activated=True)
        except Exception as e:
            self._quest_cards = None
            self.log(f"quest read error: {e!r}")

    def _quest_pick(self, target: str | None, owned: list[str], goal: int) -> str | None:
        """The brawler today's quests call for, or None to keep the plan's own answer.

        The candidates are the owned brawlers still UNDER ``goal`` (the owner's cap:
        the quest pick never farms a brawler past the goal) plus the plan's target, which
        the plan already chose. Emits exactly one quest_pick event either way, so the feed
        always says why: `chosen` a quest brawler, `target_clears` the target clears one
        itself, `no_candidate` nobody owned does, `unreadable` the visit read nothing. An
        API hiccup on the trophy read is logged and leaves the plan's answer alone."""
        if not self._quest_cards:
            self.dl.event(
                "quest_pick", brawler=None, quest=None, reason="unreadable", target=target
            )
            self.log("quest pick: no quest cards read -> using the plan")
            return None
        try:
            trophies = farmplan.owned_trophy_map(self.api)
        except Exception as e:
            self.dl.event("quest_pick_error", err=repr(e))
            self.log(f"quest pick error: {e!r}")
            return None
        wanted = questpick.norm_name(target)
        candidates = [
            name
            for name in owned
            if questpick.norm_name(name) == wanted
            or trophies.get(questpick.norm_name(name), 0) < goal
        ]
        found = questpick.resolve_quest(
            questpick.parse(self._quest_cards), candidates, trophies=trophies, prefer=target
        )
        if found is None:
            self.dl.event(
                "quest_pick", brawler=None, quest=None, reason="no_candidate", target=target
            )
            self.log("quest pick: no owned brawler clears a quest -> using the plan")
            return None
        name, quest = found
        reason = "target_clears" if questpick.norm_name(name) == wanted else "chosen"
        self.dl.event("quest_pick", brawler=name, quest=quest.line, reason=reason, target=target)
        self.log(f"quest pick: {name} clears {quest.line!r} ({reason})")
        return name

    def _do_mega_quest(self) -> None:
        """Activate the available NEW MEGA QUEST (open QUESTS, tap the card, back to menu).
        Always-on, triggered whenever the QUESTS button is gold. Wrapped so a hiccup is
        logged but never crashes the farm session."""
        self.log("mega quest: activate new")
        try:
            activated = quests.activate_new_mega_quest(self.log)
            self.dl.event("mega_quest", activated=activated)
        except Exception as e:
            self.dl.event("mega_quest_error", err=repr(e))
            self.log(f"mega quest error: {e!r}")

    def advance_results(self, screen) -> None:
        """Move forward through the two results screens toward the menu.
        screen 1 (PROCEED present) -> tap PROCEED;  screen 2 (EXIT) -> tap EXIT;
        if only PLAY AGAIN is visible, tap that. Each tap re-evaluates next loop."""
        if vision.find(screen, "proceed") is not None:
            self.tap(config.PROCEED_BUTTON, kind="proceed")
        elif vision.find(screen, "exit") is not None:
            self.tap(config.EXIT_BUTTON, kind="exit")
        else:
            self.tap(config.PLAY_AGAIN_BUTTON, kind="play_again")
        time.sleep(config.TAP_SETTLE)

    def _enter_match(self) -> None:
        """Common bookkeeping for entering a FRESH match: reset the action timers and
        the gas tracker (gas state must never leak across matches — a stale 'gassed'
        edge whose fraction now sits in the hysteresis dead zone would stick forever).
        A disconnect REJOIN deliberately does not come through here: it's the same
        match, so the gas state is still true."""
        self.set_phase("playing")
        self.last_attack = self.last_move = 0.0
        self._gas.reset()
        self._gas_edges = set()
        # Phase B: bush/relocation state must not leak across matches either.
        self._bush_jitter_at = 0.0
        self._relocations = 0
        self._bush_logged = None

    def phase_queuing(self, screen, state) -> None:
        elapsed = time.monotonic() - self.phase_started
        if state == State.IN_MATCH:
            self._enter_match()
            return
        if state in (State.RESULTS, State.TROPHY_SCREEN):
            self.set_phase("returning")
            return
        if state == State.MATCHMAKING:
            self.matchmaking_seen = True
            return
        if state == State.MENU:
            if elapsed > QUEUE_REPRESS:
                # Still on the menu this long after PLAY? A TEAM INVITE modal may be
                # sitting over it eating the taps (PLAY can stay visible beside the
                # modal, so this still classifies MENU). Check before re-pressing.
                if states.is_team_invite(screen):
                    self.handle_team_invite(screen)
                    self.phase_started = time.monotonic()
                    return
                self.tap(config.PLAY_BUTTON, kind="play_repress")
                self.phase_started = time.monotonic()
            return
        # UNKNOWN (loading). Fallback to playing once matchmaking has been seen — safe
        # now because phase_playing only moves/attacks when state == IN_MATCH.
        if self.matchmaking_seen:
            self._enter_match()
            return
        if elapsed > QUEUE_TIMEOUT:
            self.recover("queue_timeout")

    def _end_match(self) -> None:
        self.games_played += 1
        self.disconnect_count = 0
        self.log(f"match ended (game #{self.games_played})")
        self.maybe_log_battlelog()  # (the just-ended game lands a few min later)
        self.set_phase("returning")

    # --- Phase B (bush-hide + gas relocation) movement helpers ----------------
    # All fully guarded: any failure returns the WANDER fallback so the existing
    # random/gas-repelled drift always runs. Kill-switch: config.BUSH_HIDE.

    def _bush_action(self, screen, gas_fr):
        """Compute the bush decision for this move-tick (or None to wander).
        Detects bush clusters and runs the PURE match_vision.bush_decision over
        them + the current gas signals. Wrapped: a detector exception logs once
        and degrades to wander (None)."""
        if not config.BUSH_HIDE:
            return None
        try:
            clusters = match_vision.bush_clusters(screen)
            return match_vision.bush_decision(
                clusters,
                config.BUSH_SELF_POS,
                (config.SCREEN_W, config.SCREEN_H),
                set(self._gas_edges),
                gas_fr,
            )
        except Exception as e:  # never let CV trouble break the move loop
            self.dl.event("bush_error", err=repr(e))
            return None

    def _heading_to(self, target) -> float:
        """Joystick heading that drives the brawler toward screen point ``target``
        from the self anchor — the swipe vector direction equals the in-game move
        direction, so the screen-space (target - self) bearing is the heading."""
        sx, sy = config.BUSH_SELF_POS
        return math.atan2(target[1] - sy, target[0] - sx)

    def _swipe_heading(self, heading: float, mag_scale: float = 1.0) -> None:
        """One joystick swipe along ``heading`` (radians), reusing the wander's
        jittered magnitude/duration so a steered move looks like any other move."""
        mag = config.MOVE_RADIUS * mag_scale * random.uniform(0.75, 1.0)
        ox, oy = config.MOVE_ORIGIN
        dx = int(mag * math.cos(heading))
        dy = int(mag * math.sin(heading))
        dur = int(config.MOVE_DURATION_MS * random.uniform(0.8, 1.25))
        adb.swipe(ox, oy, ox + dx, oy + dy, dur)

    def _wander_move(self, gas_fr) -> None:
        """The existing random-drift move (plus the Phase A gas repulsion) — the
        always-available fallback. Drifts the heading by a random turn, biases it
        away from gassed edges when gas is on screen, and swipes."""
        self._heading += random.uniform(-config.MOVE_TURN_MAX, config.MOVE_TURN_MAX)
        if gas_fr is not None and self._gas_edges:
            self._heading = match_vision.gas_bias_heading(self._heading, gas_fr, self._gas_edges)
        self._swipe_heading(self._heading)

    def _steer_to_bush(self, action, now: float, gas_fr=None) -> None:
        """Act on a non-wander BushAction. RELOCATE: head toward the central bush,
        capped at config.BUSH_MAX_RELOCATIONS per gas event (over the cap -> fall
        back to the gas-repelled wander, which still flees). HIDE: head toward the
        nearest bush; once on top of it (at_target) hold and emit a small periodic
        micro-jitter so we never look true-AFK while concealed. ``gas_fr`` is the
        current gas fractions (or None) so the cap-fallback wander keeps its gas
        repulsion. Logs sparsely (only when the behavior changes)."""
        if action.kind == "relocate":
            if self._relocations >= config.BUSH_MAX_RELOCATIONS:
                self._wander_move(gas_fr)  # cap hit: the wander's gas repulsion flees
                return
            self._relocations += 1
            self._heading = self._heading_to(action.target)
            self._swipe_heading(self._heading)
            if self._bush_logged != "gas_relocate":
                self._bush_logged = "gas_relocate"
                self.dl.event("gas_relocate", target=list(action.target), n=self._relocations)
                self.log(f"bush: gas closing -> relocate toward {action.target}")
            return
        # HIDE
        if action.at_target:
            # Hidden: hold position, micro-jitter on its own slow cadence so we
            # register input (anti-AFK) without leaving the bush.
            interval = random.uniform(
                config.BUSH_JITTER_MIN_INTERVAL, config.BUSH_JITTER_MAX_INTERVAL
            )
            if now - self._bush_jitter_at >= interval:
                self._bush_jitter_at = now
                self._swipe_heading(
                    random.uniform(0, 2 * math.pi),
                    mag_scale=config.BUSH_JITTER_RADIUS / config.MOVE_RADIUS,
                )
            if self._bush_logged != "bush_hide":
                self._bush_logged = "bush_hide"
                self.dl.event("bush_hide", target=list(action.target))
                self.log(f"bush: hidden in bush near {action.target}")
            return
        # Not yet there: steer toward the nearest bush.
        self._heading = self._heading_to(action.target)
        self._swipe_heading(self._heading)
        if self._bush_logged != "bush_approach":
            self._bush_logged = "bush_approach"
            self.log(f"bush: approaching bush at {action.target}")

    def phase_playing(self, screen, state) -> None:
        if state in (State.RESULTS, State.TROPHY_SCREEN):
            self._end_match()
            return
        if state == State.MENU:
            self.set_phase("at_menu")
            return
        if time.monotonic() - self.phase_started > MATCH_TIMEOUT:
            self.recover("match_timeout")
            return
        if state != State.IN_MATCH:
            return  # loading / transient — do NOT blind-swipe (only act when in-match)
        # In-match modal scan (r8 self-heal, live incident: a gray "Server error:
        # 43" box sat over the match forever — the PLAYING phase never scanned for
        # modals). Cheap, but throttled to every Nth in-match iteration to protect the
        # playing-loop latency. The disconnect/RELOAD modal is a gray box too, but it's
        # classified DISCONNECT and handled before this phase ever runs, so the gate
        # only sees the no-reload error variant. Tap OK -> the game returns to menu/
        # launcher; the returning flow + existing recovery handle the rest.
        self._playing_modal_i += 1
        if self._playing_modal_i % config.INGAME_MODAL_CHECK_EVERY == 0:
            if states.in_match_modal(screen):
                self.dl.event("ingame_modal_cleared")
                self.event_shot(screen, "ingame_modal")
                self.log("in-match modal detected -> tap OK")
                self.tap(config.INGAME_MODAL_OK_BUTTON, kind="ingame_modal_ok")
                time.sleep(config.TAP_SETTLE)
                self.set_phase("returning")
                return
        now = time.monotonic()
        # Phase A: read the gas edge-bands every in-match frame (4 cheap ROI checks)
        # and fold them through the hysteresis tracker. The fractions feed the heading
        # bias below; logging only on change keeps the session log readable.
        gas_fr: dict[str, float] | None = None
        if config.GAS_AWARE:
            gas_fr = match_vision.gas_fractions(screen)
            gassed = self._gas.update(gas_fr)
            if gassed != self._gas_edges:
                self._gas_edges = gassed
                self.dl.event("gas_edges", edges=sorted(gassed))
                self.log(f"gas edges: {sorted(gassed) or 'clear'}")
                if not gassed:
                    # Gas cleared (e.g. the death/respawn camera pan) -> reset the
                    # per-gas-event relocation cap so the next real close-in gets a
                    # fresh budget (Phase B).
                    self._relocations = 0
        # Movement keeps us active (prevents idle-disconnect) and makes us roam. We drift the
        # heading by a random turn and swipe the joystick that way, on a randomized gap and
        # with a jittered magnitude/duration — so neither the cadence nor the direction forms
        # the periodic pattern bot detection looks for. Re-roll the next gap after each move.
        # With gas on screen the random drift gets a repulsion bias AWAY from the gassed
        # edges (Phase A) — randomness stays, plus a survival gradient.
        #
        # Phase B (r8): when bush-hide is on, a detected bush STEERS this wander — toward
        # the nearest bush to hide, or (gas closing) a more-central bush to relocate to.
        # Computed once per move-tick (CV only when we're about to move, never on the hot
        # attack cadence). EVERY failure path (no bush / detector exception / gate not met)
        # degrades to the random+gas wander below: bush logic only ever REPLACES the random
        # heading with a steer, never suppresses the move itself (so idle-disconnect
        # prevention is preserved). It also never overrides the attack/survival reactions.
        if now - self.last_move >= self._move_gap:
            action = self._bush_action(screen, gas_fr)
            if action is not None and action.kind != "wander":
                self._steer_to_bush(action, now, gas_fr)
            else:
                self._wander_move(gas_fr)
                if self._bush_logged is not None:
                    self._bush_logged = None  # left bush behavior -> allow a fresh log
            self.last_move = now
            self._move_gap = random.uniform(config.MOVE_INTERVAL_MIN, config.MOVE_INTERVAL_MAX)
        if now - self.last_attack >= self._attack_gap:
            self.tap(self._jitter(config.ATTACK_POINT))  # auto-aim attack (slight pos jitter)
            self.tap(self._jitter(config.SUPER_BUTTON))  # harmless no-op unless charged
            self.last_attack = now
            self._attack_gap = config.ATTACK_INTERVAL + random.uniform(
                -config.ATTACK_JITTER, config.ATTACK_JITTER
            )
        # Phase D: tap ability buttons that are lit with their ready color (gadget=
        # green / super=yellow / hypercharge=purple). Worst case is a tap on a
        # mis-read button face, which the game ignores — but a per-button cooldown
        # still bounds it so a stuck-lit read can't turn into tap spam.
        if config.ABILITY_BUTTONS_ENABLED and config.ABILITY_BUTTONS:
            for name, frac in match_vision.ready_abilities(screen).items():
                if now - self._ability_last.get(name, 0.0) < config.ABILITY_TAP_COOLDOWN:
                    continue
                self._ability_last[name] = now
                self.tap(
                    self._jitter(config.ABILITY_BUTTONS[name]["tap"]),
                    kind=f"ability_{name}",
                )
                self.log(f"ability {name} ready ({frac:.2f}) -> tap")

    def handle_disconnect(self, screen) -> None:
        """Any modal carrying the RELOAD button -> tap RELOAD to rejoin. Covers idle/network
        'kicked for inactivity', the 'Another device' modal, AND server-error screens (e.g.
        "Server 43") — they all share the same RELOAD button, which is exactly what
        states.classify keys on to flag DISCONNECT. We tap the button where it's actually
        DETECTED (not a fixed coordinate) so it works even if an error lays it out elsewhere.
        Bounded so we never loop forever."""
        other = states.is_other_device(screen)
        self.disconnect_count += 1
        self.dl.event("disconnect", count=self.disconnect_count, other_device=other)
        self.event_shot(screen, "other_device" if other else "disconnect")
        # "Another device" modal: reload for now; in the multi-account stage
        # set RECONNECT_ON_OTHER_DEVICE=False so this aborts + alerts instead.
        if other and not config.RECONNECT_ON_OTHER_DEVICE:
            self.log("ANOTHER-DEVICE modal -> abort (RECONNECT_ON_OTHER_DEVICE=False)")
            self.stop("other_device")  # future: send an alert here
            return
        label = "another-device" if other else "idle/network"
        self.log(f"disconnect modal ({label}) -> RELOAD #{self.disconnect_count}")
        if self.disconnect_count > 8:
            self.recover("disconnect_loop")
            return
        # Tap RELOAD where it's actually detected so any reload-bearing modal works regardless
        # of layout; fall back to the configured coordinate if the re-find happens to miss.
        m = vision.find(screen, "reload", threshold=0.9)
        self.tap(m.center if m is not None else config.RELOAD_BUTTON, kind="reload")
        time.sleep(4.0)
        # After rejoin it loads back into the match; resume playing (with movement).
        self.last_attack = 0.0
        self.last_move = 0.0
        self.set_phase("playing")

    def handle_daily_streak(self, screen) -> None:
        """The DAILY STREAK login-reward popup (auto-appears once a day, blocks the menu).
        Verify then act, never a blind tap: (1) tap the OCR'd CLAIM word to collect the free
        reward — claiming clears it for the day, whereas its ✕ can re-pop; (2) no CLAIM (the
        day-1 variant has none) -> tap the popup's red ✕, matched at a local threshold and
        only in its top-right area; (3) neither -> one Android BACK. Then switch to the
        returning flow, which taps through the Starr-Drop reveal that follows and any rank-up
        trophy screen back to the menu. Never costs gems/money. A popup none of these clears
        is caught by the repeat guard in _unknown_modal (recover("daily_streak_stuck"))."""
        claim = vision.find_text(screen, "CLAIM", region=config.DAILY_STREAK_CLAIM_REGION)
        if claim is not None:
            self.dl.event("daily_streak_claim")
            self.log("DAILY STREAK popup -> CLAIM")
            self.tap(claim, kind="daily_streak_claim")
        else:
            m = vision.find(screen, "close_x", threshold=config.DAILY_STREAK_CLOSE_X_THRESHOLD)
            x1, y1, x2, y2 = config.DAILY_STREAK_CLOSE_X_REGION
            if m is not None and x1 <= m.x < x2 and y1 <= m.y < y2:
                self.dl.event("popup_close", popup="daily_streak", action="close_x")
                self.log(f"DAILY STREAK popup (no CLAIM) -> close ✕ ({m.confidence:.3f})")
                self.tap(m.center, kind="close_x")
            else:
                self.dl.event("popup_close", popup="daily_streak", action="back")
                self.log("DAILY STREAK popup (no CLAIM, no ✕) -> BACK")
                adb.keyevent(4)
        time.sleep(config.TAP_SETTLE)
        self.set_phase("returning")  # clear the Starr-drop reveal + any trophy screen

    def handle_team_invite(self, screen) -> None:
        """A TEAM INVITE modal is up (someone wants to team — it blocks the menu and eats
        PLAY taps until it times out). Decline it: tap MUTE (stops repeat invites from
        that player), then REJECT. REJECT is a FIXED coordinate — its stylized label
        OCR-mangles ("REJEGT") — mirrored across from the green ACCEPT, which is never
        tapped. The in-game DND mutes (core/settings.py) make this rare, not impossible
        (e.g. a non-friend invite)."""
        self.dl.event("team_invite_decline")
        self.log("TEAM INVITE popup -> MUTE + REJECT")
        m = vision.find_text(screen, "MUTE")  # prefer the live label; fixed fallback
        self.tap(m if m is not None else config.INVITE_MUTE_BUTTON, kind="invite_mute")
        time.sleep(0.6)
        self.tap(config.INVITE_REJECT_BUTTON, kind="invite_reject")
        time.sleep(config.TAP_SETTLE)

    def _unknown_modal(self, screen, state) -> bool:
        """The off-hot-path modal checks on UNKNOWN frames (daily streak, team invite).
        Runs the matching handler through the repeat guard and returns True when one ran,
        so the loop skips the rest of the tick; returns False (and clears the guard: some
        other branch is about to run) otherwise."""
        if state == State.UNKNOWN:
            # DAILY STREAK login-reward popup: blocks the menu, close_x misses its ✕, and
            # it survives game-relaunch recovery -> a match_timeout loop.
            if states.is_daily_streak(screen):
                self._guarded_handler("daily_streak", self.handle_daily_streak, screen)
                return True
            # TEAM INVITE modal. It can classify as UNKNOWN (menu fully blocked); the
            # MENU-but-PLAY-eaten variant is caught in phase_queuing before the PLAY re-press.
            if states.is_team_invite(screen):
                self._guarded_handler("team_invite", self.handle_team_invite, screen)
                return True
        self._clear_handler_repeat()
        return False

    def _guarded_handler(self, name: str, handler, screen) -> None:
        """Run `handler(screen)` unless the same handler branch has now run
        config.HANDLER_REPEAT_MAX loop iterations in a row; then recover("<name>_stuck")
        instead. These branches `continue` past the UNKNOWN streak counter and
        check_freeze, so without this a modal the handler can't clear loops forever
        (2026-11-04: 40 minutes and 1442 taps on a Daily Streak popup)."""
        if name == self._repeat_handler:
            self._repeat_count += 1
        else:
            self._repeat_handler, self._repeat_count = name, 1
        if self._repeat_count >= config.HANDLER_REPEAT_MAX:
            self._clear_handler_repeat()
            self.recover(f"{name}_stuck")
            return
        handler(screen)

    def _clear_handler_repeat(self) -> None:
        self._repeat_handler, self._repeat_count = None, 0

    def handle_popup(self, screen) -> None:
        """Promo/news/offer popup → close it. Tiered strategy (per the scenarios doc):
        (1) tap the DETECTED ✕ (robust across popups whose ✕ isn't at the promo spot —
        e.g. the 'Join a more active club?' popup, whose ✕ sits left of CLOSE_X_BUTTON
        and where the template can weakly mis-match); (2) BACK, confirmed to dismiss most
        popups and to undo any accidental navigation; (3) tap-outside the card. Bounded so
        a popup we can't close doesn't loop forever."""
        self.popup_count += 1
        if self.popup_count == 1:
            self.dl.event("popup_close")
            self.log("promo popup -> close (✕)")
        if self.popup_count > 12:
            self.recover("popup_stuck")
            return
        if self.popup_count <= 2:
            # Prefer the live ✕ location; fall back to the configured promo coord.
            m = vision.find(screen, "close_x")
            point = (int(m.x + m.w / 2), int(m.y + m.h / 2)) if m else config.CLOSE_X_BUTTON
            self.tap(point, kind="close_x")
        elif self.popup_count <= 6:
            adb.keyevent(4)  # BACK — reliable across varied ✕ styles/positions
        else:
            self.tap(config.SAFE_DISMISS_POINT)  # tap-outside the card (tertiary)
        time.sleep(config.TAP_SETTLE)

    # --- recovery / stop -----------------------------------------------------

    def recover(self, reason: str) -> None:
        # Self-heal first (r8): on the FIRST recovery of a cycle, before the heavy
        # force-stop, try ONE bounded pass of the known dismissers (ceremony green-CTA,
        # close_x, proceed, CONTINUE) and always save an event shot of whatever we're
        # stuck on. If a dismisser acts, give the screen a chance to resolve on the next
        # returning loop instead of relaunching — without burning a force-stop attempt.
        # Gated to ONCE per cycle (_dismiss_tried, reset at the menu) so a screen the
        # ladder keeps "handling" but never truly clears still escalates to a real
        # force-stop (no soft-dismiss loop). Wrapped: a dismiss-ladder hiccup must
        # never block the real recovery below.
        try:
            shot = adb.screencap()
            self.event_shot(shot, f"recover_{reason}")
            if not self._dismiss_tried:
                self._dismiss_tried = True
                if self._try_dismiss_ladder(shot):
                    self.dl.event("recover_dismissed", reason=reason)
                    self.log(f"RECOVER ({reason}): dismiss-ladder acted — retrying")
                    self.last_change = time.monotonic()
                    self.set_phase("returning")
                    return
        except Exception as e:
            self.dl.event("recover_error", where="dismiss_ladder", err=repr(e))
        self.recovery_attempts += 1
        self.dl.event("recover", reason=reason, attempt=self.recovery_attempts)
        self.log(f"RECOVER ({reason}) attempt {self.recovery_attempts}")
        if self.recovery_attempts > MAX_RECOVERY:
            self.stop(f"too_many_recoveries:{reason}")
            return
        # A soft stop is already pending (stop.flag / caps): the session is over —
        # relaunching a wedged game just puts the account back ONLINE during what
        # should be its break (the 2026-06-10 network-stuck morning: a stuck end
        # screen kept Brawl Stars online for hours past the scheduled stop). Stop
        # here instead; stop()'s CLOSE_GAME_ON_STOP force-stop takes it offline.
        # (The composed reason can never equal "stop_flag" exactly, so the DND-off
        # etiquette path deliberately stays cold — we are NOT at a menu here.)
        soft = self._soft_stop_reason()
        if soft is not None:
            self.stop(f"recover_while_stopping:{soft}:{reason}")
            return
        # Game auto-update (2026-11-04): a new client opens to an update prompt that no
        # relaunch fixes. On the SECOND recovery of the run (the first relaunch did not
        # help), check the Play Store once. A failed update has already stopped the run.
        if (
            self.recovery_attempts == config.GAME_UPDATE_ON_ATTEMPT
            and not self._update_checked
            and not self._check_game_update()
        ):
            return
        try:
            adb.go_home()
            time.sleep(2)
            adb.force_stop()
            time.sleep(2)
            adb.launch_app()
            time.sleep(22)
        except Exception as e:
            self.dl.event("recover_error", err=str(e))
        self.last_change = time.monotonic()
        self.set_phase("returning")

    def _game_update_effects(self) -> game_update.Effects:
        return game_update.Effects(
            open_store=lambda: adb.open_store_page(config.BS_PACKAGE),
            foreground=adb.current_package,
            screenshot=adb.screencap,
            read_boxes=vision.read_lines_boxes,
            tap=lambda pt: self.tap(pt, kind="store_update"),
            version=lambda: adb.package_version(config.BS_PACKAGE),
            sleep=time.sleep,
            clock=time.monotonic,
        )

    def _check_game_update(self) -> bool:
        """Run the once-per-run Play Store check. True: carry on with the relaunch.
        False: the update failed and the run stopped (alert via the "stop" event)."""
        self._update_checked = True
        self.log("RECOVER: checking the Play Store for a game update")
        try:
            out = game_update.try_update(self._game_update_effects())
        except Exception as e:
            # try_update only catches AdbError: an OCR/numpy error from the screenshot
            # or read_boxes path lands here and takes the FAILED path below.
            out = game_update.Outcome(game_update.UpdateResult.FAILED, f"error: {type(e).__name__}")
        if out.result is game_update.UpdateResult.UPDATED:
            self.dl.event("game_updated", old=out.old, new=out.new)
            self.log(f"game updated {out.old} -> {out.new}")
            self.recovery_attempts = 0  # the fresh client gets the full recovery budget
            return True
        if out.result is game_update.UpdateResult.NO_UPDATE:
            self.dl.event("update_check", result="none", reason=out.reason)
            return True
        try:
            self.event_shot(adb.screencap(), "update_failed")
        except Exception:
            pass
        try:
            adb.go_home()  # leave the store before stop() closes the game
        except Exception:
            pass
        self.dl.event("update_failed", reason=out.reason)
        self.log(f"game update failed ({out.reason}) -> stopping")
        self.stop("update_failed")
        return False

    def stop(self, reason: str) -> None:
        if not self.running:
            return  # idempotent: don't double-log / re-recap
        self.running = False
        self.dl.event(
            "stop",
            reason=reason,
            games=self.games_played,
            minutes=round((time.monotonic() - self.start) / 60, 1),
        )
        self.log(f"STOP ({reason}). games={self.games_played}")
        self._log_recap()
        # Stop etiquette: a USER-initiated stop (from the control panel) hands the account
        # back to a human, so best-effort turn the in-game invite mutes (DND) OFF before
        # the game closes. ONLY the stop.flag path qualifies — scheduled cap stops
        # (max_games/max_minutes) keep DND on (the account comes right back) and the
        # >10-min "stop_flag_hard" backstop is excluded too (we're NOT at a menu there,
        # the nav would flail). The soft stop.flag stop fires from phase_at_menu, so the
        # menu nav can work. Guarded: a DND-off hiccup must never block the stop.
        if reason == "stop_flag" and config.DND_OFF_ON_STOP:
            try:
                result = settings.remove_dnd(self.log)
                self.dl.event("dnd_off", **result)
            except Exception as e:
                self.dl.event("dnd_off_error", err=repr(e))
                self.log(f"dnd-off error (continuing the stop): {e!r}")
        # Anti-ban (owner, 2026-06-10): a session END means the game CLOSES — the
        # account's server-side ONLINE time should look like a human's, not
        # "parked in the menu 20 h/day". force-stop drops the instance back to the
        # BlueStacks home screen; the next worker's ensure_game_open() relaunches.
        if config.CLOSE_GAME_ON_STOP:
            try:
                adb.force_stop()
                self.dl.event("game_closed", reason=reason)
                self.log("game closed -> account offline until the next session")
            except Exception as e:  # closing is best-effort; the stop already happened
                self.dl.event("game_close_error", err=repr(e))

    def _log_recap(self) -> None:
        """Print a session recap (trophies gained, games played, skins unlocked)
        and log it as a `recap` event. format_recap simply skips unknown figures."""
        if self._recap_done:
            return
        self._recap_done = True
        try:  # freshest end total -> accurate delta
            end_total = self.api.get_player().get("trophies")
            if end_total is not None:
                self._last_trophies = end_total
                if self._start_trophies is None:
                    self._start_trophies = end_total
        except Exception:
            pass
        troph = (
            self._last_trophies - self._start_trophies
            if self._start_trophies is not None and self._last_trophies is not None
            else None
        )
        self.dl.event(
            "recap",
            trophies=troph,
            games=self.games_played,
            skins=len(self.rewards.skins()),
        )
        for line in rewards.format_recap(
            self.rewards, trophies_gained=troph, games_played=self.games_played
        ).splitlines():
            self.log(line)

    def _soft_stop_reason(self) -> str | None:
        """Why this worker should stop at the NEXT MENU (between games): the
        graceful-stop flag (scheduler/watchdog/control panel) or a launch-time cap.
        Soft by design — sessions end at the menu, not mid-match."""
        if self._stop_flag_seen:
            return "stop_flag"
        if self.max_games and self.games_played >= self.max_games:
            return "max_games"
        if self.max_minutes and (time.monotonic() - self.start) / 60 >= self.max_minutes:
            return "max_minutes"
        return None

    def _check_stop_backstop(self) -> bool:
        """HARD backstop: if a soft stop has been pending >10 min (we never made
        it back to a menu — wedged recovery, endless match), stop anyway."""
        reason = self._soft_stop_reason()
        if reason is None:
            self._soft_stop_since = None
            return False
        if self._soft_stop_since is None:
            self._soft_stop_since = time.monotonic()
        if time.monotonic() - self._soft_stop_since > 600:
            self.stop(f"{reason}_hard")
            return True
        return False

    def check_freeze(self, screen) -> None:
        sig = vision.frame_signature(screen)
        now = time.monotonic()
        if sig != self.last_sig:
            self.last_sig = sig
            self.last_change = now
            return
        if now - self.last_change > config.FREEZE_TIMEOUT:
            self.log("freeze detected")
            self.recover("freeze")

    # --- shadow play session -------------------------------------------------

    def _play_shadow_on(self) -> bool:
        """Is the play model allowed to watch this tick? Shadow only for now: the one
        expression the driving mode widens."""
        return config.PLAY_SHADOW

    def _play_observe(self, state: State) -> None:
        """One shadow tick, on the same frame the farm just acted on. Builds the session
        on first use (the import lives here so shadow off costs nothing), then writes the
        rows it returns: feed rows only ever leave this thread. The session swallows its
        own errors; anything that still escapes turns shadow off for the rest of the
        process after one log line, because the farm is not allowed to care. Shadow off
        returns before it reads an attribute, so a controller built without __init__ (the
        tests that drive the loop directly) never trips over one."""
        if not self._play_shadow_on() or getattr(self, "_play_off", False):
            return
        try:
            if getattr(self, "play", None) is None:
                from brawlfarm.play import session

                self.play = session.PlaySession()
            for kind, fields in self.play.observe(state, self.phase):
                self.dl.event(kind, **fields)
        except Exception as e:
            self._play_off = True
            self.log(f"play shadow error (shadow off): {e!r}")

    def _play_close(self) -> None:
        """Drain the shadow session's closing rows (the summary, the file it wrote) into
        the feed. Called from _play_shutdown, which owns the try that swallows this."""
        for kind, fields in self.play.close():
            self.dl.event(kind, **fields)

    def _play_shutdown(self) -> None:
        """run()'s finally: stop the shadow session and write its closing rows. The
        attribute lookup is inside the try too, because a controller built without
        __init__ has no `play` and a finally that raises masks the real exception."""
        try:
            if self.play is not None:
                self._play_close()
        except Exception:
            pass

    # --- main loop -----------------------------------------------------------

    def ensure_game_open(self) -> bool:
        """Make sure Brawl Stars is in the foreground before farming starts.

        After a fresh emulator boot the instance sits on the BlueStacks launcher home,
        so we open the game the human way: OCR the "Brawl Stars" label and TAP the icon
        just above it, falling back to `adb.launch_app()` (monkey) if the label isn't
        visible. Returns True once the game's package is foreground. Idempotent — a no-op
        if the game is already open."""
        if adb.current_package() == config.BS_PACKAGE:
            return True
        self.log("game not open -> launching")
        method = "monkey"
        try:
            pt = vision.find_text(adb.screencap(), config.GAME_LABEL)
            if pt is not None:
                x, y = pt
                adb.tap(x, y - config.GAME_ICON_LABEL_DY)  # icon sits above its label
                method = "icon"
            else:
                adb.launch_app()
        except Exception as e:
            self.dl.event("launch_game_error", err=str(e))
            adb.launch_app()
        self.dl.event("launch_game", method=method)
        deadline = time.monotonic() + config.GAME_LAUNCH_TIMEOUT
        while time.monotonic() < deadline:
            if adb.current_package() == config.BS_PACKAGE:
                self.log(f"game launched ({method})")
                return True
            time.sleep(1.5)
        # tap didn't bring it up (wrong spot / odd home layout) -> force via monkey
        self.log("game didn't foreground -> monkey fallback")
        adb.launch_app()
        time.sleep(5)
        return adb.current_package() == config.BS_PACKAGE

    def run(self) -> None:
        adb.connect()
        w, h = adb.screen_size()
        if (w, h) != (config.SCREEN_W, config.SCREEN_H):
            self.dl.event("bad_resolution", got=[w, h])
            raise SystemExit(
                f"Display is {w}x{h}, expected {config.SCREEN_W}x{config.SCREEN_H}. "
                f"Lock BlueStacks resolution before running."
            )
        self.dl.event("start", max_games=self.max_games, max_minutes=self.max_minutes)
        self.log(f"started. session log -> {self.dl.session_path.name}")
        # A leftover stop.flag (from the stop that ended the PREVIOUS worker) must
        # not instantly stop THIS one — whoever launched us wants us running.
        try:
            (config.DATA_DIR / "stop.flag").unlink(missing_ok=True)
        except OSError:
            pass
        self.ensure_game_open()

        try:
            while self.running:
                if self._check_stop_backstop():
                    break
                try:
                    screen = adb.screencap()
                    self._adb_errors = 0  # screencap worked -> adb is alive again
                    # The panel's live preview, once a second off this very frame: no
                    # second screencap, and it swallows everything it could go wrong on.
                    preview.maybe_write(screen)
                    # Phase hint = faster anchor ordering only; never changes the State
                    # returned for a frame (see states.PHASE_ORDER).
                    state = states.classify(screen, phase=self.phase)
                    # Same frame, same label the farm just acted on: the recorder
                    # writes it only while record.flag is set, and returns False
                    # (never raises) every other tick.
                    self.recorder.observe(screen, state, self.phase)
                    # Same tick for the shadow model: it sees the states the farm handles
                    # and continues past below (disconnect, popup) too, so a match that
                    # ends in a modal is still a match it saw end.
                    self._play_observe(state)
                    if self.phase != self._prev_phase:
                        self._maybe_shot(screen, self.phase)
                        self._prev_phase = self.phase
                    # A disconnect modal can overlay any screen — handle it first so we
                    # don't mistake it for RESULTS and don't trip freeze detection.
                    if state == State.DISCONNECT:
                        self._clear_handler_repeat()
                        self.handle_disconnect(screen)
                        continue
                    if state == State.POPUP:
                        self._clear_handler_repeat()
                        self.handle_popup(screen)
                        continue
                    # DAILY STREAK and TEAM INVITE modals: OCR-detected only on UNKNOWN
                    # frames (keeps the hot path cheap), handled under a repeat guard that
                    # escalates to recover() when the same handler keeps coming back.
                    if self._unknown_modal(screen, state):
                        continue
                    # Fast re-entry: a sustained UNKNOWN can mean the game dropped to the
                    # BlueStacks/Android home (crash/exit). Don't wait out the ~6-min
                    # match_timeout — check the foreground app and relaunch at once. Only act
                    # on a POSITIVE non-game package (a real read of another app); "" (adb
                    # hiccup) or the game package falls through, so we never relaunch mid-match.
                    if state == State.UNKNOWN:
                        self._unknown_streak += 1
                        if self._unknown_streak >= 2 and self._unknown_streak % 5 == 2:
                            pkg = adb.current_package()
                            if pkg and pkg != config.BS_PACKAGE:
                                self.log(f"game not in foreground ({pkg}) -> re-entering")
                                self.dl.event("game_left_foreground", pkg=pkg)
                                self.ensure_game_open()
                                self._unknown_streak = 0
                                self.set_phase("returning")
                                continue
                    else:
                        self._unknown_streak = 0
                    self.check_freeze(screen)
                    if not self.running:
                        break
                    getattr(self, f"phase_{self.phase}")(screen, state)
                    self._loop_i += 1
                    if self._loop_i % 25 == 0:
                        self._write_status()
                        # Poll the graceful-stop flag on the same cheap cadence
                        # (~11 s); honored at the next menu by _soft_stop_reason.
                        try:
                            self._stop_flag_seen = (config.DATA_DIR / "stop.flag").exists()
                        except OSError:
                            pass
                        # record.flag rides the same cadence: opening or closing a
                        # recording session is never urgent.
                        self.recorder.poll()
                    time.sleep(config.LOOP_POLL_INTERVAL)
                except adb.AdbError as e:
                    # A slow/failed adb command (common under multi-instance contention)
                    # is transient — log + back off, don't crash. Only a sustained streak
                    # (device really wedged) escalates to a relaunch.
                    self._adb_errors += 1
                    self.dl.event("adb_error", err=str(e), streak=self._adb_errors)
                    self.log(f"adb error (streak {self._adb_errors}): {e}")
                    if self._adb_errors >= 8:
                        self._adb_errors = 0
                        self.recover("adb_errors")
                    else:
                        time.sleep(1.0)
        except KeyboardInterrupt:
            self.stop("keyboard_interrupt")
        except Exception as e:
            self.dl.event("crash", err=repr(e))
            self.log(f"CRASH: {e!r}")
            raise
        finally:
            # Close the recording session on EVERY exit (clean stop, Ctrl-C, crash)
            # so recorder.json stops claiming a live session. Never masks the
            # exception that got us here.
            try:
                self.recorder.close()
            except Exception:
                pass
            # Same for the shadow session: its thread is stopped and its summary written
            # on every exit. It swallows everything itself, so neither close can eat the
            # other's and neither can mask the exception that got us here.
            self._play_shutdown()
