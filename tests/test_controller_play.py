"""The farm loop's shadow-session wiring: the tick, the rows and the close.

With `shadow` off the controller builds nothing and writes nothing, so a worker that
never asked for the play model never pays for it. With it on, the rows the session
returns are written from the controller thread in the order it returned them, and one
failure anywhere in the path turns shadow off for the rest of the process.

Like tests/test_controller_questpick.py these drive the controller's methods directly:
the object is built without __init__ (no adb, no API, no DataLog), so the inputs are
exactly the config flag and the session stub.
"""

import math
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest

from brawlfarm.core import config
from brawlfarm.core import controller as controller_mod
from brawlfarm.core.controller import Controller
from brawlfarm.core.match_vision import BushAction
from brawlfarm.core.states import State
from brawlfarm.play import reader


class _DL:
    """Event-recording stand-in for DataLog (no disk writes)."""

    def __init__(self):
        self.events = []

    def event(self, etype, **fields):
        self.events.append((etype, fields))


class _Session:
    """PlaySession stand-in: returns canned rows and records the ticks it saw."""

    def __init__(self, rows=None, close_rows=None, boom=False):
        self.rows = rows or []
        self.close_rows = close_rows or []
        self.boom = boom
        self.ticks = []
        self.moves = []
        self.closed = 0

    def observe(self, state, phase, **move):
        if self.boom:
            raise RuntimeError("shadow blew up")
        self.ticks.append((state, phase))
        self.moves.append(move)
        return self.rows

    def close(self):
        self.closed += 1
        if self.boom:
            raise RuntimeError("shadow blew up")
        return self.close_rows


def _ctrl(play=None):
    """A controller mid-match, with the shadow attributes __init__ sets."""
    c = Controller.__new__(Controller)  # skip __init__ (ApiClient/DataLog/adb)
    c.dl = _DL()
    c.logs = []
    c.log = c.logs.append
    c.phase = "playing"
    c.play = play
    c._play_off = False
    return c


def test_shadow_off_builds_nothing_and_writes_nothing(monkeypatch) -> None:
    monkeypatch.setattr(config, "PLAY_SHADOW", False)
    c = _ctrl()
    c._play_observe(State.IN_MATCH)
    assert c._play_shadow_on() is False
    assert c.play is None  # the session (and the play import) never happened
    assert c.dl.events == []
    assert c.logs == []


def test_shadow_on_builds_the_session_once(monkeypatch) -> None:
    monkeypatch.setattr(config, "PLAY_SHADOW", True)
    built = []

    class _Factory:
        def __init__(self):
            built.append(self)
            self.ticks = []

        def observe(self, state, phase, **move):
            self.ticks.append((state, phase))
            return []

    monkeypatch.setattr("brawlfarm.play.session.PlaySession", _Factory)
    c = _ctrl()
    c._play_observe(State.IN_MATCH)
    c._play_observe(State.MENU)
    assert len(built) == 1
    assert built[0].ticks == [(State.IN_MATCH, "playing"), (State.MENU, "playing")]
    assert c.play is built[0]


def test_the_rows_reach_the_datalog_in_order(monkeypatch) -> None:
    monkeypatch.setattr(config, "PLAY_SHADOW", True)
    rows = [
        ("play_on", {"model": "play.onnx", "provider": "CPUExecutionProvider", "smoke": False}),
        ("play_fallback", {"reason": "no_model"}),
    ]
    session = _Session(rows=rows)
    c = _ctrl(session)
    c._play_observe(State.IN_MATCH)
    assert c.dl.events == rows
    assert session.ticks == [(State.IN_MATCH, "playing")]


def test_the_tick_hands_the_session_the_last_move(monkeypatch) -> None:
    monkeypatch.setattr(config, "PLAY_SHADOW", True)
    play = _Session()
    c = _ctrl(play)
    c._last_move_kind = "relocate"
    c.last_move = 12.5
    c._last_move_heading = 1.5
    c._last_move_target = (1, 2)
    c._play_observe(State.IN_MATCH)
    assert play.moves == [
        {
            "farm_move": "relocate",
            "moved_at": 12.5,
            "farm_heading": 1.5,
            "farm_target": (1, 2),
        }
    ]
    assert c._play_off is False


def test_the_tick_works_on_a_controller_without_the_move_attributes(monkeypatch) -> None:
    monkeypatch.setattr(config, "PLAY_SHADOW", True)
    play = _Session()
    c = _ctrl(play)  # built without __init__: no last_move, no _last_move_kind
    c._play_observe(State.IN_MATCH)
    assert play.moves == [
        {"farm_move": None, "moved_at": None, "farm_heading": None, "farm_target": None}
    ]
    assert c._play_off is False


def _mover(monkeypatch, action):
    """A controller whose next in-match tick is a move tick and nothing else."""
    monkeypatch.setattr(config, "BUSH_HIDE", True)
    monkeypatch.setattr(config, "GAS_AWARE", False)
    monkeypatch.setattr(config, "ABILITY_BUTTONS_ENABLED", False)
    monkeypatch.setattr(config, "INGAME_MODAL_CHECK_EVERY", 1000)
    c = _ctrl()
    c.phase_started = time.monotonic()
    c._playing_modal_i = 0
    c._bush_logged = None
    c.last_move = c.last_attack = 0.0
    c._move_gap = 0.0
    c._attack_gap = float("inf")
    c._bush_action = lambda screen, gas_fr: action
    c._steer_to_bush = lambda *args: None
    c._wander_move = lambda gas_fr: None
    return c


@pytest.mark.parametrize(
    ("action", "kind"),
    [
        (None, "wander"),
        (BushAction("wander"), "wander"),
        (BushAction("hide", target=(1, 2)), "hide"),
        (BushAction("relocate", target=(1, 2)), "relocate"),
    ],
)
def test_a_move_tick_records_what_it_decided(monkeypatch, action, kind) -> None:
    c = _mover(monkeypatch, action)
    c.phase_playing(None, State.IN_MATCH)
    assert c._last_move_kind == kind
    assert c.last_move > 0.0


def test_a_tick_with_no_move_due_keeps_the_last_kind(monkeypatch) -> None:
    c = _mover(monkeypatch, BushAction("hide", target=(1, 2)))
    c._last_move_kind = "wander"
    c._move_gap = float("inf")
    c.phase_playing(None, State.IN_MATCH)
    assert c._last_move_kind == "wander"


def _swiper(monkeypatch, action, uniforms=()):
    """A _mover that runs the real move code down to a stubbed adb.swipe, so a test can hold
    what the tick recorded against what it swiped. ``uniforms`` scripts random.uniform in call
    order; past the script it returns its low bound."""
    c = _mover(monkeypatch, action)
    del c._steer_to_bush, c._wander_move
    c._gas_edges = set()
    c._relocations = 0
    c._bush_jitter_at = 0.0
    c._heading = 0.0
    c.swipes = []
    values = iter(uniforms)
    monkeypatch.setattr(controller_mod.random, "uniform", lambda low, high: next(values, low))
    monkeypatch.setattr(controller_mod.adb, "swipe", lambda *a: c.swipes.append(a))
    return c


def _swiped_heading(swipe) -> float:
    x1, y1, x2, y2, _duration = swipe
    return math.atan2(y2 - y1, x2 - x1)


def _bush_at(dx, dy):
    return (config.BUSH_SELF_POS[0] + dx, config.BUSH_SELF_POS[1] + dy)


def test_a_wander_records_its_heading_and_no_target(monkeypatch) -> None:
    c = _swiper(monkeypatch, None, uniforms=[0.5, 1.0, 1.0])
    c._heading = 1.0
    c._last_move_target = (9, 9)  # the tick before steered
    c.phase_playing(None, State.IN_MATCH)
    assert c._last_move_kind == "wander"
    assert c._last_move_heading == 1.5
    assert c._last_move_target is None
    assert len(c.swipes) == 1
    assert _swiped_heading(c.swipes[0]) == pytest.approx(1.5, abs=0.02)


def _bush_ticker(monkeypatch, bush_clusters):
    """A _swiper running the real _bush_action too, so a test sees the switch and the detector
    decide the logged kind, with a wander's random draws scripted."""
    c = _swiper(monkeypatch, None, uniforms=[0.5, 1.0, 1.0])
    del c._bush_action
    c._heading = 1.0
    monkeypatch.setattr(controller_mod.match_vision, "bush_clusters", bush_clusters)
    return c


def _assert_wandered(c) -> None:
    assert c._last_move_heading == 1.5
    assert c._last_move_target is None
    assert len(c.swipes) == 1
    assert _swiped_heading(c.swipes[0]) == pytest.approx(1.5, abs=0.02)


def test_a_move_tick_with_the_bush_switch_off_logs_wander_off(monkeypatch) -> None:
    def detector_must_not_run(screen):
        raise AssertionError("the switch is off")

    c = _bush_ticker(monkeypatch, detector_must_not_run)
    monkeypatch.setattr(config, "BUSH_HIDE", False)
    c.phase_playing(None, State.IN_MATCH)
    assert c._last_move_kind == "wander_off"
    assert c.dl.events == []  # a detector run would have logged bush_error
    _assert_wandered(c)


def test_a_move_tick_that_found_no_bush_logs_wander(monkeypatch) -> None:
    c = _bush_ticker(monkeypatch, lambda screen: [])
    c.phase_playing(None, State.IN_MATCH)
    assert c._last_move_kind == "wander"
    _assert_wandered(c)


def test_a_move_tick_whose_detector_failed_logs_wander(monkeypatch) -> None:
    def detector_fails(screen):
        raise RuntimeError("cv trouble")

    c = _bush_ticker(monkeypatch, detector_fails)
    c.phase_playing(None, State.IN_MATCH)
    assert c._last_move_kind == "wander"
    assert [etype for etype, _ in c.dl.events] == ["bush_error"]
    _assert_wandered(c)


@pytest.mark.parametrize("kind", ["hide", "relocate"])
def test_a_steer_records_the_heading_it_swiped_and_the_bush(monkeypatch, kind) -> None:
    target = _bush_at(-120, 90)
    c = _swiper(monkeypatch, BushAction(kind, target=target), uniforms=[1.0, 1.0])
    c.phase_playing(None, State.IN_MATCH)
    assert c._last_move_kind == kind
    assert c._last_move_heading == math.atan2(90, -120)
    assert c._last_move_target == target
    assert len(c.swipes) == 1
    assert _swiped_heading(c.swipes[0]) == pytest.approx(c._last_move_heading, abs=0.02)


def test_a_relocate_over_the_cap_is_a_wander_with_no_target(monkeypatch) -> None:
    action = BushAction("relocate", target=_bush_at(50, 50))
    c = _swiper(monkeypatch, action, uniforms=[0.5, 1.0, 1.0])
    c._relocations = config.BUSH_MAX_RELOCATIONS
    c._heading = 1.0
    c.phase_playing(None, State.IN_MATCH)
    assert c._last_move_kind == "relocate"
    assert c._last_move_heading == 1.5
    assert c._last_move_target is None


def test_a_hold_records_the_jitter_it_swiped_not_the_wander_heading(monkeypatch) -> None:
    target = _bush_at(5, 5)
    action = BushAction("hide", target=target, at_target=True)
    # the interval, the jitter's own heading, then its magnitude and duration
    c = _swiper(monkeypatch, action, uniforms=[0.0, 2.5, 1.0, 1.0])
    c._heading = 1.0
    c.phase_playing(None, State.IN_MATCH)
    assert c._last_move_heading == 2.5
    assert c._heading == 1.0
    assert c._last_move_target == target
    assert len(c.swipes) == 1


def test_a_hold_with_no_jitter_due_swiped_nothing(monkeypatch) -> None:
    """Without the clear, the row would show the last tick's heading for a swipe never sent."""
    target = _bush_at(5, 5)
    action = BushAction("hide", target=target, at_target=True)
    c = _swiper(monkeypatch, action, uniforms=[float("inf")])
    c._last_move_heading = 1.5
    c.phase_playing(None, State.IN_MATCH)
    assert c.swipes == []
    assert c._last_move_kind == "hide"
    assert c._last_move_heading is None
    assert c._last_move_target == target


def test_a_tick_with_no_move_due_keeps_the_last_heading_and_target(monkeypatch) -> None:
    c = _swiper(monkeypatch, None)
    c._last_move_heading = 1.5
    c._last_move_target = (1, 2)
    c._move_gap = float("inf")
    c.phase_playing(None, State.IN_MATCH)
    assert c.swipes == []
    assert c._last_move_heading == 1.5
    assert c._last_move_target == (1, 2)


def test_a_new_match_forgets_the_last_move() -> None:
    """The tick after entry reaches the session before the farm's first move of this match."""

    class _Gas:
        def reset(self):
            pass

    c = _ctrl()
    c.set_phase = lambda name: None
    c._gas = _Gas()
    c._last_move_kind = "hide"
    c._last_move_heading = 1.5
    c._last_move_target = (1, 2)
    c._enter_match()
    assert c._last_move_kind is None
    assert c._last_move_heading is None
    assert c._last_move_target is None


def test_a_rejoin_forgets_the_last_move(monkeypatch) -> None:
    """It puts last_move back to 0.0, so a kept kind would reach the row with a null age."""
    monkeypatch.setattr(config, "PLAY_SHADOW", True)
    monkeypatch.setattr(controller_mod.states, "is_other_device", lambda screen: False)
    monkeypatch.setattr(controller_mod.vision, "find", lambda *a, **k: None)
    monkeypatch.setattr(controller_mod.time, "sleep", lambda seconds: None)
    play = _Session()
    c = _ctrl(play)
    c.disconnect_count = 0
    c.event_shot = lambda screen, name: None
    c.tap = lambda *a, **k: None
    c.set_phase = lambda name: None
    c.last_move = 12.5
    c._last_move_kind = "hide"
    c._last_move_heading = 1.5
    c._last_move_target = (1, 2)
    c.handle_disconnect(None)
    c._play_observe(State.IN_MATCH)
    assert play.moves == [
        {"farm_move": None, "moved_at": 0.0, "farm_heading": None, "farm_target": None}
    ]


def test_a_raising_session_turns_shadow_off_after_one_line(monkeypatch) -> None:
    monkeypatch.setattr(config, "PLAY_SHADOW", True)
    session = _Session(boom=True)
    c = _ctrl(session)
    c._play_observe(State.IN_MATCH)  # must not reach the loop
    c._play_observe(State.IN_MATCH)
    assert c._play_off is True
    assert len(c.logs) == 1
    assert "shadow blew up" in c.logs[0]
    assert c.dl.events == []


def test_the_close_drains_the_summary_rows(monkeypatch) -> None:
    monkeypatch.setattr(config, "PLAY_SHADOW", True)
    rows = [("play_summary", {"frames": 12, "fps": 4.9})]
    session = _Session(close_rows=rows)
    c = _ctrl(session)
    c._play_close()
    assert session.closed == 1
    assert c.dl.events == rows


def test_the_close_lets_its_caller_swallow_the_failure() -> None:
    c = _ctrl(_Session(boom=True))
    with pytest.raises(RuntimeError):  # _play_shutdown owns the try/except
        c._play_close()


# --- the shutdown run()'s finally calls ---------------------------------------
#
# tests/test_preview.py drives the real loop on a controller built without __init__, so
# the shutdown may not assume `play` exists either: a finally that raises masks the
# exception it was handed.


def test_the_shutdown_survives_a_controller_without_the_attribute() -> None:
    c = Controller.__new__(Controller)  # no dl, no play, no _play_off
    c._play_shutdown()  # the AttributeError is the point: it must not escape
    assert not hasattr(c, "play")


def test_the_shutdown_swallows_a_failing_close() -> None:
    session = _Session(boom=True)
    c = _ctrl(session)
    c._play_shutdown()
    assert session.closed == 1
    assert c.dl.events == []


def test_the_shutdown_drains_the_summary_rows() -> None:
    rows = [("play_summary", {"frames": 12, "fps": 4.9}), ("play_fallback", {"reason": "stalled"})]
    session = _Session(close_rows=rows)
    c = _ctrl(session)
    c._play_shutdown()
    assert session.closed == 1
    assert c.dl.events == rows


def _boom_factory():
    """A PlaySession that cannot even be built (a broken play extra looks like this)."""
    raise RuntimeError("no session today")


def test_the_tick_survives_a_controller_without_the_attributes(monkeypatch) -> None:
    """Shadow on, and the controller has neither `_play_off` nor `play` (the loop tests
    build it that way). The failure latches on the instance it was handed."""
    monkeypatch.setattr(config, "PLAY_SHADOW", True)
    monkeypatch.setattr("brawlfarm.play.session.PlaySession", _boom_factory)
    c = Controller.__new__(Controller)
    c.dl = _DL()
    c.logs = []
    c.log = c.logs.append
    c.phase = "playing"
    c._play_observe(State.IN_MATCH)
    assert c._play_off is True
    assert len(c.logs) == 1 and "no session today" in c.logs[0]
    assert c.dl.events == []


# --- the import the shadow flag pays for --------------------------------------
#
# In-process this would pass or fail for the wrong reason: tests/test_play_session.py
# imports the module anyway. A fresh interpreter is the only honest answer, like
# tests/test_play_detect.py's "pulls nothing heavy".

_IMPORT_PROBE = (
    "import os, sys; "
    "os.environ['BRAWL_PLAY_SHADOW'] = {flag!r}; "
    "from brawlfarm.core.controller import Controller; "
    "from brawlfarm.core.states import State; "
    "c = Controller.__new__(Controller); "
    "c.dl = type('D', (), {{'event': lambda self, k, **f: None}})(); "
    "c.log = lambda msg: None; "
    "c.phase = {phase!r}; "
    "c._play_observe(State.{state}); "
    "print('brawlfarm.play.session' in sys.modules)"
)


def _probe(flag: str, phase: str, state: str) -> str:
    code = _IMPORT_PROBE.format(flag=flag, phase=phase, state=state)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return done.stdout.strip()


def test_shadow_off_never_imports_the_session_module() -> None:
    # A whole match of ticks costs the worker nothing: the import sits below the guard.
    assert _probe("0", "playing", "IN_MATCH") == "False"


def test_shadow_on_imports_the_session_module() -> None:
    # The mirror, at the menu: the session is built and ticked for real, but a tick that
    # is not a match starts no stream, so this needs neither adb nor the play extra.
    assert _probe("1", "menu", "MENU") == "True"


# --- The spectate exit: the whole team is out, so tap Exit and stop playing -----------------

_SCREEN = object()  # a frame stand-in: these tests stub the one reader that looks at it


class _Gas:
    def reset(self):
        pass


def _spectator(monkeypatch, reads):
    """A _mover whose in-match ticks read ``reads`` (one bool per tick) from the spectate
    check, with every tick a move and attack tick, and taps and moves recorded, not sent."""
    c = _mover(monkeypatch, None)
    c.taps = []
    c.moves = 0

    def wander(gas_fr):
        c.moves += 1

    c._wander_move = wander
    c.tap = lambda point, kind=None: c.taps.append((point, kind))
    c.set_phase = lambda name: None
    c._gas = _Gas()
    c._reset_spectate()
    values = iter(reads)
    monkeypatch.setattr(reader, "spectating", lambda frame: next(values))
    return c


def _ticks(c, n):
    for _ in range(n):
        c.last_move = c.last_attack = 0.0
        c._attack_gap = 0.0
        c.phase_playing(_SCREEN, State.IN_MATCH)


def _exits(c):
    return [t for t in c.taps if t == (config.SPECTATE_EXIT_BUTTON, "leave_spectate")]


def _left(c):
    return [fields for kind, fields in c.dl.events if kind == "leave_spectate"]


def test_two_spectating_ticks_tap_exit_once(monkeypatch) -> None:
    c = _spectator(monkeypatch, [True, True])
    _ticks(c, 2)
    assert c.taps == [(config.SPECTATE_EXIT_BUTTON, "leave_spectate")]
    assert _left(c) == [{"tap": 1}]
    assert c.moves == 0  # no blind moves or attacks on the spectate screen


def test_one_spectating_tick_taps_nothing(monkeypatch) -> None:
    c = _spectator(monkeypatch, [True, False, True])
    _ticks(c, 3)
    assert _exits(c) == []
    assert _left(c) == []


def test_alive_ticks_play_as_before(monkeypatch) -> None:
    c = _spectator(monkeypatch, [False, False])
    _ticks(c, 2)
    assert c.moves == 2
    assert _exits(c) == []
    assert len(c.taps) == 4  # attack and super, each tick


def test_a_fresh_match_resets_the_debounce(monkeypatch) -> None:
    c = _spectator(monkeypatch, [True, True])
    _ticks(c, 1)
    c._enter_match()
    _ticks(c, 1)
    assert _exits(c) == []


def test_a_fresh_match_resets_the_taps(monkeypatch) -> None:
    c = _spectator(monkeypatch, [True] * 30 + [True, True])
    _ticks(c, 30)  # three taps, then given up for that match
    c._enter_match()
    _ticks(c, 2)
    assert [f["tap"] for f in _left(c)] == [1, 2, 3, 1]


def test_a_tap_that_takes_is_not_repeated(monkeypatch) -> None:
    c = _spectator(monkeypatch, [True, True] + [False] * 5)
    _ticks(c, 7)
    assert len(_exits(c)) == 1


def test_a_tap_that_does_not_take_retries_twice_then_gives_up(monkeypatch) -> None:
    c = _spectator(monkeypatch, [True] * 30)
    _ticks(c, 30)
    assert len(_exits(c)) == config.SPECTATE_MAX_TAPS == 3
    assert [f["tap"] for f in _left(c)] == [1, 2, 3]
    assert c._spectate_off is True
    assert c.moves > 0  # gave up: the old behaviour, playing the match out


def test_a_failing_reader_logs_once_and_plays_on(monkeypatch) -> None:
    c = _spectator(monkeypatch, [])

    def boom(frame):
        raise ValueError("bad frame")

    monkeypatch.setattr(reader, "spectating", boom)
    _ticks(c, 3)
    assert [kind for kind, _ in c.dl.events].count("leave_spectate_error") == 1
    assert c.moves == 3
    assert _exits(c) == []


def test_the_spectate_exit_ignores_the_play_switches(monkeypatch) -> None:
    monkeypatch.setattr(config, "PLAY_SHADOW", False)
    c = _spectator(monkeypatch, [True, True])
    monkeypatch.setattr(config, "BUSH_HIDE", False)
    _ticks(c, 2)
    assert len(_exits(c)) == 1


def test_the_spectate_exit_reads_a_real_frame() -> None:
    """The real reader on a synthetic frame: the Exit label alone, two ticks, one tap."""
    c = _ctrl()
    c.taps = []
    c.tap = lambda point, kind=None: c.taps.append((point, kind))
    c._reset_spectate()
    frame = np.zeros((reader.FRAME_H, reader.FRAME_W, 3), dtype=np.uint8)
    x0, y0, x1, y1 = reader.EXIT_LABEL_BOX
    label = reader.EXIT_LABEL.astype(np.uint8)
    big = cv2.resize(label, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    frame[y0:y1, x0:x1][big > 0] = 255
    assert c._leave_spectate(frame) is True
    assert c.taps == []
    assert c._leave_spectate(frame) is True
    assert _exits(c) == [(config.SPECTATE_EXIT_BUTTON, "leave_spectate")]
    assert c._leave_spectate(np.zeros_like(frame)) is True  # the tap is settling
