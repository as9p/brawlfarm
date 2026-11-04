"""Promo popup close-X tap point.

vision.Match.x and Match.y are already the centre of the matched template (find()
builds them as top_left + w // 2). handle_popup used to add half the template size
again, so the close-X tap landed half a template right and down of the live X.

All offline: Controller.__new__ + minimal attrs (the test_network_stuck.py pattern).
"""

from __future__ import annotations

import time

from brawlfarm.core import config
from brawlfarm.core import controller as controller_mod
from brawlfarm.core.controller import Controller
from brawlfarm.core.vision import Match


class _DL:
    def __init__(self):
        self.events = []

    def event(self, etype, **fields):
        self.events.append((etype, fields))


def _ctrl(monkeypatch):
    c = Controller.__new__(Controller)  # skip __init__ (ApiClient/DataLog/adb)
    c.dl = _DL()
    c.phase = "lobby"
    c.phase_started = time.monotonic()
    c.popup_count = 0
    c.taps = []
    monkeypatch.setattr(Controller, "log", lambda self, msg: None)
    monkeypatch.setattr(controller_mod.adb, "tap", lambda x, y: c.taps.append((x, y)))
    monkeypatch.setattr(controller_mod.time, "sleep", lambda s: None)
    return c


def test_close_x_tap_goes_to_match_centre(monkeypatch):
    c = _ctrl(monkeypatch)
    m = Match(name="close_x", confidence=0.95, x=1380, y=120, w=60, h=40)
    monkeypatch.setattr(controller_mod.vision, "find", lambda screen, name: m)

    c.handle_popup(screen=None)

    assert c.taps == [m.center]
    assert ("tap", {"button": "close_x", "x": 1380, "y": 120, "phase": "lobby"}) in c.dl.events


def test_close_x_falls_back_to_configured_point_without_a_match(monkeypatch):
    c = _ctrl(monkeypatch)
    monkeypatch.setattr(controller_mod.vision, "find", lambda screen, name: None)

    c.handle_popup(screen=None)

    assert c.taps == [tuple(config.CLOSE_X_BUTTON)]
