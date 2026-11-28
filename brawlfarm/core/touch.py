"""Raw touch frames for the fast input path, for BlueStacks and MuMu.

BlueStacks: "BlueStacks Virtual Touch", a type A multitouch node in the screen's own
landscape frame, range 0..32767 on both axes. Its frames come from the config constants
and are byte-identical to what the farm has always sent.

MuMu Player 12 (measured 2026-11-28 on MuMu 6.0.2, Android 15): "Xiaomi Touchscreen", a
type B node (slots and tracking ids) in the display's natural PORTRAIT frame, X 0..900 and
Y 0..1600, while the game runs at rotation 1. A landscape point (x, y) lands at node
(900 - y, x); a type A frame on that node registers nothing. Writable by the shell user,
no root. The node, its ranges and the protocol are read from `getevent -p` instead of
being written down here, so another MuMu build or another node number still works.

Everything here is pure: parsing text and building bytes. core/adb.py runs the commands.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

EV_SYN, EV_KEY, EV_ABS = 0, 1, 3
SYN_REPORT, SYN_MT_REPORT = 0, 2
ABS_MT_SLOT = 0x2F
ABS_MT_POSITION_X, ABS_MT_POSITION_Y = 0x35, 0x36
ABS_MT_TRACKING_ID = 0x39
BTN_TOUCH = 0x14A
TRACKING_ID_MAX = 65535

_ADD_RE = re.compile(r"^add device \d+:\s*(\S+)")
_NAME_RE = re.compile(r'^\s*name:\s*"(.*)"')
_TYPE_RE = re.compile(r"^\s*([A-Z]+)\s*\(([0-9a-fA-F]{4})\):(.*)$")
_ABS_RE = re.compile(r"([0-9a-fA-F]{4})\s*:\s*value\s+-?\d+,\s*min\s+(-?\d+),\s*max\s+(-?\d+)")
_HEX_RE = re.compile(r"\b[0-9a-fA-F]{4}\b")
_PROP_RE = re.compile(r"\bINPUT_PROP_[A-Z_]+\b")


def ev(t: int, c: int, v: int) -> bytes:
    """One 64-bit struct input_event: time (8 + 8), type (H), code (H), value (i), LE."""
    return struct.pack("<qqHHi", 0, 0, t, c, v)


@dataclass
class InputNode:
    """One /dev/input node as `getevent -p` describes it."""

    path: str
    name: str = ""
    keys: set[int] = field(default_factory=set)
    abs: dict[int, tuple[int, int]] = field(default_factory=dict)  # code -> (min, max)
    props: set[str] = field(default_factory=set)

    @property
    def is_multitouch(self) -> bool:
        return ABS_MT_POSITION_X in self.abs and ABS_MT_POSITION_Y in self.abs


def parse_getevent(text: str) -> list[InputNode]:
    """Parse `getevent -p` (numeric codes, no -l). Unknown lines are skipped."""
    nodes: list[InputNode] = []
    cur: InputNode | None = None
    section = ""  # "KEY", "ABS", ... or "props"
    for line in (text or "").splitlines():
        m = _ADD_RE.match(line)
        if m:
            cur = InputNode(m.group(1))
            nodes.append(cur)
            section = ""
            continue
        if cur is None:
            continue
        m = _NAME_RE.match(line)
        if m:
            cur.name = m.group(1)
            continue
        if line.strip() == "input props:":
            section = "props"
            continue
        m = _TYPE_RE.match(line)
        if m and section != "props":
            section, rest = m.group(1), m.group(3)
        elif section == "props":
            cur.props.update(_PROP_RE.findall(line))
            continue
        elif line.strip() == "events:":
            continue
        else:
            rest = line
        if section == "ABS":
            for code, lo, hi in _ABS_RE.findall(rest):
                cur.abs[int(code, 16)] = (int(lo), int(hi))
        elif section == "KEY":
            cur.keys.update(int(h, 16) for h in _HEX_RE.findall(rest))
    return nodes


def pick_touch_node(nodes: list[InputNode]) -> InputNode | None:
    """The touchscreen: a multitouch node, a direct one (INPUT_PROP_DIRECT) first, then one
    named like a touchscreen. None when no node reports multitouch positions."""
    mt = [n for n in nodes if n.is_multitouch]
    if not mt:
        return None
    for test in (
        lambda n: "INPUT_PROP_DIRECT" in n.props,
        lambda n: "touch" in n.name.lower(),
    ):
        hit = [n for n in mt if test(n)]
        if hit:
            return hit[0]
    return mt[0]


_ROTATION_PATTERNS = (
    re.compile(r"Viewport[^\n]*\borientation=(\d)\b"),
    re.compile(r"SurfaceOrientation:\s*(\d)\b"),
    re.compile(r"mCurrentRotation=(?:ROTATION_)?(\d+)\b"),
)


def parse_rotation(text: str) -> int | None:
    """The display rotation, 0 to 3, from `dumpsys input` (or `dumpsys window`) text.
    A degrees value (90, 180, 270) is turned into its quarter turn. None when absent."""
    for pat in _ROTATION_PATTERNS:
        m = pat.search(text or "")
        if m:
            value = int(m.group(1))
            if value in (90, 180, 270):
                value //= 90
            if 0 <= value <= 3:
                return value
    return None


def default_rotation(node: InputNode, screen_w: int, screen_h: int) -> int:
    """No rotation read: a node whose frame has the other aspect from the screen is taken
    to be rotation 1, the one measured on MuMu; otherwise 0."""
    x_span = node.abs[ABS_MT_POSITION_X][1] - node.abs[ABS_MT_POSITION_X][0]
    y_span = node.abs[ABS_MT_POSITION_Y][1] - node.abs[ABS_MT_POSITION_Y][0]
    node_portrait = y_span > x_span
    screen_portrait = screen_h > screen_w
    return 1 if node_portrait != screen_portrait else 0


@dataclass
class TouchProfile:
    """How to write a touch to one device: where, in what range, in which protocol, and
    how the screen is turned against the node's frame."""

    device: str
    x_range: tuple[int, int]
    y_range: tuple[int, int]
    protocol: str  # "a" (no slots) or "b" (slots and tracking ids)
    screen_w: int
    screen_h: int
    rotation: int = 0  # quarter turns of the display against the node's natural frame
    btn_touch: bool = False  # the node declares BTN_TOUCH, so send it with type B
    _next_id: int = 0

    @classmethod
    def from_node(
        cls, node: InputNode, screen_w: int, screen_h: int, rotation: int | None
    ) -> TouchProfile:
        if rotation is None:
            rotation = default_rotation(node, screen_w, screen_h)
        return cls(
            device=node.path,
            x_range=node.abs[ABS_MT_POSITION_X],
            y_range=node.abs[ABS_MT_POSITION_Y],
            protocol="b" if ABS_MT_SLOT in node.abs else "a",
            screen_w=screen_w,
            screen_h=screen_h,
            rotation=rotation,
            btn_touch=BTN_TOUCH in node.keys,
        )

    def to_node(self, x: int, y: int) -> tuple[int, int]:
        """A screen pixel (the landscape frame the farm works in) as node coordinates."""
        w, h, r = self.screen_w, self.screen_h, self.rotation % 4
        if r == 0:
            nx, ny, nw, nh = x, y, w, h
        elif r == 1:
            nx, ny, nw, nh = h - y, x, h, w
        elif r == 2:
            nx, ny, nw, nh = w - x, h - y, w, h
        else:
            nx, ny, nw, nh = y, w - x, h, w
        (x0, x1), (y0, y1) = self.x_range, self.y_range
        dx = x0 + round(nx / nw * (x1 - x0))
        dy = y0 + round(ny / nh * (y1 - y0))
        return min(max(dx, x0), x1), min(max(dy, y0), y1)

    def _pos(self, x: int, y: int) -> bytes:
        dx, dy = self.to_node(int(x), int(y))
        return ev(EV_ABS, ABS_MT_POSITION_X, dx) + ev(EV_ABS, ABS_MT_POSITION_Y, dy)

    def down(self, x: int, y: int) -> bytes:
        """Finger down at (x, y)."""
        if self.protocol == "a":
            return self._pos(x, y) + ev(EV_SYN, SYN_MT_REPORT, 0) + ev(EV_SYN, SYN_REPORT, 0)
        self._next_id = self._next_id % TRACKING_ID_MAX + 1
        out = ev(EV_ABS, ABS_MT_SLOT, 0) + ev(EV_ABS, ABS_MT_TRACKING_ID, self._next_id)
        if self.btn_touch:
            out += ev(EV_KEY, BTN_TOUCH, 1)
        return out + self._pos(x, y) + ev(EV_SYN, SYN_REPORT, 0)

    def move(self, x: int, y: int) -> bytes:
        """The held finger moves to (x, y). Type A has no held state: a move is a frame."""
        if self.protocol == "a":
            return self.down(x, y)
        return ev(EV_ABS, ABS_MT_SLOT, 0) + self._pos(x, y) + ev(EV_SYN, SYN_REPORT, 0)

    def up(self) -> bytes:
        """Finger up."""
        if self.protocol == "a":
            return ev(EV_SYN, SYN_MT_REPORT, 0) + ev(EV_SYN, SYN_REPORT, 0)
        out = ev(EV_ABS, ABS_MT_SLOT, 0) + ev(EV_ABS, ABS_MT_TRACKING_ID, -1)
        if self.btn_touch:
            out += ev(EV_KEY, BTN_TOUCH, 0)
        return out + ev(EV_SYN, SYN_REPORT, 0)


def bluestacks_profile(
    device: str, max_x: int, max_y: int, screen_w: int, screen_h: int
) -> TouchProfile:
    """The fixed BlueStacks Virtual Touch profile from the config constants."""
    return TouchProfile(device, (0, max_x), (0, max_y), "a", screen_w, screen_h)


def emulator_of(nodes: list[InputNode]) -> str:
    """Which emulator the input nodes belong to: "bluestacks" when a node is named for
    BlueStacks, else "other". MuMu names its nodes for Xiaomi, so this is how a worker
    set up for BlueStacks on 5555 notices that MuMu (which also listens on 5555) answered."""
    if any("bluestacks" in n.name.lower() for n in nodes):
        return "bluestacks"
    return "other"
