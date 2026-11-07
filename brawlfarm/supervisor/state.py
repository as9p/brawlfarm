"""Pure classification: heartbeat health from status.json, and the one instance state the
control panel shows everywhere (spec section 6). No I/O here; the loop feeds it facts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

STALE_S = 240.0  # heartbeat older than this while the PID is alive = stale
RECYCLE_FAILED_REASON = "emulator_recycle_failed"  # stays down until start(), never resumes
_TS_FMT = "%Y-%m-%dT%H:%M:%S"


class Health(StrEnum):
    HEALTHY = "healthy"
    STALE = "stale"
    DEAD = "dead"


class InstanceState(StrEnum):
    FARMING = "farming"
    STOPPED = "stopped"
    SCHEDULED_BREAK = "scheduled_break"
    RECONNECTING = "reconnecting"
    OFFLINE = "offline"
    STARTING = "starting"
    STOPPING = "stopping"


@dataclass(frozen=True)
class InstanceView:
    """Everything the panel needs about one instance, derived once per tick."""

    name: str
    adb_port: int
    state: InstanceState
    health: Health
    pid: int | None
    heartbeat_age_s: float | None
    phase: str | None
    desired: str  # "run" | "stop" | "observe"
    desired_reason: str | None
    until: datetime | None  # session end, resume time, retry time or stop deadline
    games_played: int | None
    farm_brawler: str | None
    note: str  # one plain sentence for the card, "" when nothing to say


def parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.strptime(ts, _TS_FMT)
    except (TypeError, ValueError):
        return None


def heartbeat_age_s(status: dict | None, now: datetime) -> float | None:
    if not status:
        return None
    stamped = parse_ts(str(status.get("ts", "")))
    if stamped is None:
        return None
    return (now - stamped).total_seconds()


def classify(status: dict | None, alive: bool, now: datetime) -> Health:
    """DEAD when the PID is gone; STALE when alive but the heartbeat is old or missing."""
    if not alive:
        return Health.DEAD
    age = heartbeat_age_s(status, now)
    if age is None or age >= STALE_S:
        return Health.STALE
    return Health.HEALTHY


def derive_state(
    *,
    health: Health,
    desired: str,
    desired_reason: str | None,
    desired_until: datetime | None,
    stop_pending: datetime | None | bool,
    booting: bool,
    offline_until: datetime | None,
    status: dict | None,
) -> tuple[InstanceState, datetime | None]:
    """Fold the tick's facts into one state plus the time the panel shows with it.

    ``stop_pending`` is the kill deadline when a graceful stop is in flight (or a bool
    for callers without one). Precedence, alive: stopping > reconnecting > farming.
    Dead: offline > starting > scheduled break / stopped.

    ``observe`` is a desired mode, not a state: an observing instance reads as STARTING
    then FARMING, which keeps it inside the panel's live-instance guards. The recorder
    payload's ``mode`` field is what says which kind of worker is running.
    """
    alive = health in (Health.HEALTHY, Health.STALE)
    if alive:
        if stop_pending:
            deadline = stop_pending if isinstance(stop_pending, datetime) else None
            return InstanceState.STOPPING, deadline
        s = status or {}
        if int(s.get("recovery_attempts") or 0) > 0 or int(s.get("disconnect_count") or 0) > 0:
            return InstanceState.RECONNECTING, None
        return InstanceState.FARMING, desired_until
    if offline_until is not None:
        return InstanceState.OFFLINE, offline_until
    if booting or desired in ("run", "observe"):
        return InstanceState.STARTING, None
    if desired_reason in ("override_stop", RECYCLE_FAILED_REASON):
        return InstanceState.STOPPED, None
    return InstanceState.SCHEDULED_BREAK, desired_until
