"""Alerts for the Fleet drawer (spec section 7).

The worker already writes alert-worthy events into its session JSONL and the push notifier
already turns them into a phone buzz (core/notify.py). This store collects the same kinds
for the panel: the feed tailer offers it every session line, and the supervisor hands it
the `offline` alert it raises itself. In memory only (ruling 3) -- alerts are a "look at
this now" list, not a record; the session files keep the history.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, Response

from brawlfarm.core import notify

router = APIRouter()

MAX_ALERTS = 200
# The panel's alert kinds are the spec's six (section 7). notify.ALERT_KINDS also carries
# "stop" for push notifications; a graceful stop is not a panel alert.
PANEL_ALERT_KINDS = frozenset(notify.ALERT_KINDS) - {"stop"}


@dataclass
class Alert:
    """One thing worth interrupting the owner for. Mutable so dismiss() can flip a flag
    without rebuilding the list."""

    id: int
    ts: str
    instance: str
    kind: str
    title: str
    detail: str
    dismissed: bool = False


class AlertStore:
    """Two threads write here: the feed tailer on the event loop thread and the supervisor's
    offline hook on the tick's worker thread. So every method holds `_lock` -- `_next_id += 1`
    is a read-modify-write, and iterating the deque while the other thread appends raises
    "deque mutated during iteration". Only bus.publish is left outside; it is thread-safe by
    design and must not run under a lock a listener could contend with."""

    def __init__(self, bus=None, *, maxlen: int = MAX_ALERTS) -> None:
        self._bus = bus
        self._alerts: deque[Alert] = deque(maxlen=maxlen)
        self._next_id = 1
        self._lock = threading.Lock()

    def add(self, instance: str, kind: str, fields: dict, ts: str | None = None) -> Alert:
        """Record an alert and publish it on the bus. The detail line is built the same way
        notify.maybe_alert builds its message, so the drawer and the phone agree."""
        with self._lock:
            alert = Alert(
                id=self._next_id,
                ts=ts or datetime.now().isoformat(timespec="seconds"),
                instance=instance,
                kind=kind,
                title=notify.alert_title(kind, fields),
                detail=", ".join(f"{k}={v}" for k, v in fields.items() if v is not None),
            )
            self._next_id += 1
            self._alerts.append(alert)
        if self._bus is not None:
            self._bus.publish("alert", asdict(alert))
        return alert

    def ingest(self, instance: str, record: dict) -> Alert | None:
        """A raw session line; None unless its kind is one of PANEL_ALERT_KINDS."""
        kind = str(record.get("kind") or "")
        if kind not in PANEL_ALERT_KINDS:
            return None
        stamped = record.get("ts")
        fields = {k: v for k, v in record.items() if k not in ("ts", "kind")}
        return self.add(instance, kind, fields, ts=str(stamped) if stamped else None)

    def list(self, *, include_dismissed: bool = False) -> list[Alert]:
        """Newest first, which is the order the drawer shows them in. The deque is copied
        under the lock and filtered outside it, so a writer never waits on a reader."""
        with self._lock:
            snapshot = list(self._alerts)
        return [a for a in reversed(snapshot) if include_dismissed or not a.dismissed]

    def dismiss(self, alert_id: int) -> bool:
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.dismissed = True
                    return True
        return False

    def dismiss_all(self) -> int:
        """Mark everything read; returns how many were still unread. The drawer's
        "Dismiss all" is one call rather than one per row, so a screen full of alerts
        does not become a burst of writes against the same lock."""
        dismissed = 0
        with self._lock:
            for alert in self._alerts:
                if not alert.dismissed:
                    alert.dismissed = True
                    dismissed += 1
        return dismissed

    def unread_count(self) -> int:
        with self._lock:
            return sum(1 for alert in self._alerts if not alert.dismissed)


@router.get("/api/alerts")
async def get_alerts(request: Request, include_dismissed: bool = False) -> dict:
    """The alert list for the Fleet drawer, newest first, with the count for its badge."""
    store: AlertStore = request.app.state.alerts
    return {
        "alerts": [asdict(a) for a in store.list(include_dismissed=include_dismissed)],
        "unread": store.unread_count(),
    }


@router.post("/api/alerts/dismiss-all", status_code=204)
async def dismiss_all_alerts(request: Request) -> Response:
    """Clear the drawer in one call. Always 204, even when nothing was unread: the button
    is idempotent, and a second click must not paint an error."""
    store: AlertStore = request.app.state.alerts
    store.dismiss_all()
    return Response(status_code=204)


@router.post("/api/alerts/{alert_id}/dismiss", status_code=204)
async def dismiss_alert(request: Request, alert_id: int) -> Response:
    """Mark one alert read. 404 when it has already fallen off the end of the store."""
    store: AlertStore = request.app.state.alerts
    if not store.dismiss(alert_id):
        raise HTTPException(status_code=404, detail="unknown alert")
    return Response(status_code=204)
