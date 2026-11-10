"""The FastAPI application factory.

One app per process. create_app(sup, home) stores the live supervisor and the data
directory on app.state, refuses non-loopback clients and foreign Host headers, runs one supervisor tick during
startup so the first request already has instance views to show, and serves the built web
UI from brawlfarm/web/dist when phase 4 has produced one.

Routers are included before the static mount, so /api/* always wins over the single-page
app's catch-all.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

import brawlfarm
from brawlfarm import __version__, play
from brawlfarm.api import (
    alerts,
    brawlers,
    connection,
    events,
    feed,
    instances,
    notify_routes,
    plans,
    roster,
    schedule,
    screens,
    settings_routes,
    setup_routes,
    stats,
)
from brawlfarm.api import calibration as calibration_routes
from brawlfarm.api.alerts import AlertStore
from brawlfarm.api.events import BusLogHandler, EventBus
from brawlfarm.api.feed import FeedTailer
from brawlfarm.api.instances import view_to_dict
from brawlfarm.supervisor import Supervisor

log = logging.getLogger("brawlfarm.api")

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
# The names a Host header may carry. TrustedHostMiddleware compares the part before the
# port, so 127.0.0.1:8765, a --port run and the Vite dev server on localhost:5173 all
# pass. A page that rebinds its own DNS name to 127.0.0.1 still sends that name, so the
# loopback peer check alone cannot stop it and this list does.
ALLOWED_HOSTS = ("127.0.0.1", "localhost")

PLACEHOLDER_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>brawlfarm</title>
  </head>
  <body>
    <p>brawlfarm is running. The panel arrives in phase 4. API docs at /docs.</p>
  </body>
</html>
"""


def dist_dir() -> Path:
    """Where the built web UI lands (phase 4 builds it, CI packages it into the wheel).
    A function rather than a constant so tests can point it somewhere else."""
    return Path(brawlfarm.__file__).resolve().parent / "web" / "dist"


class SpaFiles(StaticFiles):
    """StaticFiles with a single-page-app fallback: an unknown path serves index.html so
    the browser can deep-link to /instances/alpha. Starlette's html=True only covers
    directory indexes. /api/* is never rewritten -- an unknown API path stays a 404."""

    async def get_response(self, path: str, scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            # StaticFiles.get_path hands back OS separators, so normalise before the
            # prefix test: on Windows an unknown API path arrives as "api\nope".
            if exc.status_code != 404 or path.replace("\\", "/").startswith("api/"):
                raise
            return await super().get_response("index.html", scope)


def create_app(sup: Supervisor, home: Path) -> FastAPI:
    """Build the panel's app around a live supervisor. The caller serves it with uvicorn
    on 127.0.0.1 (brawlfarm/__main__.py) or drives it with starlette's TestClient."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Startup: attach the bus to this loop, mirror the brawlfarm logger onto it,
        publish every supervisor state change, collect alerts from the tailer and from the
        supervisor itself, start the feed tailer, run one tick so the first
        GET /api/instances already has views, and start the icon prewarm without
        awaiting it. Shutdown: stop the tailer and detach the log
        handler so a second app in the same process (the test suite makes many) does not
        publish into a dead bus."""
        bus = EventBus()
        bus.attach(asyncio.get_running_loop())
        app.state.bus = bus
        store = AlertStore(bus)
        app.state.alerts = store
        # tick() runs in a worker thread, so both callbacks fire OFF the loop thread;
        # EventBus.publish hops back with call_soon_threadsafe.
        app.state.sup.subscribe(lambda view: bus.publish("instance", view_to_dict(view)))
        app.state.sup.subscribe_alerts(store.add)
        handler = BusLogHandler(bus)
        logging.getLogger("brawlfarm").addHandler(handler)
        tailer = FeedTailer(app.state.home, app.state.sup, bus, alerts=store)
        app.state.tailer = tailer
        tailing = asyncio.create_task(tailer.run())
        try:
            # One tick before the first request, so GET /api/instances is never empty on a
            # cold start. It runs in a thread because a tick shells out to adb and writes
            # files.
            try:
                await asyncio.to_thread(app.state.sup.tick)
            except Exception:  # a failed startup tick must not stop the app from serving
                log.exception("startup tick failed")
            # One pass over the cached rosters, started and not awaited: start-up must
            # never block on the CDN, and a cold cache simply has nothing to warm.
            asyncio.create_task(
                asyncio.to_thread(
                    brawlers.prewarm,
                    app.state.home,
                    app.state.sup.settings.connection.brawl_api_token,
                    app.state.roster.cached_names(),
                )
            )
            yield
        finally:
            tailing.cancel()
            with suppress(asyncio.CancelledError):
                await tailing
            logging.getLogger("brawlfarm").removeHandler(handler)

    app = FastAPI(title="brawlfarm", version=__version__, lifespan=lifespan)
    app.state.sup = sup
    app.state.home = Path(home).resolve()
    app.state.started_at = time.monotonic()
    # One asyncio.Lock per instance, filled lazily by the screenshot route: two
    # concurrent screencaps against one BlueStacks window fight each other.
    app.state.screenshot_locks = {}
    # The preview route's fallback frames, one per instance: an instance with no worker
    # writing preview.jpg is captured live at most once every LIVE_MIN_INTERVAL_S.
    app.state.preview_cache = {}
    # One roster cache for the process: per instance, five-minute TTL, stale on failure.
    # Not built in the lifespan because it holds nothing loop-bound until its first use.
    app.state.roster = roster.RosterCache()
    # One connection check for the process: five-minute TTL, one lock. Built here for the
    # same reason the roster cache is, and its lock is made on first use.
    app.state.connection = connection.ConnectionCache()

    @app.middleware("http")
    async def _loopback_only(request: Request, call_next):
        """Spec section 3: the panel binds 127.0.0.1 and has no authentication, so a
        request from any other host is refused before it reaches a route. request.client
        is None for an ASGI transport that does not report a peer; that is us."""
        client = request.client
        if client is not None and client.host not in LOOPBACK_HOSTS:
            return JSONResponse({"detail": "loopback only"}, status_code=403)
        return await call_next(request)

    # Added after the loopback guard, so it runs first: a request with any other Host
    # header gets the middleware's 400 "Invalid host header" before a route sees it.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(ALLOWED_HOSTS))

    @app.get("/api/health")
    async def health(request: Request) -> dict:
        """The shell's probe: which version is running, where its data lives, how many
        instances it manages, how long this process has been up and whether the play
        extra is installed."""
        state = request.app.state
        return {
            "version": __version__,
            "home": str(state.home),
            "instances": len(state.sup.settings.instances),
            "uptime_s": round(time.monotonic() - state.started_at, 3),
            "play_available": play.available(),
        }

    # --- routers ---------------------------------------------------------------------
    # Included before the static mount below, so /api/* always wins over the SPA.
    app.include_router(events.router)
    app.include_router(instances.router)
    app.include_router(settings_routes.router)
    app.include_router(notify_routes.router)
    app.include_router(setup_routes.router)
    app.include_router(screens.router)
    app.include_router(plans.router)
    app.include_router(connection.router)
    app.include_router(schedule.router)
    app.include_router(feed.router)
    app.include_router(alerts.router)
    app.include_router(stats.router)
    app.include_router(brawlers.router)
    app.include_router(calibration_routes.router)

    # --- the web UI ------------------------------------------------------------------
    dist = dist_dir()
    if (dist / "index.html").exists():
        app.mount("/", SpaFiles(directory=dist, html=True), name="web")
    else:

        @app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            """Placeholder until phase 4 builds the UI into brawlfarm/web/dist."""
            return HTMLResponse(PLACEHOLDER_HTML)

    return app
