"""The app factory: loopback-only access, the Host header guard, the health probe, the startup tick that fills
the instance views before the first request, and the index (placeholder now, the built
single-page app once brawlfarm/web/dist exists)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brawlfarm.api import app as app_module
from tests.apihelpers import LOOPBACK, PANEL_URL, make_client


@pytest.fixture()
def api(tmp_path: Path):
    client, sup, home = make_client(tmp_path, ("alpha", "bravo"))
    try:
        yield client, sup, home
    finally:
        client.__exit__(None, None, None)


def test_health_reports_version_home_and_instances(api) -> None:
    client, _sup, home = api
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["version"].startswith("1.")
    assert body["home"] == str(home.resolve())
    assert body["instances"] == 2
    assert body["uptime_s"] >= 0.0


def test_health_says_whether_the_play_extra_is_installed(api, monkeypatch) -> None:
    from brawlfarm import play

    client, _sup, _home = api
    monkeypatch.setattr(play, "available", lambda: False)
    assert client.get("/api/health").json()["play_available"] is False
    monkeypatch.setattr(play, "available", lambda: True)
    assert client.get("/api/health").json()["play_available"] is True


def test_the_lifespan_runs_one_tick_so_views_exist(api) -> None:
    _client, sup, _home = api
    assert [v.name for v in sup.views()] == ["alpha", "bravo"]


def test_a_non_loopback_client_is_refused(api) -> None:
    client, _sup, _home = api
    stranger = TestClient(client.app, base_url=PANEL_URL, client=("10.0.0.5", 1))
    r = stranger.get("/api/health")
    assert r.status_code == 403
    assert r.json() == {"detail": "loopback only"}


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8765", "192.168.1.20:8765"])
def test_a_foreign_host_header_is_refused(api, host) -> None:
    """DNS rebinding: the peer is 127.0.0.1 but the Host header names the attacker's site."""
    client, _sup, _home = api
    r = client.get("/api/settings", headers={"host": host})
    assert r.status_code == 400
    assert r.text == "Invalid host header"


@pytest.mark.parametrize(
    "host", ["127.0.0.1:8765", "127.0.0.1:9000", "localhost:5173", "localhost"]
)
def test_the_hosts_the_panel_really_uses_pass(api, host) -> None:
    client, _sup, _home = api
    r = client.get("/api/health", headers={"host": host})
    assert r.status_code == 200


def test_the_default_test_client_host_is_the_panel_address(api) -> None:
    client, _sup, _home = api
    plain = TestClient(client.app, client=LOOPBACK)  # Host: testserver
    assert plain.get("/api/health").status_code == 400


def test_the_index_is_a_placeholder_until_the_ui_is_built(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "dist_dir", lambda: tmp_path / "no-dist-here")
    client, _sup, _home = make_client(tmp_path)
    try:
        r = client.get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert "The panel arrives in phase 4" in r.text
    finally:
        client.__exit__(None, None, None)


def test_a_built_ui_is_served_with_a_single_page_fallback(tmp_path: Path, monkeypatch) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>panel</title>", encoding="utf-8")
    (dist / "app.js").write_text("export const ok = 1;\n", encoding="utf-8")
    monkeypatch.setattr(app_module, "dist_dir", lambda: dist)
    client, _sup, _home = make_client(tmp_path / "home")
    try:
        assert client.get("/").text.startswith("<!doctype html>")
        assert "export const ok" in client.get("/app.js").text
        # A client-side route the browser deep-links into still gets index.html ...
        assert client.get("/instances/alpha").text.startswith("<!doctype html>")
        # ... but an unknown API path is still a 404, never the shell.
        assert client.get("/api/nope").status_code == 404
        assert client.get("/api/health").status_code == 200
    finally:
        client.__exit__(None, None, None)
