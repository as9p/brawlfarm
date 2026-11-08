"""GET and PUT /api/settings: the whole config.toml document round-trips, a bad field is
named in the 422, and an instance whose worker is alive cannot be removed."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from brawlfarm import settings as S
from brawlfarm.supervisor import InstanceState
from tests.apihelpers import FakeWorld, make_client


@pytest.fixture()
def api(tmp_path: Path):
    client, sup, home = make_client(tmp_path, ("alpha", "bravo"))
    try:
        yield client, sup, home
    finally:
        client.__exit__(None, None, None)


def _heartbeat(home: Path, name: str, pid: int, now: datetime) -> None:
    d = S.instance_dir(home, name)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": (now - timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%S"),
        "pid": pid,
        "phase": "playing",
        "games_played": 3,
    }
    (d / "status.json").write_text(json.dumps(payload), encoding="utf-8")


def test_get_returns_the_whole_document(api) -> None:
    client, _sup, _home = api
    body = client.get("/api/settings").json()
    assert body["app"]["port"] == 8765
    assert body["scheduler"]["default_enabled"] is True
    assert body["connection"]["adb_path"].endswith("HD-Adb.exe")
    assert [i["name"] for i in body["instances"]] == ["alpha", "bravo"]


def test_defaults_are_the_model_defaults_with_no_instances(api) -> None:
    client, _sup, _home = api
    r = client.get("/api/settings/defaults")
    assert r.status_code == 200
    body = r.json()
    # No fleet: this is what a switch would go back to, not a document anyone saves.
    assert body["instances"] == []
    assert body["advanced"] == S.AppSettings().advanced.model_dump(mode="json")


def test_put_saves_the_document_and_reapplies_it(api) -> None:
    client, sup, home = api
    doc = client.get("/api/settings").json()
    doc["app"]["theme"] = "dark"
    doc["instances"].append({"name": "charlie", "adb_port": 5575, "player_tag": "#8GC9Q2"})
    r = client.put("/api/settings", json=doc)
    assert r.status_code == 200
    assert r.json()["app"]["theme"] == "dark"
    assert sup.settings.app.theme == "dark"
    assert [i.name for i in sup.settings.instances] == ["alpha", "bravo", "charlie"]
    assert "charlie" in S.config_path(home).read_text(encoding="utf-8")


def test_put_round_trips_the_shadow_flag_no_component_renders(api) -> None:
    client, sup, home = api
    doc = client.get("/api/settings").json()
    assert doc["behavior"]["shadow"] is False
    doc["behavior"]["shadow"] = True
    r = client.put("/api/settings", json=doc)
    assert r.status_code == 200
    assert r.json()["behavior"]["shadow"] is True
    assert sup.settings.behavior.shadow is True
    assert client.get("/api/settings").json()["behavior"]["shadow"] is True
    assert "shadow = true" in S.config_path(home).read_text(encoding="utf-8")


def test_put_round_trips_the_bush_quiet_flag(api) -> None:
    client, sup, home = api
    doc = client.get("/api/settings").json()
    assert doc["behavior"]["bush_quiet"] is False
    doc["behavior"]["bush_quiet"] = True
    r = client.put("/api/settings", json=doc)
    assert r.status_code == 200
    assert r.json()["behavior"]["bush_quiet"] is True
    assert sup.settings.behavior.bush_quiet is True
    assert client.get("/api/settings").json()["behavior"] == r.json()["behavior"]
    assert "bush_quiet = true" in S.config_path(home).read_text(encoding="utf-8")


def test_put_round_trips_the_emulator_recycle_fields(api) -> None:
    client, sup, home = api
    doc = client.get("/api/settings").json()
    assert doc["behavior"]["emulator_recycle_gb"] == 0.0
    assert doc["behavior"]["commit_floor_gb"] == 0.0
    assert doc["behavior"]["emulator_path"].endswith("HD-Player.exe")
    doc["behavior"]["emulator_recycle_gb"] = 40.5
    doc["behavior"]["commit_floor_gb"] = 12.0
    doc["behavior"]["emulator_path"] = r"D:\BlueStacks\HD-Player.exe"
    r = client.put("/api/settings", json=doc)
    assert r.status_code == 200
    assert sup.settings.behavior.emulator_recycle_gb == 40.5
    assert sup.settings.behavior.commit_floor_gb == 12.0
    assert client.get("/api/settings").json()["behavior"] == r.json()["behavior"]
    assert S.load(home).behavior.emulator_path == r"D:\BlueStacks\HD-Player.exe"


@pytest.mark.parametrize("field", ["emulator_recycle_gb", "commit_floor_gb"])
def test_put_refuses_a_negative_memory_limit(api, field) -> None:
    client, sup, home = api
    doc = client.get("/api/settings").json()
    doc["behavior"][field] = -1.0
    r = client.put("/api/settings", json=doc)
    assert r.status_code == 422
    assert field in r.json()["detail"]


def test_put_names_the_field_it_rejected(api) -> None:
    client, sup, _home = api
    doc = client.get("/api/settings").json()
    doc["app"]["prot"] = 1
    r = client.put("/api/settings", json=doc)
    assert r.status_code == 422
    assert "app.prot" in r.json()["detail"]
    assert sup.settings.app.port == 8765  # nothing was applied


def test_put_refuses_to_remove_an_instance_that_is_still_running(tmp_path: Path) -> None:
    world = FakeWorld()
    world.alive.add(4242)
    _heartbeat(tmp_path, "bravo", 4242, world.now)
    client, sup, _home = make_client(tmp_path, ("alpha", "bravo"), world=world)
    try:
        live = next(v for v in sup.views() if v.name == "bravo")
        assert live.state == InstanceState.FARMING
        doc = client.get("/api/settings").json()
        doc["instances"] = [i for i in doc["instances"] if i["name"] != "bravo"]
        r = client.put("/api/settings", json=doc)
        assert r.status_code == 409
        assert r.json() == {"detail": "Stop bravo before removing it"}
        assert [i.name for i in sup.settings.instances] == ["alpha", "bravo"]
    finally:
        client.__exit__(None, None, None)


def test_put_allows_removing_an_offline_instance_and_keeps_its_data(tmp_path: Path) -> None:
    world = FakeWorld()
    world.online[5555] = False  # alpha's BlueStacks window is not there
    client, sup, home = make_client(tmp_path, ("alpha", "bravo"), world=world)
    try:
        assert next(v for v in sup.views() if v.name == "alpha").state == InstanceState.OFFLINE
        S.instance_dir(home, "alpha").mkdir(parents=True, exist_ok=True)
        doc = client.get("/api/settings").json()
        doc["instances"] = [i for i in doc["instances"] if i["name"] != "alpha"]
        r = client.put("/api/settings", json=doc)
        assert r.status_code == 200
        assert [i["name"] for i in r.json()["instances"]] == ["bravo"]
        assert [i.name for i in sup.settings.instances] == ["bravo"]
        assert S.instance_dir(home, "alpha").exists()  # spec: removing keeps the data folder
    finally:
        client.__exit__(None, None, None)


def test_reset_restores_every_default_and_keeps_the_instances(api) -> None:
    client, sup, home = api
    doc = client.get("/api/settings").json()
    doc["app"]["theme"] = "dark"
    doc["behavior"]["gas_aware"] = False
    doc["notifications"]["ntfy_topic"] = "brawlfarm-test"
    doc["instances"][0]["player_tag"] = "#2P0YLQ9"
    assert client.put("/api/settings", json=doc).status_code == 200

    r = client.post("/api/settings/reset")
    assert r.status_code == 200
    body = r.json()
    assert body["app"]["theme"] == "system"
    assert body["behavior"]["gas_aware"] is True
    assert body["notifications"]["ntfy_topic"] == ""
    # The one thing a reset must not touch is the fleet, tags and ports included.
    assert [i["name"] for i in body["instances"]] == ["alpha", "bravo"]
    assert body["instances"][0]["player_tag"] == "#2P0YLQ9"
    assert body["instances"][1]["adb_port"] == 5565
    assert sup.settings.app.theme == "system"
    assert sup.settings.notifications.ntfy_topic == ""
    # It was written, not only applied: a restart has to come back reset.
    text = S.config_path(home).read_text(encoding="utf-8")
    assert "brawlfarm-test" not in text
    assert "bravo" in text


def test_reset_works_while_an_instance_is_farming(tmp_path: Path) -> None:
    world = FakeWorld()
    world.alive.add(4242)
    _heartbeat(tmp_path, "bravo", 4242, world.now)
    client, sup, _home = make_client(tmp_path, ("alpha", "bravo"), world=world)
    try:
        assert next(v for v in sup.views() if v.name == "bravo").state == InstanceState.FARMING
        sup.settings.app.theme = "dark"
        r = client.post("/api/settings/reset")
        # Nothing is removed by a reset, so the 409 that guards PUT cannot happen here.
        assert r.status_code == 200
        assert r.json()["app"]["theme"] == "system"
        assert [i.name for i in sup.settings.instances] == ["alpha", "bravo"]
    finally:
        client.__exit__(None, None, None)


def test_open_data_folder_asks_windows_to_open_the_home_directory(api, monkeypatch) -> None:
    client, _sup, home = api
    opened: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "startfile", lambda path: opened.append(path), raising=False)
    r = client.post("/api/settings/open-data-folder")
    assert r.status_code == 204
    assert r.content == b""
    assert opened == [str(home)]


def test_open_data_folder_off_windows_says_so(api, monkeypatch) -> None:
    client, _sup, _home = api
    monkeypatch.setattr(sys, "platform", "linux")
    r = client.post("/api/settings/open-data-folder")
    assert r.status_code == 501
    assert r.json() == {"detail": "Only on Windows"}


def test_open_data_folder_reports_a_refusal_without_naming_the_path(api, monkeypatch) -> None:
    client, _sup, home = api

    def denied(path):
        raise OSError("access is denied")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "startfile", denied, raising=False)
    r = client.post("/api/settings/open-data-folder")
    assert r.status_code == 500
    assert r.json() == {"detail": "could not open the data folder"}
    # The home directory is a Windows user path; it never goes into an error message.
    assert str(home) not in r.text
