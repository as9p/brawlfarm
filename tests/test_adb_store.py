"""Play Store helpers for the game auto-update: open the store page and read the
installed version. Every call is checked as an argument list; nothing reaches a real adb."""

from __future__ import annotations

import pytest

from brawlfarm.core import adb, config


@pytest.fixture()
def calls(monkeypatch) -> list[list[str]]:
    seen: list[list[str]] = []

    def fake_run(args, **kwargs):
        seen.append(list(args))
        return ""

    monkeypatch.setattr(adb, "_run", fake_run)
    monkeypatch.setattr(config, "ADB_SERIAL", "127.0.0.1:5555")
    return seen


def test_open_store_page_sends_a_quoted_market_url(calls) -> None:
    adb.open_store_page("com.supercell.brawlstars")
    assert calls == [
        [
            "-s",
            "127.0.0.1:5555",
            "shell",
            "am",
            "start",
            "-a",
            "android.intent.action.VIEW",
            "-d",
            "'market://details?id=com.supercell.brawlstars'",
        ]
    ]


@pytest.mark.parametrize("bad", ["", "com.x;reboot", "a b", "com.x'$(id)'", "com/x"])
def test_open_store_page_refuses_unsafe_package(calls, bad) -> None:
    with pytest.raises(ValueError):
        adb.open_store_page(bad)
    assert calls == []


def test_package_version_parses_version_name(monkeypatch) -> None:
    out = (
        "Packages:\n  Package [com.supercell.brawlstars] (abc):\n"
        "    versionCode=69230 minSdk=24 targetSdk=34\n    versionName=62.264\n"
    )
    seen: list[list[str]] = []

    def fake_run(args, **kwargs):
        seen.append(list(args))
        return out

    monkeypatch.setattr(adb, "_run", fake_run)
    monkeypatch.setattr(config, "ADB_SERIAL", "127.0.0.1:5555")
    assert adb.package_version("com.supercell.brawlstars") == "62.264"
    assert seen == [
        ["-s", "127.0.0.1:5555", "shell", "dumpsys", "package", "com.supercell.brawlstars"]
    ]


def test_package_version_none_when_missing_or_adb_fails(monkeypatch) -> None:
    monkeypatch.setattr(adb, "_run", lambda args, **k: "Unable to find package")
    assert adb.package_version("com.supercell.brawlstars") is None

    def boom(args, **kwargs):
        raise adb.AdbError("offline")

    monkeypatch.setattr(adb, "_run", boom)
    assert adb.package_version("com.supercell.brawlstars") is None


def test_package_version_refuses_unsafe_package(calls) -> None:
    with pytest.raises(ValueError):
        adb.package_version("x; rm -rf /")
    assert calls == []


def test_store_constants_are_sane() -> None:
    assert config.PLAY_STORE_PACKAGE == "com.android.vending"
    assert config.STORE_UPDATE_NEEDLE == "update"
    assert "uninstall" in config.STORE_NEVER_TAP
    assert config.GAME_UPDATE_ON_ATTEMPT == 2
    assert config.GAME_UPDATE_TIMEOUT_S == 600
    assert config.GAME_UPDATE_POLL_S == 15
    assert config.STORE_OPEN_TIMEOUT_S == 15
