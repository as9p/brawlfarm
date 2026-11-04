"""Every core module imports with no .env, no BRAWL_* env, and an empty home directory."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest

MODULES = [
    "adb",
    "api",
    "brawlers",
    "config",
    "controller",
    "datalog",
    "events",
    "farmplan",
    "game_update",
    "janitor",
    "jsonio",
    "match_vision",
    "notify",
    "observer",
    "onboarding",
    "preview",
    "quests",
    "questpick",
    "recalib",
    "rewards",
    "scheduler",
    "settings",
    "states",
    "stats",
    "status",
    "upgrade_gate",
    "vision",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(f"brawlfarm.core.{name}")


def test_config_has_no_instances_or_tag_at_import(tmp_path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BRAWL_", "DISCORD_"))}
    env["BRAWLFARM_HOME"] = str(tmp_path)
    code = (
        "from brawlfarm.core import config; "
        "assert config.INSTANCES == {}, config.INSTANCES; "
        "assert config.PLAYER_TAG == '', config.PLAYER_TAG; "
        "assert config.HOME_DIR.exists(); "
        "assert config.TEMPLATES_DIR.joinpath('play.png').exists(), config.TEMPLATES_DIR"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_set_instances_feeds_scheduler_and_onboarding() -> None:
    from brawlfarm.core import config, onboarding

    config.set_instances({"Pie64": {"port": "5555", "tag": "", "data": "data/Pie64"}})
    try:
        assert "5555" in onboarding.farm_ports()
        assert config.INSTANCES["Pie64"]["data"] == "data/Pie64"
    finally:
        config.set_instances({})


def test_worker_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "brawlfarm.worker", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--max-games" in result.stdout


def test_worker_help_lists_observe() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "brawlfarm.worker", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--observe" in result.stdout
