"""config.toml settings model: defaults when the file is missing, round-trip through
save/load, validation of instance names, ports and tags, and the worker env contract."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from brawlfarm import settings as S


def _two_instances() -> S.AppSettings:
    return S.AppSettings(
        instances=[
            S.InstanceSettings(name="Pie64", adb_port=5555, player_tag="#2P0YLQ9"),
            S.InstanceSettings(name="Rome64", adb_port=5565),
        ]
    )


def test_defaults_match_the_spec() -> None:
    s = S.AppSettings()
    assert s.app.port == 8765
    assert s.app.theme == "system"
    assert s.connection.adb_path == r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"
    assert s.connection.brawl_api_token == ""
    assert s.behavior.winrate_aware is True
    assert s.behavior.opportunity_cost is False
    assert s.behavior.gas_aware is True
    assert s.behavior.bush_hide is False
    assert s.behavior.close_game_on_stop is True
    assert s.behavior.dnd_at_start is True
    assert s.behavior.shadow is False
    assert s.behavior.emulator_recycle_gb == 0.0
    assert s.behavior.commit_floor_gb == 0.0
    assert s.behavior.emulator_path == r"C:\Program Files\BlueStacks_nxt\HD-Player.exe"
    assert s.scheduler.default_enabled is True
    assert s.notifications.webhook_url == ""
    assert s.notifications.ntfy_topic == ""
    assert s.notifications.ntfy_server == "https://ntfy.sh"
    assert s.notifications.healthchecks_url == ""
    assert s.notifications.events == ["crash", "recover", "offline", "wrong_mode", "recalibrate"]
    assert s.instances == []
    assert s.advanced.fast_input is True
    assert s.advanced.raw_cap is True
    assert s.advanced.gray_match is True
    assert s.advanced.phase_classify is True
    assert s.advanced.ability_buttons is True
    assert s.advanced.recalib_tripwire is True
    assert s.advanced.dnd_off_on_stop is True


def test_load_missing_file_gives_defaults(tmp_path: Path) -> None:
    assert S.load(tmp_path) == S.AppSettings()
    assert not S.config_path(tmp_path).exists()


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    original = _two_instances()
    original.app.port = 9000
    original.behavior.bush_hide = True
    original.behavior.shadow = True
    original.notifications.events = ["crash"]
    path = S.save(original, tmp_path)
    assert path == tmp_path / "config.toml"
    assert S.load(tmp_path) == original
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    assert raw["instances"][0] == {"name": "Pie64", "adb_port": 5555, "player_tag": "#2P0YLQ9"}
    assert raw["app"]["port"] == 9000


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    S.save(S.AppSettings(), tmp_path)
    S.save(_two_instances(), tmp_path)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "config.toml"]
    assert leftovers == []
    assert len(S.load(tmp_path).instances) == 2


def test_malformed_toml_raises_settings_error(tmp_path: Path) -> None:
    S.config_path(tmp_path).write_text("[app\nport = 1\n", encoding="utf-8")
    with pytest.raises(S.SettingsError) as exc:
        S.load(tmp_path)
    assert "config.toml" in str(exc.value)


def test_config_written_before_the_shadow_flag_still_loads(tmp_path: Path) -> None:
    S.config_path(tmp_path).write_text("[behavior]\nbush_hide = true\n", encoding="utf-8")
    s = S.load(tmp_path)
    assert s.behavior.bush_hide is True
    assert s.behavior.shadow is False


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    S.config_path(tmp_path).write_text("[app]\nprot = 1\n", encoding="utf-8")
    with pytest.raises(S.SettingsError) as exc:
        S.load(tmp_path)
    assert "prot" in str(exc.value)


@pytest.mark.parametrize("bad", ["", "a" * 33, "bad name", "../x", "dot.name", "ok\n"])
def test_instance_name_is_validated(bad: str) -> None:
    with pytest.raises(ValueError):
        S.InstanceSettings(name=bad, adb_port=5555)


@pytest.mark.parametrize("port", [0, 65536, -1])
def test_adb_port_range(port: int) -> None:
    with pytest.raises(ValueError):
        S.InstanceSettings(name="ok", adb_port=port)


def test_player_tag_is_normalised_or_rejected() -> None:
    assert S.InstanceSettings(name="a", adb_port=1, player_tag="2p0ylq9").player_tag == "#2P0YLQ9"
    assert S.InstanceSettings(name="a", adb_port=1, player_tag="  ").player_tag == ""
    with pytest.raises(ValueError):
        S.InstanceSettings(name="a", adb_port=1, player_tag="#ABC")  # A and B are not tag letters
    # The validator strips whitespace before matching, so a trailing newline never reaches
    # the pattern through the model; assert on the pattern itself, which must reject one.
    assert not S.PLAYER_TAG_RE.match("#2P0YLQ9\n")


def test_duplicate_names_and_ports_are_rejected() -> None:
    with pytest.raises(ValueError):
        S.AppSettings(
            instances=[
                S.InstanceSettings(name="x", adb_port=1),
                S.InstanceSettings(name="x", adb_port=2),
            ]
        )
    with pytest.raises(ValueError):
        S.AppSettings(
            instances=[
                S.InstanceSettings(name="x", adb_port=1),
                S.InstanceSettings(name="y", adb_port=1),
            ]
        )


def test_theme_is_restricted() -> None:
    with pytest.raises(ValueError):
        S.AppSettings(app={"theme": "neon"})


def test_default_home_prefers_env_then_localappdata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BRAWLFARM_HOME", str(tmp_path / "h"))
    assert S.default_home() == (tmp_path / "h").resolve()
    monkeypatch.delenv("BRAWLFARM_HOME")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    assert S.default_home() == (tmp_path / "lad" / "brawlfarm").resolve()


def test_instance_dir_and_table() -> None:
    s = _two_instances()
    home = Path("C:/h")
    assert S.instance_dir(home, "Pie64") == home / "instances" / "Pie64"
    assert S.instances_table(s) == {
        "Pie64": {"port": "5555", "tag": "#2P0YLQ9", "data": "instances/Pie64"},
        "Rome64": {"port": "5565", "tag": "", "data": "instances/Rome64"},
    }


def test_worker_env_is_the_full_contract(tmp_path: Path) -> None:
    s = _two_instances()
    s.behavior.bush_hide = False
    s.behavior.opportunity_cost = True
    s.advanced.raw_cap = False
    s.connection.brawl_api_token = "tok"
    s.notifications.webhook_url = "https://example.invalid/hook"
    s.notifications.ntfy_topic = "farm"
    s.notifications.events = ["crash", "offline"]
    env = S.worker_env(s, s.instances[0], tmp_path)
    assert env == {
        "BRAWLFARM_HOME": str(tmp_path.resolve()),
        "BRAWL_DATA_DIR": "instances/Pie64",
        "BRAWL_ADB_PORT": "5555",
        "BRAWL_PLAYER_TAG": "#2P0YLQ9",
        "BRAWL_ADB_PATH": r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe",
        "BRAWL_API_TOKEN": "tok",
        "BRAWL_WINRATE_AWARE": "1",
        "BRAWL_WINRATE_OPPORTUNITY_COST": "1",
        "BRAWL_GAS_AWARE": "1",
        "BRAWL_BUSH_HIDE": "0",
        "BRAWL_BUSH_QUIET": "0",
        "BRAWL_PLAY_SHADOW": "0",
        "BRAWL_CLOSE_GAME_ON_STOP": "1",
        "BRAWL_FAST_INPUT": "1",
        "BRAWL_RAW_CAP": "0",
        "BRAWL_GRAY_MATCH": "1",
        "BRAWL_PHASE_CLASSIFY": "1",
        "BRAWL_ABILITY_BUTTONS": "1",
        "BRAWL_RECALIB_TRIPWIRE": "1",
        "BRAWL_DND_OFF_ON_STOP": "1",
        "BRAWL_WEBHOOK_URL": "https://example.invalid/hook",
        "NTFY_TOPIC": "farm",
        "NTFY_SERVER": "https://ntfy.sh",
        "BRAWL_NOTIFY_EVENTS": "crash,offline",
    }
    assert all(isinstance(v, str) for v in env.values())


def test_worker_env_carries_shadow_when_it_is_on(tmp_path: Path) -> None:
    s = _two_instances()
    s.behavior.shadow = True
    assert S.worker_env(s, s.instances[0], tmp_path)["BRAWL_PLAY_SHADOW"] == "1"


def test_config_play_shadow_follows_the_env() -> None:
    """The worker reads the other end of the flag at import time, so check it in a fresh
    process rather than by reloading the core into this one."""
    base = {k: v for k, v in os.environ.items() if not k.startswith(("BRAWL_", "DISCORD_"))}
    code = "from brawlfarm.core import config; print('1' if config.PLAY_SHADOW else '0')"
    for value, expected in ((None, "0"), ("0", "0"), ("1", "1")):
        env = dict(base)
        if value is not None:
            env["BRAWL_PLAY_SHADOW"] = value
        result = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected, value


def test_worker_env_carries_bush_quiet_when_it_is_on(tmp_path: Path) -> None:
    s = _two_instances()
    assert s.behavior.bush_quiet is False
    s.behavior.bush_quiet = True
    assert S.worker_env(s, s.instances[0], tmp_path)["BRAWL_BUSH_QUIET"] == "1"


def test_config_bush_quiet_follows_the_env() -> None:
    """Off unless the env says 1, read at import time in a fresh process."""
    base = {k: v for k, v in os.environ.items() if not k.startswith(("BRAWL_", "DISCORD_"))}
    code = "from brawlfarm.core import config; print('1' if config.BUSH_QUIET else '0')"
    for value, expected in ((None, "0"), ("0", "0"), ("1", "1")):
        env = dict(base)
        if value is not None:
            env["BRAWL_BUSH_QUIET"] = value
        result = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected, value


def test_worker_args_follow_dnd_and_cap() -> None:
    s = S.AppSettings()
    assert S.worker_args(s, None) == ["--select-brawler", "--dnd"]
    s.behavior.dnd_at_start = False
    assert S.worker_args(s, 42.7) == ["--select-brawler", "--max-minutes", "42.7"]


def test_worker_args_for_observe_carry_nothing_that_plays() -> None:
    s = S.AppSettings()
    s.behavior.dnd_at_start = True
    assert S.worker_args(s, None, observe=True) == ["--observe"]
    assert S.worker_args(s, 42.7, observe=True) == ["--observe"]


def test_settings_module_does_not_import_the_core() -> None:
    """settings.py decides BRAWLFARM_HOME before the core (which reads it at import) loads,
    so it must never import brawlfarm.core. Checked on the source, not by reloading modules."""
    source = Path(S.__file__).read_text(encoding="utf-8")
    offenders = [
        line
        for line in source.splitlines()
        if re.match(r"^\s*(from|import)\s+brawlfarm\.core", line)
    ]
    assert offenders == []
