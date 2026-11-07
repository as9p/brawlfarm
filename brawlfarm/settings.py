"""Typed settings: config.toml in the data directory, and the translation of those
settings into the worker's environment so the farm core never reads TOML.

This module must not import brawlfarm.core: it decides where the home directory is
before the core (which reads BRAWLFARM_HOME at import) is loaded.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

INSTANCE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}\Z")
# The game's tag alphabet: no vowels, no 1/3/4/5/6/7, so tags never spell words.
PLAYER_TAG_RE = re.compile(r"^#[0289PYLQGRJCUV]{3,}\Z")
DEFAULT_ADB_PATH = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"
DEFAULT_EMULATOR_PATH = r"C:\Program Files\BlueStacks_nxt\HD-Player.exe"
CONFIG_NAME = "config.toml"


class SettingsError(ValueError):
    """config.toml is unreadable or invalid; the message names the file and the reason."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AppSection(_Section):
    port: int = Field(default=8765, ge=1, le=65535)
    theme: Literal["system", "dark", "light"] = "system"


class ConnectionSection(_Section):
    adb_path: str = DEFAULT_ADB_PATH
    brawl_api_token: str = ""


class BehaviorSection(_Section):
    winrate_aware: bool = True
    opportunity_cost: bool = False  # off by default: it thrashed near-maxed rosters
    gas_aware: bool = True
    bush_hide: bool = False  # experimental, not validated live
    close_game_on_stop: bool = True
    dnd_at_start: bool = True
    shadow: bool = False  # play mode's detector runs beside the farm loop and logs; sends no input
    # HD-Player leaks memory, so the supervisor restarts it. 0 turns a limit off.
    emulator_recycle_gb: float = Field(default=0.0, ge=0)  # its private memory, above this
    commit_floor_gb: float = Field(default=0.0, ge=0)  # free commit charge, below this
    emulator_path: str = DEFAULT_EMULATOR_PATH


class AdvancedSection(_Section):
    """Performance and safety switches the core exposes as BRAWL_* flags. All on."""

    fast_input: bool = True
    raw_cap: bool = True
    gray_match: bool = True
    phase_classify: bool = True
    ability_buttons: bool = True
    recalib_tripwire: bool = True
    dnd_off_on_stop: bool = True


class SchedulerSection(_Section):
    default_enabled: bool = True


class NotificationsSection(_Section):
    webhook_url: str = ""  # any webhook that accepts {"content": ...}
    ntfy_topic: str = ""
    ntfy_server: str = "https://ntfy.sh"
    healthchecks_url: str = ""
    events: list[str] = Field(
        default_factory=lambda: ["crash", "recover", "offline", "wrong_mode", "recalibrate"]
    )


class InstanceSettings(_Section):
    name: str
    adb_port: int = Field(ge=1, le=65535)
    player_tag: str = ""  # optional; only used for the Brawl Stars API

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not INSTANCE_NAME_RE.match(v):
            raise ValueError("instance name must match [A-Za-z0-9_-]{1,32} (it becomes a folder)")
        return v

    @field_validator("player_tag")
    @classmethod
    def _normalise_tag(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            return ""
        if not v.startswith("#"):
            v = "#" + v
        if not PLAYER_TAG_RE.match(v):
            raise ValueError("player tag must be # followed by letters from 0289PYLQGRJCUV")
        return v


class AppSettings(_Section):
    app: AppSection = Field(default_factory=AppSection)
    connection: ConnectionSection = Field(default_factory=ConnectionSection)
    behavior: BehaviorSection = Field(default_factory=BehaviorSection)
    advanced: AdvancedSection = Field(default_factory=AdvancedSection)
    scheduler: SchedulerSection = Field(default_factory=SchedulerSection)
    notifications: NotificationsSection = Field(default_factory=NotificationsSection)
    instances: list[InstanceSettings] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_instances(self) -> AppSettings:
        names = [i.name for i in self.instances]
        ports = [i.adb_port for i in self.instances]
        if len(set(names)) != len(names):
            raise ValueError("instance names must be unique")
        if len(set(ports)) != len(ports):
            raise ValueError("instance adb ports must be unique (one worker per instance)")
        return self

    def instance(self, name: str) -> InstanceSettings:
        for inst in self.instances:
            if inst.name == name:
                return inst
        raise KeyError(name)


# --- Locations -----------------------------------------------------------------------


def default_home() -> Path:
    """BRAWLFARM_HOME if set (development, tests), else %LOCALAPPDATA%\\brawlfarm."""
    env = os.environ.get("BRAWLFARM_HOME", "").strip()
    if env:
        return Path(env).resolve()
    local = os.environ.get("LOCALAPPDATA", "").strip() or str(Path.home() / "AppData" / "Local")
    return (Path(local) / "brawlfarm").resolve()


def config_path(home: Path) -> Path:
    return Path(home) / CONFIG_NAME


def instance_dir(home: Path, name: str) -> Path:
    return Path(home) / "instances" / name


# --- Load / save ---------------------------------------------------------------------


def load(home: Path) -> AppSettings:
    """Read config.toml under ``home``; a missing file yields defaults."""
    path = config_path(home)
    if not path.exists():
        return AppSettings()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SettingsError(f"{path}: cannot read config.toml: {exc}") from exc
    try:
        return AppSettings.model_validate(raw)
    except ValidationError as exc:
        raise SettingsError(f"{path}: invalid config.toml: {_explain(exc)}") from exc


def _explain(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)


def save(settings: AppSettings, home: Path) -> Path:
    """Write config.toml atomically (temp file, then os.replace). Rewrites the whole file;
    comments a user added by hand are not preserved."""
    path = config_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = tomli_w.dumps(settings.model_dump(mode="json"))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


# --- Worker contract -----------------------------------------------------------------


def instances_table(settings: AppSettings) -> dict[str, dict[str, str]]:
    """The runtime instance table the core expects from ``config.set_instances``."""
    return {
        i.name: {"port": str(i.adb_port), "tag": i.player_tag, "data": f"instances/{i.name}"}
        for i in settings.instances
    }


def _flag(value: bool) -> str:
    return "1" if value else "0"


def worker_env(settings: AppSettings, inst: InstanceSettings, home: Path) -> dict[str, str]:
    """Everything a worker process reads from its environment, as strings. Passed to
    Popen(env=...) because core/config.py reads these at import time."""
    b, a, n, c = settings.behavior, settings.advanced, settings.notifications, settings.connection
    return {
        "BRAWLFARM_HOME": str(Path(home).resolve()),
        "BRAWL_DATA_DIR": f"instances/{inst.name}",
        "BRAWL_ADB_PORT": str(inst.adb_port),
        "BRAWL_PLAYER_TAG": inst.player_tag,
        "BRAWL_ADB_PATH": c.adb_path,
        "BRAWL_API_TOKEN": c.brawl_api_token,
        "BRAWL_WINRATE_AWARE": _flag(b.winrate_aware),
        "BRAWL_WINRATE_OPPORTUNITY_COST": _flag(b.opportunity_cost),
        "BRAWL_GAS_AWARE": _flag(b.gas_aware),
        "BRAWL_BUSH_HIDE": _flag(b.bush_hide),
        "BRAWL_PLAY_SHADOW": _flag(b.shadow),
        "BRAWL_CLOSE_GAME_ON_STOP": _flag(b.close_game_on_stop),
        "BRAWL_FAST_INPUT": _flag(a.fast_input),
        "BRAWL_RAW_CAP": _flag(a.raw_cap),
        "BRAWL_GRAY_MATCH": _flag(a.gray_match),
        "BRAWL_PHASE_CLASSIFY": _flag(a.phase_classify),
        "BRAWL_ABILITY_BUTTONS": _flag(a.ability_buttons),
        "BRAWL_RECALIB_TRIPWIRE": _flag(a.recalib_tripwire),
        "BRAWL_DND_OFF_ON_STOP": _flag(a.dnd_off_on_stop),
        "BRAWL_WEBHOOK_URL": n.webhook_url,
        "NTFY_TOPIC": n.ntfy_topic,
        "NTFY_SERVER": n.ntfy_server,
        "BRAWL_NOTIFY_EVENTS": ",".join(n.events),
    }


def worker_args(
    settings: AppSettings, max_minutes: float | None, *, observe: bool = False
) -> list[str]:
    """CLI flags for ``python -m brawlfarm.worker``: brawler selection always, DND when
    the behaviour setting says so, and the scheduler's session cap when there is one.

    Observe mode is the exception and takes the flag alone: it never plays, so a startup
    task that would touch the game has no business in its command line."""
    if observe:
        return ["--observe"]
    args = ["--select-brawler"]
    if settings.behavior.dnd_at_start:
        args.append("--dnd")
    if max_minutes is not None:
        args += ["--max-minutes", f"{max_minutes:g}"]
    return args
