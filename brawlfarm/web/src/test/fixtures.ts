/**
 * Payload builders for the tests. Shapes match the API exactly, so a component test
 * fails the same way the real payload would. The names, ports and tag are invented:
 * Pie64 / Pie64_1 / Pie64_3 on 5555 / 5565 / 5585 with the made-up tag #2P0YLQ9, which
 * tools/scrub_check.py does not match.
 */
import type {
  Alert,
  AppSettings,
  ConnectionCheck,
  FeedRecord,
  InstancePayload,
  LastSession,
  PlanResponse,
  RosterBrawler,
  SchedulePayload,
  StatsResponse,
} from "../api/types";

export function makeInstance(overrides: Partial<InstancePayload> = {}): InstancePayload {
  return {
    name: "Pie64",
    adb_port: 5555,
    state: "farming",
    health: "healthy",
    pid: 4242,
    heartbeat_age_s: 3.5,
    phase: "playing",
    desired: "run",
    desired_reason: "session",
    until: "2026-09-11T20:18:00",
    games_played: 12,
    farm_brawler: "NORI",
    note: "",
    player_tag: "#2P0YLQ9",
    session: {
      minutes_elapsed: 72,
      start_trophies: 41200,
      last_trophies: 41286,
      disconnect_count: 1,
      recovery_attempts: 0,
      session: "session-20260911-190540.jsonl",
    },
    today: { games: 12, trophies: 86 },
    last_session: null,
    ...overrides,
  };
}

export function makeAlert(overrides: Partial<Alert> = {}): Alert {
  return {
    id: 1,
    ts: "2026-09-11T19:42:00",
    instance: "Pie64",
    kind: "crash",
    title: "Bot crashed",
    detail: "err=adb did not answer",
    dismissed: false,
    ...overrides,
  };
}

export function makeFeedRecord(overrides: Partial<FeedRecord> = {}): FeedRecord {
  return {
    ts: "2026-09-11T19:05:40",
    seq: 1,
    event: "phase",
    category: "matches",
    fields: { to: "playing", frm: "queuing", games: 3 },
    ...overrides,
  };
}

export function makeRosterBrawler(overrides: Partial<RosterBrawler> = {}): RosterBrawler {
  return {
    id: 16000101,
    name: "NORI",
    trophies: 812,
    highest: 830,
    rank: 25,
    power: 11,
    ...overrides,
  };
}

export function makePlan(overrides: Partial<PlanResponse> = {}): PlanResponse {
  return {
    mode: "ladder",
    prestige_start: "highest",
    goal_trophies: 1000,
    maxed_fallback: null,
    quest_aware: false,
    current: { brawler: "NORI", trophies: 812, goal: 1000 },
    roster: [
      makeRosterBrawler(),
      makeRosterBrawler({ id: 16000000, name: "SHELLY", trophies: 615, highest: 700, rank: 20 }),
      makeRosterBrawler({ id: 16000002, name: "TARA", trophies: 540, highest: 615, rank: 19 }),
    ],
    queue: ["TARA", "SHELLY"],
    roster_status: "ok",
    ...overrides,
  };
}

export function makeSchedule(overrides: Partial<SchedulePayload> = {}): SchedulePayload {
  return {
    enabled: true,
    override: null,
    plan_date: "2026-09-11",
    sessions: [
      { start: "2026-09-11T09:00:00", end: "2026-09-11T11:00:00" },
      { start: "2026-09-11T13:30:00", end: "2026-09-11T15:00:00" },
      { start: "2026-09-11T19:00:00", end: "2026-09-11T20:30:00" },
    ],
    day_end: "2026-09-11T20:30:00",
    desired: null,
    games_played_today: 12,
    now: "2026-09-11T14:15:00",
    ...overrides,
  };
}

/** The whole settings document at its model defaults, with one instance. Every settings
 * and wizard test starts from here, so a field that changes shape breaks one file rather
 * than twenty. The defaults mirror brawlfarm/settings.py exactly, including the empty
 * token: a fixture that carried a real one would be a leak waiting to happen. */
export function makeSettings(overrides: Partial<AppSettings> = {}): AppSettings {
  return {
    app: { port: 8765, theme: "system" },
    connection: {
      adb_path: "C:\\Program Files\\BlueStacks_nxt\\HD-Adb.exe",
      brawl_api_token: "",
    },
    behavior: {
      winrate_aware: true,
      opportunity_cost: false,
      gas_aware: true,
      bush_hide: false,
      close_game_on_stop: true,
      dnd_at_start: true,
      shadow: false,
      emulator_recycle_gb: 0,
      commit_floor_gb: 0,
      emulator_path: "C:\\Program Files\\BlueStacks_nxt\\HD-Player.exe",
    },
    advanced: {
      fast_input: true,
      raw_cap: true,
      gray_match: true,
      phase_classify: true,
      ability_buttons: true,
      recalib_tripwire: true,
      dnd_off_on_stop: true,
    },
    scheduler: { default_enabled: true },
    notifications: {
      webhook_url: "",
      ntfy_topic: "",
      ntfy_server: "https://ntfy.sh",
      healthchecks_url: "",
      events: ["crash", "recover", "offline", "wrong_mode", "recalibrate"],
    },
    instances: [{ name: "Pie64", adb_port: 5555, player_tag: "#2P0YLQ9" }],
    ...overrides,
  };
}

/** The whole stats aggregate for one range: two instances, four games, enough of every
 * block that a component test never has to hand-write one. The tag-free names and the
 * invented brawler names keep tools/scrub_check.py quiet. */
export function makeStats(overrides: Partial<StatsResponse> = {}): StatsResponse {
  return {
    range: "7d",
    instances: ["Pie64", "Pie64_1"],
    summary: {
      games: 4,
      trophies: 37,
      trophies_per_hour: 24.7,
      avg_placement: 3,
      win_rate: 25,
      hours_farmed: 1.5,
      trophies_by_placement: [
        { placement: 1, games: 1, avg: 17 },
        { placement: 2, games: 1, avg: 11 },
        { placement: 3, games: 0, avg: null },
        { placement: 4, games: 1, avg: 2 },
      ],
    },
    series: [
      {
        instance: "Pie64",
        points: [
          { t: "2026-09-12T21:00:00", cum: 12 },
          { t: "2026-09-12T21:30:00", cum: 8 },
          { t: "2026-09-12T22:00:00", cum: 25 },
        ],
      },
      {
        instance: "Pie64_1",
        points: [
          { t: "2026-09-12T21:10:00", cum: -3 },
          { t: "2026-09-12T22:10:00", cum: 12 },
        ],
      },
    ],
    brawlers: [
      { name: "NORI", games: 3, net: 29, avg_placement: 2.3, win_rate: 33.3 },
      { name: "SHELLY", games: 1, net: 8, avg_placement: 5, win_rate: 0 },
    ],
    placements: [
      { placement: 1, games: 1 },
      { placement: 2, games: 1 },
      { placement: 4, games: 1 },
      { placement: 5, games: 1 },
    ],
    recent: [
      {
        instance: "Pie64",
        t: "2026-09-12T22:00:00",
        brawler: "NORI",
        placement: 1,
        trophy_change: 17,
        map: "Feast or Famine",
        mode: "soloShowdown",
      },
      {
        instance: "Pie64_1",
        t: "2026-09-12T21:10:00",
        brawler: "SHELLY",
        placement: 5,
        trophy_change: -3,
        map: null,
        mode: null,
      },
    ],
    ...overrides,
  };
}

export function makeConnection(overrides: Partial<ConnectionCheck> = {}): ConnectionCheck {
  return { status: "ok", checked_at: "2026-09-12T22:14:07", ...overrides };
}

export function makeLastSession(overrides: Partial<LastSession> = {}): LastSession {
  return {
    games: 12,
    trophies: 86,
    avg_placement: 3.4,
    disconnects: 1,
    duration_s: 4447,
    interrupts: 2,
    ended_at: "2026-09-12T22:14:07",
    ...overrides,
  };
}
