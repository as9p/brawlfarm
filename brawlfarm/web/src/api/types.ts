/**
 * The API's payloads as TypeScript.
 *
 * Mirrors brawlfarm/api/*.py exactly: optional in Python means `| null` here, never `?`,
 * because the API always sends the key.
 */

export type InstanceState =
  | "farming"
  | "stopped"
  | "scheduled_break"
  | "reconnecting"
  | "offline"
  | "starting"
  | "stopping";

export type Health = "healthy" | "stale" | "dead";

export type Phase = "at_menu" | "queuing" | "playing" | "returning";

export interface InstanceSession {
  minutes_elapsed: number | null;
  start_trophies: number | null;
  last_trophies: number | null;
  disconnect_count: number | null;
  recovery_attempts: number | null;
  session: string | null;
}

export interface Today {
  games: number;
  trophies: number;
}

export interface InstancePayload {
  name: string;
  adb_port: number;
  state: InstanceState;
  health: Health;
  pid: number | null;
  heartbeat_age_s: number | null;
  /** The worker's own phase string; unknown values are shown as they arrive. */
  phase: Phase | string | null;
  desired: "run" | "stop" | "observe";
  desired_reason: string | null;
  until: string | null;
  games_played: number | null;
  farm_brawler: string | null;
  note: string;
  player_tag: string;
  session: InstanceSession | null;
  today: Today;
  /** The newest finished session in the instance folder, so a stopped card is not all
   * zeros on a cold load. Null when that instance has never written one. */
  last_session: LastSession | null;
}

export interface InstancesResponse {
  instances: InstancePayload[];
}

export interface FarmPlan {
  mode: "ladder" | "prestige";
  prestige_start: "highest" | "lowest";
  goal_trophies: number;
  maxed_fallback: string | null;
  quest_aware: boolean;
}

export interface RosterBrawler {
  id: number;
  name: string;
  trophies: number;
  highest: number;
  rank: number;
  power: number;
}

export type RosterStatus = "ok" | "no_token" | "no_tag" | "unavailable";

export interface PlanCurrent {
  brawler: string | null;
  trophies: number | null;
  goal: number;
}

export interface PlanResponse extends FarmPlan {
  current: PlanCurrent;
  roster: RosterBrawler[] | null;
  queue: string[];
  roster_status: RosterStatus;
}

export interface ScheduleSession {
  start: string;
  end: string;
}

export interface ScheduleOverride {
  mode: "run" | "stop";
  until: string;
  set_at: string | null;
}

export interface SchedulePayload {
  enabled: boolean;
  override: ScheduleOverride | null;
  plan_date: string | null;
  sessions: ScheduleSession[];
  day_end: string | null;
  desired: Record<string, unknown> | null;
  games_played_today: number;
  now: string;
}

export interface SchedulePatch {
  enabled?: boolean;
  redraw?: boolean;
  clear_override?: boolean;
}

export type FeedCategory = "matches" | "interrupts" | "errors" | "other";

export type FeedKind = "all" | "matches" | "interrupts" | "errors";

export interface FeedRecord {
  ts: string;
  /** The line's 1-based position in the session file; unique within one session. */
  seq: number;
  event: string;
  category: FeedCategory;
  fields: Record<string, unknown>;
}

export interface FeedResponse {
  session: string | null;
  records: FeedRecord[];
}

/** The SSE "feed" payload. */
export interface FeedEvent {
  instance: string;
  session: string | null;
  record: FeedRecord;
}

export interface Alert {
  id: number;
  ts: string;
  instance: string;
  kind: string;
  title: string;
  detail: string;
  dismissed: boolean;
}

export interface AlertsResponse {
  alerts: Alert[];
  unread: number;
}

export interface StatsSummary {
  games: number;
  trophies: number;
  trophies_per_hour: number | null;
  avg_placement: number | null;
  /** Percent of placed games that finished first. */
  win_rate: number | null;
  hours_farmed: number;
  /** Exactly four entries, placements 1 to 4. */
  trophies_by_placement: StatsTrophiesByPlacement[];
}

export interface StatsTrophiesByPlacement {
  placement: number;
  games: number;
  avg: number | null;
}

export type StatsRange = "today" | "7d" | "30d" | "all";

export interface StatsPoint {
  t: string;
  cum: number;
}

export interface StatsSeries {
  instance: string;
  points: StatsPoint[];
}

export interface StatsBrawler {
  name: string;
  games: number;
  net: number;
  avg_placement: number | null;
  win_rate: number | null;
}

export interface StatsPlacement {
  placement: number;
  games: number;
}

export interface StatsGame {
  instance: string | null;
  t: string;
  brawler: string | null;
  placement: number | null;
  trophy_change: number | null;
  map: string | null;
  mode: string | null;
}

/** StatsResponse grows from the summary-only phase 4 shape to the whole aggregate. */
export interface StatsResponse {
  range: string;
  instances: string[];
  summary: StatsSummary;
  series: StatsSeries[];
  brawlers: StatsBrawler[];
  /** Every placement in the data, ascending, 5 and above included. */
  placements: StatsPlacement[];
  recent: StatsGame[];
}

/** GET /api/connection/check. RosterStatus deliberately stays four values, because the
 * plan route still answers with those four; "rejected" and "unreachable" live only here. */
export type ConnectionStatus = "ok" | "no_token" | "no_tag" | "rejected" | "unreachable";

export interface ConnectionCheck {
  status: ConnectionStatus;
  checked_at: string;
}

export interface LastSession {
  games: number;
  trophies: number;
  avg_placement: number | null;
  disconnects: number;
  duration_s: number;
  interrupts: number;
  ended_at: string;
}

/**
 * The whole config.toml document, as GET /api/settings serves it and PUT takes it back.
 *
 * connection.brawl_api_token is modelled because the document is whole: the API is
 * loopback-only and hands the token over in full, and masking it is the panel's job, not
 * the server's. It reaches exactly one control, the masked Field on Settings > Connection
 * and on the wizard's Stats step. It is never logged, never stored anywhere but the query
 * cache and the PUT body, never rendered unmasked by default, and never put into a toast,
 * an error message or a screenshot.
 */
export interface AppSettings {
  app: { port: number; theme: "system" | "dark" | "light" };
  connection: { adb_path: string; brawl_api_token: string };
  behavior: {
    winrate_aware: boolean;
    opportunity_cost: boolean;
    gas_aware: boolean;
    bush_hide: boolean;
    close_game_on_stop: boolean;
    dnd_at_start: boolean;
    shadow: boolean;
    emulator_recycle_gb: number;
    commit_floor_gb: number;
    emulator_path: string;
  };
  advanced: {
    fast_input: boolean;
    raw_cap: boolean;
    gray_match: boolean;
    phase_classify: boolean;
    ability_buttons: boolean;
    recalib_tripwire: boolean;
    dnd_off_on_stop: boolean;
  };
  scheduler: { default_enabled: boolean };
  notifications: {
    webhook_url: string;
    ntfy_topic: string;
    ntfy_server: string;
    healthchecks_url: string;
    events: string[];
  };
  instances: { name: string; adb_port: number; player_tag: string }[];
}

/** One row of POST /api/setup/scan: a BlueStacks instance as bluestacks.conf describes it,
 * with `online` from `adb devices`. `adb_port` is null when the conf line was unreadable,
 * which is what the wizard's "Add a port" field is for. */
export interface ScanInstance {
  name: string;
  display_name: string;
  adb_port: number | null;
  width: number | null;
  height: number | null;
  dpi: number | null;
  online: boolean;
}

export interface ScanResponse {
  adb_path: string | null;
  adb_found: boolean;
  conf_found: boolean;
  instances: ScanInstance[];
}

export interface PortTestResponse {
  ok: boolean;
  detail: string;
}

/** POST /api/setup/display-check. `hint` is checks.DISPLAY_HINT, rendered verbatim so the
 * sentence telling the user where to change the display has exactly one source. */
export interface DisplayCheckResponse {
  ok: boolean;
  width: number | null;
  height: number | null;
  dpi: number | null;
  detail: string;
  hint: string;
  expected: { width: number; height: number; dpi: number };
}

export interface NotifyTestResponse {
  sent: string[];
  failed: string[];
}

/** GET /api/health. Not called `Health`: that name is already this file's instance-health
 * union ("healthy" | "stale" | "dead"), and the two mean different things. */
export interface HealthResponse {
  version: string;
  home: string;
  instances: number;
  uptime_s: number;
}
