/**
 * The settings document.
 *
 * GET and PUT are both whole-document: the API has no PATCH, so every write goes through
 * settings/useSettingsPatch.ts, which reads, mutates a clone and puts the whole thing back.
 * The server masks connection.brawl_api_token and notifications.webhook_url in every
 * answer (eight asterisks plus the last four characters). Putting that masked value back
 * unchanged keeps the stored secret, so a whole-document PUT from any screen is safe; a
 * value that differs from the mask is saved as typed.
 */
import { api } from "./client";
import type { AppSettings, NotifyTestResponse } from "./types";

export function getSettings(): Promise<AppSettings> {
  return api<AppSettings>("/api/settings");
}

/** GET /api/settings/defaults. Every section at its model default and no instances, which
 * is how a section shows what a switch would go back to. Read only: the panel never saves
 * this document, it only compares against it. */
export function getSettingsDefaults(): Promise<AppSettings> {
  return api<AppSettings>("/api/settings/defaults");
}

export function putSettings(doc: AppSettings): Promise<AppSettings> {
  return api<AppSettings>("/api/settings", { method: "PUT", body: JSON.stringify(doc) });
}

/** POST /api/settings/reset. Returns the document it wrote: every section back to its model
 * default, with the instances list carried over untouched. */
export function resetSettings(): Promise<AppSettings> {
  return api<AppSettings>("/api/settings/reset", { method: "POST" });
}

/** POST /api/settings/open-data-folder. 204 on Windows; 501 "Only on Windows" anywhere
 * else, which the caller toasts. The path itself never crosses the wire. */
export function openDataFolder(): Promise<void> {
  return api<void>("/api/settings/open-data-folder", { method: "POST" });
}

/** POST /api/notifications/test. The body is the server's own list of channel names that
 * answered and channel names that did not; nothing here sees a URL or a topic. */
export function testNotifications(): Promise<NotifyTestResponse> {
  return api<NotifyTestResponse>("/api/notifications/test", { method: "POST" });
}
