/**
 * Settings > Connection: how brawlfarm reaches BlueStacks and the Brawl Stars API.
 *
 * The adb path carries a chip from one scan run when the section mounts. One, not one per
 * keystroke: the scan shells out to adb and can take seconds, and a cached one would lie
 * about what is installed right now. A scan that failed outright leaves the chip off rather
 * than guessing, because "Not found" would be a claim about adb that the panel cannot make
 * when it could not reach its own server.
 *
 * Both fields save 500 ms after the last keystroke and at once on blur. Neither has a Save
 * button, and neither ever logs what was typed: the token reaches the masked Field, the PUT
 * body and nothing else. The server only ever sends the token back masked, so the Field
 * shows that mask until a new token is typed over it.
 *
 * Check connection asks GET /api/connection/check, which reads the token and the player tags
 * from disk and never takes a typed value, so it waits until the form has nothing unsaved in
 * it. Its answer is the same strip the stats page shows, because the four statuses that name
 * something to fix already have one sentence each and a second wording for them would be a
 * second thing to keep true. A pass has nothing to fix, and no strip, so it raises a toast.
 */
import { useEffect, useState } from "react";
import { Link } from "react-router";

import { SettingRow } from "./SettingRow";
import {
  type SettingsPatch,
  fieldError,
  saveSettingAsync,
  useDebouncedSave,
} from "./useSettingsPatch";
import { getConnection } from "../api/connection";
import { scanSetup } from "../api/setup";
import type { ConnectionCheck, ScanResponse } from "../api/types";
import { Button } from "../components/ui/Button";
import { Chip } from "../components/ui/Chip";
import { ErrorBlock } from "../components/ui/ErrorBlock";
import { Field } from "../components/ui/Field";
import { toast } from "../lib/toast";
import { ConnectionStrip } from "../stats/ConnectionStrip";

const TOKEN_LINE = "Create a key at developer.brawlstars.com and allow this machine's IP address.";

export function Connection({ settingsPatch }: { settingsPatch: SettingsPatch }) {
  const { settings, patch, fieldErrors } = settingsPatch;
  const [scan, setScan] = useState<ScanResponse | null>(null);
  const [scanning, setScanning] = useState(true);
  const [check, setCheck] = useState<ConnectionCheck | null>(null);
  const [checking, setChecking] = useState(false);
  const [failure, setFailure] = useState<unknown>(null);

  useEffect(() => {
    let alive = true;
    void scanSetup()
      .then((found) => {
        if (alive) setScan(found);
      })
      .catch(() => {
        if (alive) setScan(null);
      })
      .finally(() => {
        if (alive) setScanning(false);
      });
    return () => {
      alive = false;
    };
  }, []);

  const adbPath = useDebouncedSave(settings?.connection.adb_path ?? "", (value) =>
    saveSettingAsync(
      patch,
      (document) => {
        document.connection.adb_path = value;
      },
      setFailure,
    ),
  );
  const token = useDebouncedSave(settings?.connection.brawl_api_token ?? "", (value) =>
    saveSettingAsync(
      patch,
      (document) => {
        document.connection.brawl_api_token = value;
      },
      setFailure,
    ),
  );

  /** The route reads what is on disk, so a field with something typed into it has nothing
   * to check yet. */
  const unsaved =
    adbPath.value !== (settings?.connection.adb_path ?? "") ||
    token.value !== (settings?.connection.brawl_api_token ?? "");

  /** The instance the no_tag sentence names, which is the first one with no tag on it. */
  const withoutTag = (settings?.instances ?? []).find((inst) => inst.player_tag === "");

  const runCheck = () => {
    setChecking(true);
    void getConnection()
      .then((answer) => {
        setCheck(answer);
        // The strip has a sentence for every status but ok, where it renders nothing at all,
        // so the one answer with nothing to fix is the one that needs a toast.
        if (answer.status === "ok") {
          toast("The Brawl Stars API accepted the token.", { tone: "ok" });
        }
      })
      .catch((error: unknown) => {
        setFailure(error);
      })
      .finally(() => {
        setChecking(false);
      });
  };

  return (
    <div>
      {failure !== null && <ErrorBlock error={failure} />}

      <SettingRow
        title="ADB path"
        description="brawlfarm needs HD-Adb.exe from the BlueStacks folder."
        error={fieldError(fieldErrors, "connection.adb_path")}
        layout="stacked"
      >
        <div className="space-y-2" onBlur={adbPath.onBlur}>
          <Field
            label="ADB path"
            id="connection-adb-path"
            value={adbPath.value}
            onChange={adbPath.onChange}
            width="full"
          />
          <div className="flex flex-wrap items-center gap-2">
            {scanning && (
              <>
                <Chip tone="idle">Scanning…</Chip>
                <span className="text-[12px] text-muted">Asking adb for devices</span>
              </>
            )}
            {!scanning && scan !== null && (
              <Chip tone={scan.adb_found ? "ok" : "bad"}>
                {scan.adb_found ? "Found" : "Not found"}
              </Chip>
            )}
            <Link
              to="/setup"
              className="inline-flex h-8 items-center rounded-[6px] border border-line bg-panel-2 px-3 text-[13px] font-medium text-text transition-colors duration-[120ms] hover:border-accent"
            >
              Run setup again
            </Link>
          </div>
          {!scanning && scan !== null && !scan.adb_found && (
            <p className="text-[12px] text-muted">
              Install BlueStacks, or type the path to HD-Adb.exe above.
            </p>
          )}
        </div>
      </SettingRow>

      <SettingRow
        title="Brawl Stars API token"
        description={TOKEN_LINE}
        error={fieldError(fieldErrors, "connection.brawl_api_token")}
      >
        <div onBlur={token.onBlur}>
          <Field
            label="Brawl Stars API token"
            id="connection-token"
            value={token.value}
            onChange={token.onChange}
            type="password"
            width="full"
            spellCheck={false}
            autoComplete="off"
          />
          <div className="mt-2 space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <Button
                variant="secondary"
                size="sm"
                disabled={unsaved}
                disabledReason="Save first"
                onClick={runCheck}
              >
                Check connection
              </Button>
              {checking && <Chip tone="idle">Checking…</Chip>}
            </div>
            <p className="text-[12px] text-muted">Uses the saved token and player tag.</p>
            {!checking && check !== null && (
              <ConnectionStrip
                status={check.status}
                instanceWithoutTag={withoutTag?.name ?? null}
              />
            )}
          </div>
        </div>
      </SettingRow>
    </div>
  );
}
