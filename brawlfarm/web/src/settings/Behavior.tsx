/**
 * Settings > Behavior: how a worker plays.
 *
 * Thirteen switches over two sections of the settings document, six of them plain and seven
 * behind Advanced, which starts collapsed because they are performance and safety switches
 * that are on for a reason. The schedule new instances start with sits under the plain six,
 * because a schedule is a default you set once and then forget about.
 *
 * Each row is built by one of the two helpers below rather than by a literal table, so the
 * key is checked against the settings type at compile time and the dotted loc the API sends
 * a 422 under is derived from it instead of typed out twice. The schedule row is written out
 * instead, because it writes scheduler.default_enabled rather than a behavior key.
 *
 * An advanced row also says what turning it off costs, and carries a Changed tag when it is
 * not at its model default. The defaults come from the API the first time Advanced opens
 * rather than from a copy kept here, so one pydantic model stays the only place they live.
 */
import { type ReactNode, useRef, useState } from "react";

import { SettingRow } from "./SettingRow";
import {
  type SettingsPatch,
  fieldError,
  saveSetting,
  saveSettingAsync,
} from "./useSettingsPatch";
import { getSettingsDefaults } from "../api/settings";
import type { AppSettings } from "../api/types";
import { Button } from "../components/ui/Button";
import { Chip } from "../components/ui/Chip";
import { ErrorBlock } from "../components/ui/ErrorBlock";
import { Switch } from "../components/ui/Switch";
import { toast } from "../lib/toast";

interface Toggle {
  /** The API's dotted loc, which is also the key a 422 for this row arrives under. */
  loc: string;
  title: string;
  description: string;
  /** What turning an advanced switch off costs. Undefined on a basic row, which is also
   * what says a row carries no Changed tag. */
  off?: string;
  read: (settings: AppSettings) => boolean;
  write: (settings: AppSettings, next: boolean) => void;
}

/** The behavior keys a switch can hold: the section also has number and string settings. */
type BehaviorSwitch = {
  [K in keyof AppSettings["behavior"]]: AppSettings["behavior"][K] extends boolean ? K : never;
}[keyof AppSettings["behavior"]];

function behavior<K extends BehaviorSwitch>(
  key: K,
  title: string,
  description: string,
): Toggle {
  return {
    loc: `behavior.${key}`,
    title,
    description,
    read: (settings) => settings.behavior[key],
    write: (settings, next) => {
      settings.behavior[key] = next;
    },
  };
}

function advanced<K extends keyof AppSettings["advanced"]>(
  key: K,
  title: string,
  description: string,
  off: string,
): Toggle {
  return {
    loc: `advanced.${key}`,
    title,
    description,
    off,
    read: (settings) => settings.advanced[key],
    write: (settings, next) => {
      settings.advanced[key] = next;
    },
  };
}

const BASIC: readonly Toggle[] = [
  behavior(
    "winrate_aware",
    "Prefer brawlers that win",
    "Picks the brawler with the best win rate in the current step.",
  ),
  behavior(
    "opportunity_cost",
    "Skip far-off tiers",
    "Skips brawlers whose next tier is more than a session away.",
  ),
  behavior("gas_aware", "Leave the gas early", "Moves away from the gas one ring sooner."),
  behavior("bush_hide", "Hide in bushes", "Hides in bushes when the map allows."),
  behavior(
    "close_game_on_stop",
    "Close the game on stop",
    "Closes Brawl Stars when the instance stops.",
  ),
  behavior(
    "dnd_at_start",
    "Do Not Disturb on start",
    "Turns on Do Not Disturb when the instance starts.",
  ),
];

/** What the schedule row promises, taken from brawlfarm/core/scheduler.py: the session and
 * break ranges it draws from, the longer outing, the start window and the daily total. */
const SCHEDULE_DESCRIPTION =
  "Sessions run 30 minutes to 2 hours, sometimes up to 2 and a half, with breaks of 45 " +
  "minutes to 2 hours between them and one or two longer outings of 2 to 4 hours. Play " +
  "starts within 2 hours of midnight and adds up to about 9 hours a day. Change the hours " +
  "for one instance on its own page.";

const ADVANCED: readonly Toggle[] = [
  advanced(
    "fast_input",
    "Faster taps",
    "Send taps through the faster adb path.",
    "Off: taps go through the slower path.",
  ),
  advanced(
    "raw_cap",
    "Raw frames",
    "Read frames without re-encoding them.",
    "Off: frames are re-encoded before they are read.",
  ),
  advanced(
    "gray_match",
    "Grayscale matching",
    "Match templates in grayscale.",
    "Off: templates match in color.",
  ),
  advanced(
    "phase_classify",
    "Match phase detection",
    "Work out the match phase from the screen.",
    "Off: the match phase is worked out from timing instead of the screen.",
  ),
  advanced(
    "ability_buttons",
    "Use abilities",
    "Use the gadget and super buttons.",
    "Off: the gadget and super buttons are left alone.",
  ),
  advanced(
    "recalib_tripwire",
    "Warn when the season changed the screens",
    "Warn when the screens stop matching what brawlfarm expects.",
    "Off: no warning when a season changes the screens.",
  ),
  advanced(
    "dnd_off_on_stop",
    "Do Not Disturb off on stop",
    "Turn Do Not Disturb back off when the instance stops.",
    "Off: Do Not Disturb stays on after the instance stops.",
  ),
];

/** The id the disclosure button's aria-controls names. */
const ADVANCED_BLOCK = "settings-advanced";

/** One row's description: the sentence saying what the switch does, and on an advanced row
 * the second line saying what off costs. The off line is its own element so each line is
 * one readable sentence rather than the two running together. */
function describe(row: Toggle): ReactNode {
  if (row.off === undefined) return row.description;
  return (
    <>
      {row.description}
      <br />
      <span>{row.off}</span>
    </>
  );
}

export function Behavior({ settingsPatch }: { settingsPatch: SettingsPatch }) {
  const { settings, patch, fieldErrors } = settingsPatch;
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [failure, setFailure] = useState<unknown>(null);
  /** Every section at its model default, fetched the first time Advanced opens. Null until
   * it lands, and still null if it never does. */
  const [defaults, setDefaults] = useState<AppSettings | null>(null);
  const askedForDefaults = useRef(false);

  if (settings === undefined) return null;

  const toggleAdvanced = (): void => {
    const opening = !showAdvanced;
    setShowAdvanced(opening);
    if (!opening || askedForDefaults.current) return;
    askedForDefaults.current = true;
    void getSettingsDefaults().then(setDefaults, () => {
      // Silent on purpose: the Changed tags and the Reset link are all this document is
      // for, and neither is load bearing enough to put an error on the screen for.
    });
  };

  const differs = (row: Toggle): boolean =>
    defaults !== null && row.read(defaults) !== row.read(settings);

  /** Every advanced key back to its default in one write, with the values it had before as
   * the undo. Basic keys are left alone: this link is the Advanced block's, not the page's.
   *
   * This path has a note of its own where a plain field save has none, because the note is
   * what carries the way back. */
  const resetAdvanced = (): void => {
    if (defaults === null) return;
    const before = structuredClone(settings.advanced);
    const wanted = structuredClone(defaults.advanced);
    void saveSettingAsync(
      patch,
      (document) => {
        document.advanced = wanted;
      },
      setFailure,
    )
      .then(() => {
        toast("Advanced switches back to defaults", {
          undo: () =>
            saveSettingAsync(
              patch,
              (document) => {
                document.advanced = before;
              },
              setFailure,
            ),
        });
      })
      .catch(() => undefined);
  };

  const rows = (list: readonly Toggle[], tagged: boolean) =>
    list.map((row) => (
      <SettingRow
        key={row.loc}
        title={
          tagged && differs(row) ? (
            <span className="inline-flex flex-wrap items-center gap-2">
              {row.title}
              <Chip tone="idle">Changed</Chip>
            </span>
          ) : (
            row.title
          )
        }
        description={describe(row)}
        error={fieldError(fieldErrors, row.loc)}
      >
        <Switch
          checked={row.read(settings)}
          onChange={(next) =>
            saveSetting(patch, (document) => row.write(document, next), setFailure)
          }
          label={row.title}
        />
      </SettingRow>
    ));

  return (
    <div>
      {failure !== null && <ErrorBlock error={failure} />}
      <div>
        {rows(BASIC, false)}
        <SettingRow
          title="Schedule"
          description={SCHEDULE_DESCRIPTION}
          error={fieldError(fieldErrors, "scheduler.default_enabled")}
        >
          <Switch
            checked={settings.scheduler.default_enabled}
            onChange={(next) =>
              saveSetting(
                patch,
                (document) => {
                  document.scheduler.default_enabled = next;
                },
                setFailure,
              )
            }
            label="Schedule on by default"
          />
        </SettingRow>
      </div>

      <div className="mt-4">
        <div className="flex items-center gap-2">
          <h3 className="text-[13px] font-semibold">Advanced</h3>
          <Button
            variant="quiet"
            size="sm"
            onClick={toggleAdvanced}
            aria-expanded={showAdvanced}
            aria-controls={ADVANCED_BLOCK}
          >
            {showAdvanced ? "Hide" : "Show"}
          </Button>
        </div>
        {showAdvanced && (
          <div id={ADVANCED_BLOCK} className="mt-2">
            {rows(ADVANCED, true)}
            {defaults !== null && (
              <div className="mt-2">
                <Button
                  variant="quiet"
                  size="sm"
                  onClick={resetAdvanced}
                  disabled={!ADVANCED.some(differs)}
                  disabledReason="Already at the defaults"
                >
                  Reset to defaults
                </Button>
              </div>
            )}
          </div>
        )}
      </div>

      <p className="mt-4 text-[12px] text-muted">Applies the next time an instance starts.</p>
    </div>
  );
}
