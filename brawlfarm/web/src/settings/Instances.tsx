/**
 * Settings > Instances: the fleet as config.toml holds it.
 *
 * The table joins two sources on the name. settings.instances is what is on disk and what
 * this screen edits; useInstances() is the supervisor's live view, which an instance it has
 * not derived yet simply does not have. That row is shown as stopped, because for the few
 * seconds between adding it by hand and the supervisor picking it up it is not running.
 *
 * The player tag is editable in place, on a 500 ms debounce and at once on blur, because a
 * tag is the one field people come here to fix and opening an edit row for it is three
 * clicks too many. Everything else goes through the inline form, and Add opens the same form
 * as a new last row.
 *
 * Removing an instance is not here: it is irreversible, so it sits with the other two
 * irreversible acts in the Danger zone of Settings, Data. This row edits and nothing else.
 *
 * A refusal is shown where it happened: a 409 is the API's own sentence in the row it names,
 * until the next save succeeds, and a 422 is already under its field through the mapper in
 * useSettingsPatch. Anything else is the section's ErrorBlock.
 */
import { useEffect, useRef, useState } from "react";

import { type SettingsPatch, fieldError } from "./useSettingsPatch";
import { ApiError } from "../api/client";
import { scanSetup } from "../api/setup";
import type { ScanInstance } from "../api/types";
import { useInstances } from "../api/useInstances";
import { Button } from "../components/ui/Button";
import { ErrorBlock } from "../components/ui/ErrorBlock";
import { Field } from "../components/ui/Field";
import { StateChip } from "../components/ui/StateChip";
import { type Column, Table } from "../components/ui/Table";
import { ELLIPSIS } from "../lib/copy";
import { plural } from "../lib/format";
import { knownState } from "../lib/states";
import { toast } from "../lib/toast";

/** The key of the row that does not exist yet. An instance name can never be empty. */
const NEW_ROW = "";
const DEBOUNCE_MS = 500;

interface Row {
  name: string;
}

interface Draft {
  name: string;
  adb_port: string;
  player_tag: string;
}

const BLANK: Draft = { name: "", adb_port: "", player_tag: "" };

export function Instances({ settingsPatch }: { settingsPatch: SettingsPatch }) {
  const { settings, patch, fieldErrors } = settingsPatch;
  const { data: views } = useInstances();
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft>(BLANK);
  const [rowErrors, setRowErrors] = useState<Record<string, string>>({});
  const [scanning, setScanning] = useState(false);
  const [failure, setFailure] = useState<unknown>(null);
  const [tagText, setTagText] = useState<Record<string, string>>({});
  const timers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  // A navigation away must not let a debounce fire a write into a screen that is gone.
  useEffect(() => {
    const pending = timers.current;
    return () => {
      for (const timer of Object.values(pending)) clearTimeout(timer);
    };
  }, []);

  const instances = settings?.instances ?? [];
  const rows: Row[] = [
    ...instances.map((inst) => ({ name: inst.name })),
    ...(editing === NEW_ROW ? [{ name: NEW_ROW }] : []),
  ];

  const succeeded = () => {
    setRowErrors({});
    setFailure(null);
  };

  const failed = (key: string, error: unknown) => {
    if (error instanceof ApiError && error.status === 422) return; // already under its field
    if (error instanceof ApiError && error.status === 409) {
      setRowErrors((errors) => ({ ...errors, [key]: error.detail }));
      return;
    }
    setFailure(error);
  };

  const saveTag = (name: string, value: string) => {
    delete timers.current[name];
    void patch((document) => {
      const row = document.instances.find((inst) => inst.name === name);
      if (row !== undefined) row.player_tag = value;
    })
      .then(() => {
        succeeded();
        // The model upper-cases the tag and puts the # back, so the box goes back to showing
        // what is on disk rather than what was typed into it.
        setTagText((text) => {
          const next = { ...text };
          delete next[name];
          return next;
        });
      })
      .catch((error: unknown) => failed(name, error));
  };

  const onTagChange = (name: string, next: string) => {
    setTagText((text) => ({ ...text, [name]: next }));
    const running = timers.current[name];
    if (running !== undefined) clearTimeout(running);
    timers.current[name] = setTimeout(() => saveTag(name, next), DEBOUNCE_MS);
  };

  const onTagBlur = (name: string) => {
    const running = timers.current[name];
    if (running === undefined) return; // nothing was typed, so there is nothing to flush
    clearTimeout(running);
    saveTag(name, tagText[name] ?? "");
  };

  const startEdit = (name: string) => {
    const inst = instances.find((row) => row.name === name);
    setEditing(name);
    setDraft(
      inst === undefined
        ? BLANK
        : { name: inst.name, adb_port: String(inst.adb_port), player_tag: inst.player_tag },
    );
  };

  const cancelEdit = () => {
    setEditing(null);
    setDraft(BLANK);
  };

  const saveRow = () => {
    const original = editing;
    if (original === null) return;
    const port = Number(draft.adb_port);
    void patch((document) => {
      const at = document.instances.findIndex((inst) => inst.name === original);
      const row = {
        // Fields this form does not show (the emulator kind) stay as they were on disk.
        ...(at < 0 ? {} : document.instances[at]),
        name: draft.name.trim(),
        // A blank or unparseable port goes as 0, which the API answers with a 422 naming
        // adb_port: the model is the validator here, not this form.
        adb_port: Number.isFinite(port) ? port : 0,
        player_tag: draft.player_tag.trim(),
      };
      if (at < 0) document.instances.push(row);
      else document.instances[at] = row;
    })
      .then(() => {
        succeeded();
        cancelEdit();
      })
      .catch((error: unknown) => failed(original, error));
  };

  const scanAgain = () => {
    setScanning(true);
    const known = new Set(instances.map((inst) => inst.name));
    void scanSetup()
      .then((scan) => {
        const found = scan.instances.filter(
          (row): row is ScanInstance & { adb_port: number } =>
            row.adb_port !== null && !known.has(row.name),
        );
        if (found.length === 0) {
          toast("No new instances found");
          return;
        }
        return patch((document) => {
          for (const row of found) {
            document.instances.push({ name: row.name, adb_port: row.adb_port, player_tag: "" });
          }
        }).then(() => {
          setRowErrors({});
          setFailure(null);
          toast(`Added ${plural(found.length, "instance")}`);
        });
      })
      .catch((error: unknown) => setFailure(error))
      .finally(() => setScanning(false));
  };

  const isEditing = (row: Row) => editing === row.name;
  /** The label is hidden in the cell, so it names the row rather than repeating the column:
   * three boxes called "Player tag" are three boxes a screen reader cannot tell apart. */
  const tagLabel = (name: string) => `Player tag for ${name}`;

  /** The row that does not exist yet is saved onto the end, so that is the index the API's
   * 422 names it by. */
  const rowIndex = (row: Row) =>
    row.name === NEW_ROW
      ? instances.length
      : instances.findIndex((inst) => inst.name === row.name);

  /** What the mapper put under this field, as the line that goes below its box. Every
   * column shows it, in the edit form and in the cell alike: a refusal the reader cannot
   * see is a save that looks like it did nothing. */
  const fieldNote = (row: Row, field: string) => {
    const message = fieldError(fieldErrors, `instances.${rowIndex(row)}.${field}`);
    return message === undefined ? null : (
      <p className="mt-1 font-sans text-[12px] whitespace-normal text-bad">{message}</p>
    );
  };

  const columns: readonly Column<Row>[] = [
    {
      key: "name",
      label: "Name",
      mono: true,
      render: (row) => (
        <div>
          {isEditing(row) ? (
            <span className="[&_label]:sr-only">
              <Field
                label="Name"
                id={`instance-name-${row.name}`}
                value={draft.name}
                onChange={(value) => setDraft((current) => ({ ...current, name: value }))}
                width="full"
              />
            </span>
          ) : (
            row.name
          )}
          {fieldNote(row, "name")}
          {/* The region is in the row before a refusal is: a live region that arrives with
              its own text is a region a screen reader never announces. */}
          <div aria-live="polite">
            {rowErrors[row.name] !== undefined && (
              <p className="mt-1 font-sans text-[12px] whitespace-normal text-bad">
                {rowErrors[row.name]}
              </p>
            )}
          </div>
        </div>
      ),
    },
    {
      key: "adb_port",
      label: "Port",
      title: "The adb port BlueStacks listens on",
      mono: true,
      width: "140px",
      render: (row) => (
        <div>
          {isEditing(row) ? (
            <span className="[&_label]:sr-only">
              <Field
                label="ADB port"
                id={`instance-port-${row.name}`}
                value={draft.adb_port}
                onChange={(value) => setDraft((current) => ({ ...current, adb_port: value }))}
                type="number"
                width="full"
              />
            </span>
          ) : (
            String(instances.find((inst) => inst.name === row.name)?.adb_port ?? "")
          )}
          {fieldNote(row, "adb_port")}
        </div>
      ),
    },
    {
      key: "player_tag",
      label: "Player tag",
      mono: true,
      width: "180px",
      render: (row) => {
        const stored = instances[rowIndex(row)]?.player_tag ?? "";
        const note = fieldNote(row, "player_tag");
        if (isEditing(row)) {
          return (
            <div>
              <span data-private className="[&_label]:sr-only">
                <Field
                  label="Player tag"
                  id={`instance-draft-tag-${row.name}`}
                  value={draft.player_tag}
                  onChange={(value) => setDraft((current) => ({ ...current, player_tag: value }))}
                  width="full"
                />
              </span>
              {note}
            </div>
          );
        }
        return (
          <div onBlur={() => onTagBlur(row.name)}>
            <span data-private className="[&_label]:sr-only">
              <Field
                label={tagLabel(row.name)}
                id={`instance-tag-${row.name}`}
                value={tagText[row.name] ?? stored}
                onChange={(value) => onTagChange(row.name, value)}
                placeholder={`#2P0YLQ9${ELLIPSIS}`}
                width="full"
              />
            </span>
            {note}
          </div>
        );
      },
    },
    {
      key: "folder",
      label: "Data folder",
      mono: true,
      render: (row) =>
        row.name === NEW_ROW ? null : (
          // Relative to the home directory: an absolute path here would name a Windows user.
          <span className="text-muted">{`instances/${row.name}`}</span>
        ),
    },
    {
      key: "status",
      label: "Status",
      render: (row) => {
        if (row.name === NEW_ROW) return null;
        const view = (views ?? []).find((entry) => entry.name === row.name);
        return <StateChip state={knownState(view?.state)} />;
      },
    },
    {
      key: "actions",
      label: "",
      render: (row) =>
        isEditing(row) ? (
          <div className="flex items-center gap-2">
            <Button variant="primary" size="sm" onClick={saveRow}>
              Save
            </Button>
            <Button variant="quiet" size="sm" onClick={cancelEdit}>
              Cancel
            </Button>
          </div>
        ) : (
          <Button variant="quiet" size="sm" onClick={() => startEdit(row.name)}>
            Edit
          </Button>
        ),
    },
  ];

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Button variant="primary" size="sm" onClick={() => startEdit(NEW_ROW)}>
          Add instance
        </Button>
        <Button
          variant="secondary"
          size="sm"
          disabled={scanning}
          disabledReason="Scanning…"
          onClick={scanAgain}
        >
          Scan again
        </Button>
      </div>

      <p className="text-[12px] text-muted">Scan again finds running BlueStacks instances.</p>

      {failure !== null && <ErrorBlock error={failure} />}

      <Table
        columns={columns}
        rows={rows}
        rowKey={(row) => row.name}
        empty="No instances yet. Add one or scan for BlueStacks."
      />
    </div>
  );
}
