# Setting up brawlfarm

## What you need

Windows 11, BlueStacks 5, Python 3.13 and [uv](https://docs.astral.sh/uv/). Set aside about 20 minutes. You do not need to have used adb before; the setup wizard finds it for you.

## Install BlueStacks and Brawl Stars

Download BlueStacks 5 from [bluestacks.com](https://www.bluestacks.com/) and install it. When it asks which Android version to create, pick a Pie 64-bit instance: that is the one brawlfarm is tested against.

Start the instance, open the Play Store inside it and install Brawl Stars. Log in to the account you want to farm. If the account is new, play through the tutorial to the end, because brawlfarm reads the normal main menu and the tutorial replaces it with a guided one.

## Enable Android Debug Bridge

brawlfarm talks to each instance over adb, which BlueStacks keeps switched off by default.

In BlueStacks, open Settings, then Advanced, and turn Android Debug Bridge on. BlueStacks shows a port number next to the switch; write it down, because that is the port you give brawlfarm for this instance.

Every instance has its own port. If you add instances later with the Multi-Instance Manager, each new one needs the switch turned on and has a different port.

## Set the display

In BlueStacks, open Settings, then Display. Set the resolution to 1600 x 900 and the pixel density to 240 DPI, then restart the instance so the change takes.

This is not a preference. Every template brawlfarm matches against was cut from a 1600 x 900 screen at 240 DPI, so a different size moves every button. The worker asserts the size when it starts and exits if it is wrong, rather than tapping at coordinates that no longer mean anything.

The wizard's Display step checks this for you: it asks each instance for its real size and density over adb and only opens Continue when every instance answers 1600 x 900 at 240.

## Install brawlfarm

Clone the repository and run it from the checkout:

```
git clone https://github.com/as9pa/brawlfarm.git
cd brawlfarm
uv sync
uv run brawlfarm
```

### From PyPI

Once the first release is on PyPI you can skip the clone. `uv tool install brawlfarm` puts a `brawlfarm` command on your PATH, and `uvx brawlfarm` runs it once without installing anything:

```
uv tool install brawlfarm
brawlfarm
```

The desktop window and tray icon are an extra, so install `uv tool install "brawlfarm[desktop]"` if you want `brawlfarm --window`. Everything below says `uv run brawlfarm` because it assumes the checkout; with a tool install, drop the `uv run` and type `brawlfarm`.

`uv run brawlfarm` supervises every configured instance, serves the panel on `http://127.0.0.1:8765/` and opens your browser on it a second later. It prints the same URL to the console. The flags you are most likely to want:

```
uv run brawlfarm --no-browser    # serve the panel without opening a browser
uv run brawlfarm --port 9000     # serve the panel somewhere else
uv run brawlfarm --once          # one supervisor tick, print the instances, exit
uv run brawlfarm --home <path>   # use a different data directory for this run
```

The panel binds 127.0.0.1 only and has no login, so anything that can reach it can drive your instances. Do not port-forward it and do not put it behind a reverse proxy.

If you would rather have a window than a browser tab, `uv sync --group desktop` in a checkout (or `uv tool install "brawlfarm[desktop]"` from PyPI) installs the optional desktop extras and `uv run brawlfarm --window` then opens the panel in its own window with a tray icon. Closing the window only hides it: the tray icon brings it back, and Quit there stops the run the way Ctrl+C does. Without the extras brawlfarm says so and opens the browser instead.

## Run the setup wizard

The wizard lives at `http://127.0.0.1:8765/setup`. With nothing configured yet, the Fleet page offers it behind an Open setup button, and once you are past it, Settings, Connection has a Run setup again link.

It has five steps, listed down the left as BlueStacks, Instances, Display, Stats and Done. Every step writes straight to `config.toml` as you go, so you can close the tab and pick it up later. You can press a step you have already passed to go back and check what you typed; a step you have not reached yet is disabled.

### BlueStacks

The step scans for adb the moment it opens and tells you what it found. While it is working it shows a Scanning chip and the line "Asking adb for devices", which can take a few seconds.

If it finds adb, it shows a Found chip with the path and writes that path to `connection.adb_path` for you, so there is nothing to press but Continue.

If it does not, you get a field labelled "Where is BlueStacks installed" and the note "brawlfarm needs HD-Adb.exe from the BlueStacks folder". Type the path to your BlueStacks folder and press Scan again. Continue opens once adb has been found.

### Instances

This step picks which BlueStacks instances brawlfarm farms. It lists what the scan found in a table with a checkbox, the instance Name, its BlueStacks Display name, its ADB port and a status.

Tick the instances you want to farm. Test on a row probes that one port and reports back on that row alone, so you can tell a working instance from a sleeping one before you commit to it. If the table says "No instances found. Start a BlueStacks instance, enable ADB, then Scan again", start the instance, check the adb switch and press Scan again.

BlueStacks does not always list every instance in its own config file. Add a port opens an ADB port field where you can type a port by hand; the row is named after the port, and that name becomes the instance's folder name under the data home.

Continue writes the instances you ticked to the `[[instances]]` tables in `config.toml`, each with its `name`, its `adb_port` and an empty `player_tag`.

### Display

One card per instance, each checking itself over adb. This is the only step that saves nothing; it is a gate, not a setting.

A card that fails prints the fix sentence the API sends back: "Set the display to 1600 x 900 and pixel density 240 in BlueStacks: Settings, Display, then restart the instance." Change it in BlueStacks, restart that instance, then press Recheck on the card. Back goes to Instances. Continue only opens when every instance passes.

### Stats

Everything on this step is optional. Nothing here is needed to farm; it only buys you per-game numbers on the Stats page, which come from the official Brawl Stars API rather than from the screen.

There is a masked field labelled "Brawl Stars API token" and one field per instance for that instance's player tag, each labelled with the instance name and showing `#TAG` as a placeholder. Both save as you type: the token goes to `connection.brawl_api_token` and each tag to that instance's `player_tag`. The tag is upper-cased and given its `#` back for you.

Back returns to Display, Skip for now marks the step finished and moves on, and Continue does the same once you have filled in what you want.

### Done

A four-line summary of what setup did: how many instances and their names, the adb path, whether the token is set or skipped, and how many player tags are filled in. The token itself is never printed here, only the words "Token: set" or "Token: skipped".

Below it is a "Start <instance> now" switch for the first instance, on by default, and an Open Fleet button. Open Fleet takes you to the panel, starting that instance on the way if the switch is on.

## The Brawl Stars API token

The token is what lets brawlfarm read your battle log, so the Stats page can show per-game trophy changes rather than only what it saw on screen. Farming works without it.

Get one at [developer.brawlstars.com](https://developer.brawlstars.com/): create an account, then create a key. The key is locked to the IP address you enter when you create it, so it stops working when your public IP changes and you need a new key. Your player tag is the one under your name in the game, for example `#2P0YLQ9`; with or without the hash is fine.

The token lives in `%LOCALAPPDATA%\brawlfarm\config.toml` as `brawl_api_token` under `[connection]`. It is never shown in the panel and never written to the logs, so a screenshot of the panel or a log file you paste somewhere does not leak it.

When the token cannot be used, the Stats page says why in one quiet line above the numbers:

- No token set: "Battle log unavailable. Add a Brawl Stars API token in Settings to see per-game stats."
- An instance has no player tag: "Add a player tag for <instance> in Settings, Instances to see its games."
- The token was refused: "The Brawl Stars API rejected the token. Check the token, and the IP address it was created for, in Settings, Connection." This is what you see after your IP changes.
- The API did not answer: "The Brawl Stars API did not answer. Stats show what was logged so far."

In every one of these cases the rest of the page still shows everything brawlfarm logged itself, and farming carries on.

## Where files live

Everything brawlfarm writes lives under one data home, `%LOCALAPPDATA%\brawlfarm`. Set `BRAWLFARM_HOME` to point it somewhere else, or pass `--home` for one run.

- `config.toml`, the whole configuration, which the panel and the wizard both edit.
- `instances/<name>/` for each instance: `status.json` (what the supervisor reads to find a running worker, including its PID), `games.csv` (one row per finished match), `session-*.jsonl` (the narration behind the activity feed, one file per session) and `farmplan.json` (the roster, the queue and the current brawler).
- `cache/brawlers/`, the brawler portraits, each fetched once from the Brawlify CDN and kept.
- `logs/`, the supervisor's own log.

## First farming session

Leave the "Start <instance> now" switch on at the end of the wizard and press Open Fleet, or press Start all at the top of the Fleet page.

The card goes to Starting while the worker comes up and checks the display, then to Farming once it is queuing. Underneath the state it names the phase it is in: at menu, queuing, playing or returning. The activity feed on the instance page says the same thing in plain sentences, one line per match and per interruption.

The anti-ban scheduler is on by default. It draws a human-shaped set of sessions for the day, with breaks between them, and the worker sits in Scheduled break between them instead of farming around the clock. You can see today's sessions on the instance page and turn the scheduler off, or override it for the day, in Settings.

Stop on the card finishes the current match first: the card reads "Stopping after this match" until that match ends, then Stopped. Stop all at the top of the page does the same for every instance, and Restart on a card stops the worker and brings it back on the next tick.

Closing brawlfarm itself does not stop the workers. The next start reattaches to them through their status files.

## When something looks wrong

| Symptom | What the panel says | What to do |
| --- | --- | --- |
| The card is red and nothing happens | Offline | The instance is not answering on its adb port. Start the BlueStacks instance, check Settings, Advanced, Android Debug Bridge is still on, then press Retry now on the card. |
| The worker starts and quits at once | "Wrong resolution: 1280 x 720, need 1600 x 900" in the feed | Set that instance to 1600 x 900 at 240 DPI in BlueStacks, Settings, Display, restart the instance and start it again. |
| It queued into the wrong game mode | "Wrong mode detected", or "Wrong mode detected, switched back" | Nothing, if it says it switched back. If it keeps happening, the mode banner has moved and the instance needs recalibrating. |
| Queuing times out over and over after a game update | "Recalibration needed: <surface>" | A game update moved something brawlfarm matches against. There is no recalibration page yet; it is phase 8. Until then, stop the instance and follow the repository for an updated set of templates. |
| The Stats page has no per-game numbers | "The Brawl Stars API rejected the token. Check the token, and the IP address it was created for, in Settings, Connection." | Your public IP has probably changed. Create a new key at developer.brawlstars.com for your current IP and paste it into Settings, Connection. |
| The instance stopped after an emulator restart | "Emulator restart failed; stopped until you start it again" | BlueStacks did not close or did not boot in time. Check the instance opens by hand and `emulator_path` is right, then press Start. See [Emulator restarts](install.md#emulator-restarts). |
| A match crashed | "Crash: <error>" in the feed | The worker recovers by itself and says so in the feed. A crash that repeats is worth reporting with the session file from `instances/<name>/`. |
