# Install and run

The full requirements, every install option and every command line flag. [setup.md](setup.md) is the step by step walkthrough and [release.md](release.md) is how a release ships.

## Requirements

### Platform

- Windows 11.
- BlueStacks 5 (the nxt line) with Android Debug Bridge enabled.
- Each instance's display at 1600 x 900 and pixel density 240.
- Python 3.13 and [uv](https://docs.astral.sh/uv/).

[platform-support.md](platform-support.md) explains why it is only this platform today.

### Farm mode

The reference setup gives each BlueStacks instance 2 CPUs and 2048 MB of RAM, and runs one Python worker per instance. The reference machine is an 8-core Ryzen 7 7800X3D with 32 GB of RAM, and it runs one instance while the panel and the worker are on.

The panel and the worker are light next to BlueStacks itself. Each extra instance means one more BlueStacks instance of that size plus one more worker. For the host baseline, go by BlueStacks' own system requirements.

### Play extras

`brawlfarm[play]` adds PyAV, which decodes the scrcpy stream and the recordings. You do not install scrcpy yourself: the scrcpy 4.1 server ships inside the package and is pushed onto the instance over adb when a stream starts. Measured on the reference machine:

- A 1600 x 900 H.264 stream at 30 frames a second is about 4.3 Mbit/s.
- Decoding costs about 2 ms a frame on the host.
- The BlueStacks player process goes from about 22 percent of one core idle to about 81 percent of one core while streaming.

Shadow mode takes adb captures instead of the stream; its cost to the farm loop is in [shadow-mode.md](shadow-mode.md).

`brawlfarm[play-gpu]` needs an NVIDIA GPU with the display driver installed. The CUDA and cuDNN runtime wheels come with the extra and are well over a gigabyte. The reference GPU is an RTX 4080 SUPER 16 GB. No model ships yet, so nothing exercises it until you supply one. The Install section below covers the one catch with install order.

## Install

Clone the repository and `uv sync`, or install it from PyPI as a tool:

```
uv tool install brawlfarm             # a brawlfarm command on your PATH
uv tool install "brawlfarm[desktop]"  # the same, plus --window and the tray icon
uv tool install "brawlfarm[play]"     # the same, plus the play stream (PyAV)
uv tool install "brawlfarm[play-gpu]" # the same, plus onnxruntime on CUDA for the detector
uvx brawlfarm                         # run it once without installing it
```

The `play-gpu` extra downloads NVIDIA's CUDA and cuDNN runtime wheels (well over a gigabyte) and needs only the NVIDIA display driver on the machine. Shadow mode's detector uses it when it is present: it picks the CUDA execution provider then, and the CPU one otherwise. No model ships yet, so nothing exercises it until you supply one. One known catch: `onnxruntime` (which the text reader needs) and `onnxruntime-gpu` install into the same folder, and which one wins depends on install order. To check, run `python -c "import onnxruntime as o; print(o.get_available_providers())"` in the environment; if `CUDAExecutionProvider` is missing, run `uv pip install --reinstall-package onnxruntime-gpu onnxruntime-gpu` there. The GPU package also carries the CPU provider, so the text reader keeps working.

The commands below assume the checkout and say `uv run brawlfarm`; with a tool install the command is just `brawlfarm`.

## Setup

Start brawlfarm and open `http://127.0.0.1:8765/setup`. The wizard finds adb, lists your BlueStacks instances and their ports, checks each one is 1600 x 900 at pixel density 240, and optionally takes a Brawl Stars API token and your player tags. Every step writes straight to `config.toml`, so you can close it and come back. With nothing configured yet, the Fleet page offers the same wizard behind an Open setup button, and Settings, Connection has a Run setup again link once you are past it.

To change the display: BlueStacks, Settings, Display, set 1600 x 900 and pixel density 240, then restart the instance. The step by step walkthrough is [setup.md](setup.md).

## Running

```
uv run brawlfarm                 # supervise every instance and open the panel
uv run brawlfarm --no-browser    # same, without opening a browser
uv run brawlfarm --port 9000     # serve the panel somewhere else
uv run brawlfarm --once          # one supervisor tick, print the instances, exit
uv run brawlfarm --window        # a desktop window and a tray icon instead of the browser
```

`--window` needs the optional desktop extras, which `uv sync --group desktop` installs in a checkout and `uv tool install "brawlfarm[desktop]"` installs from PyPI; without them brawlfarm says so and opens the browser as usual. Closing the window only hides it to the tray icon, whose menu has Open panel and Quit.

The play extras add PyAV, which decodes the scrcpy play stream and old `match-N.h264` recordings (see [play-training.md](play-training.md)). In observe mode, with the recorder on, each match is also saved as a `match-N/` folder of screen captures in the session folder; that needs no extra. `uv sync --group play` installs it in a checkout.

Turning recordings and video into a labelled training set for the play mode detector is its own kit under `tools/play/`; see [play-training.md](play-training.md).

## Settings

Settings live in `%LOCALAPPDATA%\brawlfarm\config.toml` (override the folder with `BRAWLFARM_HOME`). Add one `[[instances]]` table per BlueStacks instance with its `name` and `adb_port`; each instance's files live under `instances/<name>/`. Stopping the process leaves workers running; the next start reattaches to them through their status files.

### Emulator restarts

BlueStacks' HD-Player leaks memory under the farm, so the supervisor can restart it between matches. Two settings under `[behavior]`, both off by default (0):

- `emulator_recycle_gb`: restart when HD-Player's private memory reaches this many GB.
- `commit_floor_gb`: restart when free commit charge drops below this many GB, but only if HD-Player itself holds at least 8 GB, because a restart cannot help when other programs hold the memory.

The check is commit and not free physical memory because commit is what runs out. Windows refuses allocations once the commit limit is reached, and that has happened with about 5.6 GB of physical memory still free, so a physical memory check would never have seen it.

The checks run only while a worker is running and the instance is wanted running. A restart lets the worker finish its match, closes HD-Player, reopens it from `emulator_path`, waits up to 6 minutes for Android to boot, then 2 minutes more before the worker starts. While the worker finishes its match the panel shows `stopping`, and a worker still going 6 minutes after the ask is killed. After that the instance reads as `scheduled_break`, reason `emulator_recycle`. An instance restarts at most once per 30 minutes. If a step fails you get one alert, titled Emulator restart failed, and the instance stays stopped, reason `emulator_recycle_failed`, until you press Start. Stopping an instance while the worker is still finishing its match abandons the restart and nothing is closed. From the close onward the restart finishes, then the instance stays stopped.

Closing brawlfarm mid-restart is safe: progress is kept in `recycle.json` in the instance's folder, and the next start resumes it, reopening the emulator if it is still closed. If the instance is set to stop at that start, or the job is more than 8 minutes old (after a power cut, say), the restart and its `recycle.json` are dropped instead and the emulator stays as it is. A failed restart stays failed across that, and the 30-minute limit holds too (`recycle_last.json`). Removing an instance from the settings drops its restart and both files, so an emulator that was closed at that moment stays closed until you open it.

The panel has no field for these yet. Set them in `config.toml` or through `PUT /api/settings`.
