# Shadow mode

Shadow mode runs the detector beside the farm loop while a match plays and logs what it sees. It sends no input: farm mode drives exactly as it does today.

## Turning it on

Set `shadow = true` under `[behavior]` in `config.toml` under `%LOCALAPPDATA%\brawlfarm`, then restart the instance. There is no switch in the panel yet.

It needs a model: `models\play.onnx` and `models\play.json` under `%LOCALAPPDATA%\brawlfarm`. Nothing else: `onnxruntime` is already a core dependency, so a plain install can run shadow mode. No model ships yet; without one, shadow mode says `model_missing` once in the feed and does nothing. With the `play-gpu` extra installed the detector runs on CUDA; see [install.md](install.md).

## What it writes

Shadow files land under `instances\<name>\shadow\`, one JSON lines file per match. Each row holds boxes, timings, HUD readings, rule calls and the farm's last move kind. Rule calls are logged only and never sent to the game. The newest 50 files are kept.

Three feed rows carry shadow mode: `play_on` when a session starts, `play_summary` once the match ends, and `play_fallback` when a session gives up, with a reason such as `model_missing`, `model_invalid`, `source_start`, `source_error`, `stale`, `detector_error` or `session_error`.

## Cost

Shadow mode captures the instance screen about 5 times a second. Measured against a run with it off on the same night, it costs the in-match farm loop about 8 percent at the median tick and about 13 percent at the ninety-fifth.
