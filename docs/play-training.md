# Training the play mode detector

## What this kit is

Tools under `tools/play/` that turn video of Brawl Stars into a labelled training set, and
turn a labelled set into a scored detector. A **class** is one kind of thing the detector
learns to find, such as `enemy` or `play_button`. A **box** is the rectangle drawn around one
instance of a class in one frame.

## What this kit is not

No model ships with this pull request. Running every step below leaves you with a folder of
data and, if you go all the way to training, a `play.onnx` file on your own disk. The first
model the project ships arrives later, in its own calibration pull request, together with the
training set hash, the score report, and the held-out numbers.

## Where the data lives

Nothing this kit touches enters the repository: no video, no frame, no label, no model. Every
tool takes `--root`, and without it the default is `%LOCALAPPDATA%\brawlfarm\datasets\play` (or
under `BRAWLFARM_HOME` if you have that set). The layout under the root:

```
videos/<source>.mp4              downloads kept for re-extraction
frames/<source>/<source>-NNNNNN.jpg   1600 x 900 JPEG frames
index.jsonl                      one line per kept frame
labelstudio/config.xml           the labelling interface
labelstudio/tasks.json           the import file, with template pre-label boxes
labelstudio/autolabel.json       the import file the auto-labeller writes, boxes as predictions
autolabel/                       the auto-labeller's work folder: checkpoint and SAM files
exclusions.json                  per-source facecam rectangles the auto-labeller skips
coco/{train,valid,test}/         the training set, split by source
runs/<YYYYmmdd-HHMMSS>/          training output
models/play.onnx, models/play.json    export output
label-pack/, label-pack.zip      a pack for labelling on another computer
```

## Step 1: frames from your own recordings

If you have run brawlfarm with the recorder on, you already have sessions under
`%LOCALAPPDATA%\brawlfarm\calibration\recordings\<instance>\<session>`. Only frames recorded
at full size (1600 x 900) are usable; a session recorded at half size has nothing to give here.

```
uv run python tools/play/frames.py --session "$env:LOCALAPPDATA\brawlfarm\calibration\recordings\Pie64\20260918-101112"
```

In observe mode each match the classifier sees is also saved in the session folder as a
`match-N/` folder: the emulator's own screen captures as full-size JPEGs (`0000.jpg`,
`0001.jpg`, ...), at most 5 a second, plus a `frames.jsonl` with one line per frame (`i`, the
frame number; `t`, the wall-clock time it was written; `age`, how old the capture was then, in
seconds; `state`, what the classifier saw). Recording a match needs no extra. `--session`
picks up every `match-N/` folder beside the screenshots and ingests each as its own source,
worth having because they cover in-match play that the menu frames do not.

Sessions recorded before this change hold `match-N.h264` clips instead. `--session` still
picks those up, and one clip can be extracted on its own (this path needs the play extra):

```
uv run python tools/play/frames.py --match match-1.h264 --source pie64-match-1
```

`frames.py` also takes `--video` for any container video, `--source` to name the source
(default is the file or folder name), and `--again` to add a source that is already in the
index. Extraction samples 2 frames a second and throws away near-duplicates, so a long session
keeps far fewer frames than it has seconds: a 181 frame menu session kept 138.

## Step 2: frames from YouTube

Put one URL per line in a text file, `#` for a comment line. Only
`https://www.youtube.com/watch?v=<id>` and `https://youtu.be/<id>` are accepted, and only the
11 character video id ever reaches the downloader or a file name.

```
uv run python tools/play/ingest_youtube.py urls.txt
```

`--root` picks the dataset root and `--keep-going` carries on after a failed download instead
of stopping the whole list. The video list is your own choice: pick mobile gameplay with a
clean, uncovered HUD, because anything covering the joystick, the cards or the buttons is a bad
example for the detector and for the HUD reader that comes later. The videos are downloaded
into `videos/` and stay on your disk; two 27 minute videos gave 2677 kept frames from 877 MB of
video, in about 8 minutes. Some creators bake black borders into their video; those are
trimmed per source before frames are cut, and the trim is recorded in `index.jsonl` so it can
be reversed.

By default the ingest also drops any frame where the mobile touch controls are not visible: a
frame with no HUD on it carries no action to learn from, so it is not worth the disk space. The
per video line reports how many frames were dropped this way, for example `seen 1200, kept 950,
duplicates 200, no HUD 50`. If a video gives up nothing at all, you will see
`<source>: no frame showed a mobile HUD; check that it is mobile Showdown play with a clean HUD`,
which almost always means the footage is not mobile gameplay, or something is covering the
controls for the whole video. Pass `--keep-no-hud` to keep every frame anyway, which is mostly
useful for checking the filter itself rather than for building a training set.

## Step 3: read the actions

Once a YouTube source is in the index, `actions.py` reads what the player was doing on every
frame that shows the HUD: where the joystick was pushed, where the shot was aimed, and what
state each button was in.

```
uv run python tools/play/actions.py
```

`--source` labels only the named source, and can be repeated for more than one; without it,
every YouTube source already in the index is labelled. `--root` picks the dataset root. For each
source this writes `frames/<source>/actions.jsonl`, one line per frame, and
`frames/<source>/actions.meta.json` beside it, recording what the source was learned to look
like.

Each line of `actions.jsonl` has:

- `file`: the frame's path.
- `t`: the frame's timestamp, when one is known.
- `hud`: whether the mobile HUD was visible on this frame.
- `move`: the joystick's push, as an `(x, y)` pair from -1 to 1 in each direction, or `null`
  when it could not be read. `x` grows to the right, `y` grows downward.
- `aim`: how far the attack stick has been dragged from its resting position, the same shape as
  `move`.
- `aiming`: whether that drag is large enough to count as a deliberate aim rather than noise
  around the resting position.
- `super`, `gadget`, `hyper`: the state of each button, or `null` when the button is not on
  screen at all. States are colour names rather than "on" or "off", because what a colour means
  for a given brawler is a question for whatever later reads this file, not for the HUD reader:
  `super` is `blue`, `gold`, `white` or `dark`; `gadget` is `green` or `grey`; `hyper` is
  `purple`.

There is no plain "attack was tapped" flag. Frames are only 0.5 seconds apart, and a tap that
lands and releases in between is never caught in a still picture, so aiming is the only signal
this reader has for the attack button.

A channel can be rejected for a whole source, most often because a creator has put their own
overlay over the joystick or the attack stick; when that happens `move` or `aim` is `null` on
every row for that source, the reason is printed, and it is recorded in `actions.meta.json`
rather than filling the file with guesses. Even on a clean source, individual frames can still
give a `null` move: the joystick's push is read from a small dot at the centre of its ring, and
when the stick is resting near the middle the knob sitting on top of it hides that dot. That is
exactly the case where the move label matters least, but it is worth knowing that a string of
`null` move values on their own does not mean the reader has failed.

Nothing reads `actions.jsonl` yet. It exists for the cloned policy proposed in release 1.3.0,
which trains on it; this step only produces the file.

## Step 4: auto-labels

Nobody draws boxes by hand. The auto-labeller reads every frame in the index and writes a
Label Studio import file whose predictions are the labels:

    uv run python tools/play/autolabel.py

It works in three passes. The first runs in the project environment: the template matcher
boxes the menu buttons and cards, colour masks box `bush` and `power_cube`, and OCR finds the
health number above each brawler, whose bar colour, ammo bar and ground ring decide between
`self`, `teammate`, `enemy` and `box`. The second hands those bars to SAM 2.1
(`facebook/sam2.1-hiera-large`, Apache 2.0) in `tools/play/segment.py`, a script with its own
torch environment like `train.py`; `uv run` builds it on first use and downloads the model
from Hugging Face (several GB on the first run). The third merges everything into
`labelstudio/autolabel.json` under the dataset root, next to `labelstudio/config.xml`.

`--root` picks the dataset root, `--source` restricts the run to one source (repeat the flag
for more), `--no-sam` skips the SAM pass and boxes every brawler with a fixed-size box under its
bar (quick and needs no GPU, but the boxes are looser), and `--model` picks another SAM
checkpoint. On an RTX 4080 SUPER the SAM pass alone takes about 0.4 s a frame once the model has
loaded. The first pass also runs OCR on the CPU, and about 0.9 s a frame overall was measured in
the spike; a 6-frame run took 21 s including the model load.

A full run takes well over an hour, so it keeps its progress in the work folder `autolabel/`
under the dataset root: the first pass appends every finished frame to `pass1.jsonl`, and the
SAM pass appends every finished frame to `boxes.jsonl`. A stopped run is continued with
`--resume` (plus the same `--root`, `--source` and other flags): finished frames are not done
again, and the summary says how many came from the checkpoint and how many SAM frames were
already done. Without `--resume` the run starts over and drops both files first. Checkpoint
lines written by another auto-labeller version are redone, never mixed in.

Nothing is labelled inside the HUD: the counters and player cards at the top left, the attack,
super and gadget buttons at the bottom right, and the joystick corner. A box that falls mostly
inside one of them is dropped. A streamer's facecam is the same problem in a place that changes
per video, so it goes in `exclusions.json` at the dataset root: one entry per source, each a list
of `[x0, y0, x1, y1]` rectangles in 1600 x 900 frame pixels, end exclusive.

    {"yt-AAAAAAAAAAA": [[0, 370, 290, 900]]}

A new source with a facecam is one more line in that file: open one of its frames, read the
facecam's corners off it, and add the rectangle before running the auto-labeller.
`--exclusions` points at another file.

What it does not label: `skull_star`, `team_up_panel` and `event_tab` have neither a template
nor a colour rule, so no frame carries them yet. Measured on frames it had never seen, `self`
came out right 11 times in 12, `enemy` about 6 in 8, `box` and `bush` (within 3 px) reliably,
and about half the power cubes were found; brawler boxes run 5 to 15 px loose.

### Template pre-labels only

The template part also runs on its own, which is useful for a quick look without waiting for the
rest. The farm's own template matcher already knows how to find a handful of screens and draws
a first draft of boxes for them.

```
uv run python tools/play/prelabel.py
```

`--root` picks the dataset root, and `--source` restricts pre-labelling to one source (repeat
the flag for more). Pre-labelling covers only `showdown_card`, `play_button`,
`play_again_button`, `proceed_button`, `exit_button` and `close_x`. Everything that happens
inside a match (`self`, `enemy`, `teammate`, `power_cube`, `box`, `bush`) and `skull_star`,
`team_up_panel`, `event_tab` have no template behind them. Pre-labelling
2815 frames took about 9.5 minutes.

## Step 5: checking labels in Label Studio (optional)

Label Studio is only needed to look at the auto-labels or correct a few by hand. Import
`labelstudio/autolabel.json` from the dataset root; the rest of this step works the same.

[Label Studio](https://labelstud.io/) is the tool you draw boxes in. Version 1.23.0 is what
this kit was checked against, on Python 3.12 (Python 3.13 has not been tried). Start it with
the local file server turned on and pointed at the dataset root, PowerShell:

```
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "$env:LOCALAPPDATA\brawlfarm\datasets\play"
uvx --python 3.12 label-studio start
```

Then, in the browser window Label Studio opens:

1. Create a project.
2. Settings, Labeling Interface, Code: paste the contents of `labelstudio/config.xml`.
3. Settings, Cloud Storage, Add Source Storage, Local files: set the absolute local path to
   the dataset root's `frames` folder. Do not press Sync; the tasks come from the import file
   in the next step, and this storage only lets Label Studio serve the images.
4. Import: `labelstudio/autolabel.json` (or `labelstudio/tasks.json` for the template
   pre-labels only). The import screen shows how many predictions it found; opening a task
   shows a frame with its auto-label boxes already drawn, where it found any.

What each class means:

- `self`: your own brawler.
- `enemy`: an opposing brawler.
- `teammate`: a brawler on your team, not yourself.
- `power_cube`: a power cube on the ground.
- `box`: a breakable box.
- `bush`: a bush a brawler can hide in.
- `showdown_card`: the card shown at the start or end of a Showdown match.
- `skull_star`: the star marking the last standing player in Showdown.
- `team_up_panel`: the panel offering a team-up in Showdown.
- `event_tab`: the tab for the current event on the home screen.
- `play_button`: the main Play button.
- `play_again_button`: the Play Again button on a results screen.
- `proceed_button`: the Proceed button, such as after a reward screen.
- `exit_button`: an Exit button.
- `close_x`: a close (X) control on a dialog or banner.

Labelling rules:

- Box the whole sprite, not just the part that is easiest to see.
- Skip frames with no game in them (loading screens, black frames): leave them with no boxes.
- Never label names or tags. A player's name or club tag showing in a screenshot is not a
  class this kit tracks, and it should never be typed into a label.

### Labelling on another computer

To label on a computer that has no checkout of this repository, a Mac for example, build a
label pack after Step 4:

```
uv run python tools/play/pack.py
```

It writes `label-pack` under the dataset root and `label-pack.zip` next to it: the frames the
import file references, `labelstudio/config.xml`, `labelstudio/tasks.json`, a start script
for macOS and Linux (`start.sh`) and one for Windows (`start.ps1`), and a README that carries
the rest of the recipe, from installing `uv` to exporting the labels. `--source` packs only
the named sources (repeat the flag for more), `--out` picks another folder, and `--no-zip`
skips the zip. Copy the zip over, follow its README, and bring the exported JSON file back
for Step 6.

## Step 6: export and build the COCO set

Export the project as **JSON**, not JSON-MIN and not COCO. Label Studio's own COCO export does
not carry the split rule this kit needs, so `coco.py` does the conversion:

```
uv run python tools/play/coco.py export.json
```

`--root` picks the dataset root. `--from-predictions` builds the set from a task file's
predictions instead of hand-made annotations, which is the normal route with the auto-labeller:

    uv run python tools/play/coco.py <dataset root>/labelstudio/autolabel.json --from-predictions

The split is by source, never by frame: consecutive frames of one video look almost identical,
so splitting by frame would put near-copies of the same picture on both sides of the split and
make the score look better than the model really is. Each frame gets a **split key**, and each
key is assigned to a **split** (`train`, `valid` or `test`) by hashing it, so the same frames
land in the same split on every run and adding sources later does not move the old ones. Train
always keeps the largest split.

A YouTube source (`yt-<video id>`) is one key: the whole video goes to one split. A recorder or
capture session is one long sitting, and hashed whole it would put every frame it holds, and
every tap anchor it shows, on one side, which leaves valid and test with no tap-anchor truth.
So a session is cut into contiguous pieces, and each piece is its own key. A session frame whose
file name carries a time, such as `20261104-063119-00133.jpg`, is keyed by its 5-minute window
(`WINDOW_MINUTES` in `coco.py`). The key is `<session>@<window>`, for example
`capture-20261104b@20261104-0630`. `frames.py` saves frames as `<source>-<number>.jpg`, which
carries no time, but it numbers a session's frames upwards in the order they were captured. So a
frame named that way is keyed by its block of 60 frames instead (`SESSION_BLOCK_FRAMES` in
`coco.py`). The key is `<session>@b<block>`, the frame number divided by 60 and rounded down, so
`capture-20261104-000123.jpg` is in `capture-20261104@b2`. Sixty frames is three minutes of a
capture at one frame every 3 seconds: a 30-minute capture becomes ten keys and a 138-frame
recorder session three. A time in the name always wins over a number, and the frames of one
window or block always stay together. A recorder session is named after the second it started,
and that stamp is never read as a frame's time. A frame whose name carries neither a time nor its
source's frame number falls back to its whole source as the key.
With fewer than three split keys there is not enough to split at all, so `coco.py` writes the
same frames into all three splits and marks the layout as a smoke set; every later tool in this
kit warns when it sees that mark.

`coco.py` prints the box count of every class, zeros included, then the tap-anchor truth in each
split; `split.json` keeps the same per-split counts under `split_boxes`. A threshold or a score
drawn from a handful of boxes means nothing, so `coco.py` prints a WARNING for every tap anchor
that has fewer than 30 truth boxes in `valid` or in `test`. No template finds `skull_star`,
`team_up_panel` or `event_tab`, so they have no label source: they are listed as such and never
warned about. The warnings do not stop the build. `autolabel.py` and `train.py` also list every
class, zeros included.

## Step 7: train

`train.py` is a self-contained script: `uv run tools/play/train.py` builds its own environment
on first run, including torch from the CUDA 12.8 index, about 3 GB. It never touches the
project's own `pyproject.toml` or lock file.

```
uv run tools/play/train.py
```

`--root` picks the dataset root, `--epochs` the number of passes over the training set (an
**epoch** is one full pass over the training data; the default is 60, a real run needs many
more than the 8 used for a smoke test), `--batch` the number of images per training step, and
`--accum` the number of steps to accumulate gradients over before updating the model, which
lets a small GPU behave like a bigger one. On an RTX 4080 SUPER, 8 epochs on a tiny set took
about 50 seconds; a real run takes much longer. Output goes to `runs/<YYYYmmdd-HHMMSS>/`.

Tap anchors (the buttons, cards and panels the controller taps) are rare in the frames, so
training **oversamples** them: every training image that holds at least one tap-anchor box goes
into the training data four times instead of once. `--anchor-repeat` sets how many times, as a
whole number of 1 or more, and `--anchor-repeat 1` turns it off. The repeats are written as
annotation files in `runs/<YYYYmmdd-HHMMSS>/feed/` that point at the frames in `coco/`, so no
frame is copied, and the run trains on that folder instead of `coco/`. Only the training split
is oversampled: the validation and test images still go in once each, so the thresholds export
picks and the score are measured on images counted once. Training prints how many images it
repeated. When no training image holds a tap anchor, or the flag is 1, it trains on `coco/`
directly.

## Step 8: export

```
uv run tools/play/export.py
```

`--root` picks the dataset root, `--run` the training run to export (a folder path, or just the
folder's name under `runs/`, for example `--run 20260918-101112`; without it the most recent run is
used). Export converts the trained model to
**ONNX**, a portable model file format, then runs the ONNX model over the validation images and
checks its output against the training library's own output on the same images, so a broken
conversion is caught here rather than later. On the same RTX 4080 SUPER, export took about 25
seconds and produced a `play.onnx` file of about 119 MB.

Export also picks a **threshold** for every class: the score above which a detection counts.
For each class, it is the lowest threshold between 0.30 and 0.95 that still keeps
**precision** (the fraction of the model's detections of that class that are actually correct)
at or above a target on the validation images. The target depends on what the class is for.
The tap anchors (the buttons, cards and panels the controller taps, where one false positive
could spend gems) always need 0.98, and no flag changes that. Every other class, such as
`self`, `teammate` and `enemy`, is only watched and never tapped, so it needs 0.90; pass
`--min-precision-other` with a number above 0 and at most 1 to change that target. A class with
no validation images, or one that never reaches its target, gets 0.95, the most cautious
threshold available. Thresholds, the target each class was held to and the run they came from
are written to `play.json` next to `play.onnx`.

Next to each threshold, `play.json` records how many validation boxes of that class there were
(`truth`) and whether the threshold is **certified** (`certified`). A threshold is certified
only when there were at least 30 validation boxes of its class and it really reached its
precision target. With fewer boxes, one box moves the precision too far to trust it, and the
0.95 fallback is never certified, however many boxes there were. Export prints every class that
is not certified, with the reason. An uncertified threshold is still written and the detector
still loads it: it means the model is not proven on that class yet, and the fix is more frames
that show it. 30 is the same number `coco.py` warns about when a tap anchor has too few boxes in
the validation or test split.

## Step 9: score

```
uv run python tools/play/score.py
```

`score.py` runs the exported model over the **held-out test split**, the frames `coco.py` put
in `coco/test/` and training never saw, and compares what it finds with the farm's own
templates on every 1600 x 900 frame there. It trusts that split only when `coco/split.json`
carries the same training set hash as `play.json`, since a build made after training can have
moved training frames into the test folder. The report is written to `score.json` next to
`play.onnx` and ends in one of three verdicts, which is also the exit code:

- **PASS** (exit code 0): every tap anchor with a template behind it was measured and found
  often enough, and none was detected where the templates see none.
- **FAIL** (exit code 1): the model produced a **false positive** (a detection that should not
  be there) on a tap anchor, a class the farm would act on, because a tap on the wrong pixel
  can spend gems or leave a match queue. A single extra detection next to a correct one counts.
  A tap anchor with at least `--min-truth` template boxes also fails when its **recall** (the
  fraction of those boxes the model found too) is under `--min-recall`.
- **INCONCLUSIVE** (exit code 2): nothing in the run proves the model safe. That is the verdict
  when a tap anchor with a template has fewer than `--min-truth` template boxes in the scored
  frames, when no frame was scored, when there is no COCO build, when the model was trained on
  a smoke set, and for every folder scan.

A FAIL wins over an INCONCLUSIVE, because a false positive is real wherever it was seen. For
each tap anchor, `score.json` lists its truth (the template boxes in the scored frames), true
positives, misses, false positives and recall, plus the reasons the verdict is not PASS, and
the same reasons are printed under the verdict. Three tap anchors have no template behind them
(`skull_star`, `team_up_panel`, `event_tab`): they are listed as having no label source, any
detection of one is listed as unverified for you to check by hand, and they never change the
verdict.

`--root` picks the dataset root and `--models` the folder holding `play.onnx` (default
`models/`). `--min-truth` is how many template boxes a tap anchor needs before its recall
counts (default 30), and `--min-recall` is the recall it must then reach (default 0.5).
`--cuda` runs the model on the CUDA execution provider; if onnxruntime cannot give it, the run
stops with an error instead of quietly using the CPU, and the fix is the `play-gpu` extra
described in [install.md](install.md). `--scan-folders` scores every picture under the dataset
root's own frames and the calibration recordings instead of the test split, and `--frames`
scans a folder you name instead (repeat the flag for more). Those folders hold training frames
too, so a scan is only for looking at footage and always ends INCONCLUSIVE.

## How the first model ships

No `play.onnx` or `play.json` is committed by this kit. Once a run passes `score.py`, the
model and its thresholds move into a calibration pull request of their own, one that also
carries the training set hash from `coco/split.json`, the score report, and the held-out
numbers the model was judged on. That is the pull request an owner reviews before the detector
is allowed anywhere near a live instance.
