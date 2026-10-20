# doomscroll-tracker

Tracking software for my art class. It tracks thumb flicks (scrolls) in a camera recording of me
scrolling an iPhone, and turns each upward swipe into a deep-red watercolor brushstroke on a
peach field. Each stroke slowly bleeds into the surface, gradually turning it red.

Strokes are upward thumb movements tracked inside the calibrated screen area. That is a proxy for
swipes: it does not prove screen contact or that the phone scrolled, and these are not
fingerprints. Detection accuracy has not been measured.

Footage, data and renders stay local and are git-ignored.

## Setup

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p models && curl -fL -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
.venv/bin/python -m pytest -q
```

The model comes from Google's official MediaPipe Hand Landmarker page
(<https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker>).
A harmless `objc: Class AVFFrameReceiver is implemented in both ...` warning may appear.

## Use

Put the recording in `footage/`, then:

```bash
V=footage/test.mov; PY=.venv/bin/python
$PY -m doomscroll calibrate --video $V                      # click screen corners TL, TR, BR, BL; Enter saves
$PY -m doomscroll track     --video $V                      # slow (MediaPipe); run once
$PY -m doomscroll detect    --video $V                      # fast; rerun after any config/calibration change
$PY -m doomscroll diagnose  --video $V --start 20 --end 36  # check tracking (video with audio)
$PY -m doomscroll diagnose  --video $V --contact-sheet      # one image per detected swipe
$PY -m doomscroll render    --video $V --start 0 --end 60   # preview
$PY -m doomscroll render    --video $V                      # full, real time, 720x1280 @ 30 fps
```

Useful options: `render --width 360 --height 640` (small preview), `render --speed 10`
(accelerated preview, labeled as such), `--config config/mine.toml` (any command).

**Calibration** assumes the phone doesn't move. If it does, add a calibration that applies from a
given time onward: `calibrate --video $V --append-at 95 --time 100`. Calibration keys: `u` undo,
`r` reset, `,` `.` ±1 s, `[` `]` ±10 s, `q` cancel.

**Timing:** t = 0 is the first video frame of the Mac recording, and all times in the data use this
clock. The render is silent; `renders/<name>/timing.json` records how its time maps to the source.

## Tuning

Copy `config/default.toml` to `config/mine.toml` and edit it. Units are explained in the file.

| want | change |
|---|---|
| more or fewer detected swipes | `detect.min_displacement`, `detect.min_speed` |
| strokes dissolve faster or slower | `render.bleed.bleed_time_s` |
| brown spreads farther or faster | `render.bleed.distance_after_60s_px` |
| canvas turns brown sooner or later | `render.palette.density_scale`, `render.brush.density_per_pass` |
| stronger or weaker stroke ghosts in the brown | `render.bleed.residue_fraction` |
| brush look | `render.brush.width_px`, `bristle_softness` (higher = more of a wash), `dry_length`, `dry_break` |
| watercolor look | `render.wet.edge_darkening`, `pooling`, `granulation`, `opacity`; `render.bleed.wet_spread_after_60s_px` |
| colors | `render.palette.stops` (ground), `render.wet.color_light` / `color_dense` (fresh paint) |
| your own fingerprint texture | `render.brush.texture = "path/to/image.png"` |
| track a different finger point | `hand.point_landmarks` |

`$PY -m doomscroll synth` makes a clearly labeled synthetic gestures file for testing the renderer.
