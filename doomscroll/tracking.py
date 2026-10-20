"""Stage 1: run MediaPipe Hand Landmarker over every decoded frame and cache the raw results.

Nothing here selects a hand, maps to the screen, smooths, or interpolates. The cache keeps every
detected hand (up to num_hands) per frame, with NaN where nothing was detected, so all later
choices can be changed without rerunning MediaPipe.

Cache file (data/<stem>/tracking_raw.npz):
  pts             (N,)        int64   raw stream presentation timestamps
  t               (N,)        float64 source-relative seconds
  mp_timestamp_ms (N,)        int64   timestamp passed to MediaPipe (strictly increasing)
  n_hands         (N,)        int8    number of hands detected in the frame
  landmarks_px    (N,H,21,3)  float32 x,y in display-oriented ORIGINAL image pixels (y may exceed the
                                      frame height when the hand extends below the bottom edge); z = MediaPipe relative depth
                                      (NaN when that hand slot is empty)
  handedness      (N,H)       int8    0 = "Left", 1 = "Right" (MediaPipe label; assumes mirrored input), -1 empty
  handedness_score(N,H)       float32 MediaPipe handedness confidence (the only per-hand score the Python
                                      Tasks API returns); NaN when empty
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

from .common import (PipelineError, Progress, file_fingerprint, read_json, require_file,
                     sha256_file, write_json)
from .video import iter_frames, probe, scan_timestamps

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


def preprocess(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Image actually given to MediaPipe. Geometry change is only padding at the bottom."""
    import cv2
    if cfg.get("clahe"):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    pad = int(cfg.get("pad_bottom_px", 0))
    if pad > 0:
        img = cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return np.ascontiguousarray(img)


def _make_landmarker(cfg: dict):
    model = Path(cfg["model_path"])
    if not model.is_absolute():
        from .common import PROJECT_ROOT
        model = PROJECT_ROOT / model
    require_file(model, "MediaPipe hand landmarker model",
                 f"download it with:  curl -fL -o models/hand_landmarker.task {MODEL_URL}")
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=int(cfg["num_hands"]),
        min_hand_detection_confidence=float(cfg["min_hand_detection_confidence"]),
        min_hand_presence_confidence=float(cfg["min_hand_presence_confidence"]),
        min_tracking_confidence=float(cfg["min_tracking_confidence"]),
    )
    try:
        return mp, vision.HandLandmarker.create_from_options(opts), model
    except Exception as e:
        raise PipelineError(f"could not load MediaPipe model {model}: {e}") from e


def run_tracking(video: str, out_dir: Path, cfg: dict, start_s: float = 0.0,
                 end_s: float | None = None) -> Path:
    tcfg = cfg["tracking"]
    info = probe(video)
    times = scan_timestamps(video)
    sel = (times >= start_s - 1e-9) & ((times < end_s) if end_s is not None else True)
    expected = int(sel.sum())
    H = int(tcfg["num_hands"])
    mp, landmarker, model_path = _make_landmarker(tcfg)

    pts_l, t_l, ts_l, n_l, lm_l, hd_l, hs_l = [], [], [], [], [], [], []
    last_ts = -1
    prog = Progress("tracking", expected)
    with landmarker:
        for fr in iter_frames(video, start_s, end_s, fmt="rgb24"):
            w = fr.image.shape[1]
            mp_img = preprocess(fr.image, tcfg)
            h = mp_img.shape[0]  # normalized y is relative to the (padded) image MediaPipe saw
            # MediaPipe VIDEO mode needs strictly increasing integer ms; true times are kept separately.
            ts = max(int(round(fr.t * 1000.0)), last_ts + 1)
            last_ts = ts
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_img), ts)
            lm = np.full((H, 21, 3), np.nan, np.float32)
            hd = np.full(H, -1, np.int8)
            hs = np.full(H, np.nan, np.float32)
            for k, hand in enumerate(res.hand_landmarks[:H]):
                lm[k] = [(p.x * w, p.y * h, p.z) for p in hand]
                if k < len(res.handedness) and res.handedness[k]:
                    cat = res.handedness[k][0]
                    hd[k] = 0 if cat.category_name == "Left" else 1
                    hs[k] = cat.score
            pts_l.append(fr.pts); t_l.append(fr.t); ts_l.append(ts)
            n_l.append(len(res.hand_landmarks)); lm_l.append(lm); hd_l.append(hd); hs_l.append(hs)
            prog.update(extra=f"t={fr.t:7.2f}s hands={len(res.hand_landmarks)}")
    prog.done()
    if not t_l:
        raise PipelineError("no frames were decoded in the requested range")

    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "tracking_raw.npz"
    np.savez_compressed(
        cache, pts=np.array(pts_l, np.int64), t=np.array(t_l), mp_timestamp_ms=np.array(ts_l, np.int64),
        n_hands=np.array(n_l, np.int8), landmarks_px=np.stack(lm_l), handedness=np.stack(hd_l),
        handedness_score=np.stack(hs_l))
    n = np.array(n_l)
    meta = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "source_video": file_fingerprint(video),
        "video_info": info.to_dict(),
        "range_s": [start_s, end_s],
        "frames_tracked": len(t_l),
        "frames_expected_from_packets": expected,
        "frames_with_any_hand": int((n > 0).sum()),
        "frames_with_two_hands": int((n > 1).sum()),
        "model": {"path": str(model_path), "sha256": sha256_file(model_path), "source_url": MODEL_URL},
        "mediapipe_version": mp.__version__,
        "tracking_config": tcfg,
        "landmark_names": LANDMARK_NAMES,
        "notes": "raw per-frame output for all detected hands; no selection, smoothing or interpolation",
    }
    write_json(out_dir / "tracking_meta.json", meta)
    print(f"tracked {len(t_l)} frames ({expected} expected); any hand in {meta['frames_with_any_hand']} "
          f"({100*meta['frames_with_any_hand']/len(t_l):.1f}%), two hands in {meta['frames_with_two_hands']}")
    print(f"wrote {cache}")
    return cache


def load_tracking(out_dir: Path, video: str | None = None) -> tuple[dict, dict]:
    cache = require_file(out_dir / "tracking_raw.npz", "tracking cache",
                         "run:  .venv/bin/python -m doomscroll track --video footage/<file>.mov")
    meta = read_json(out_dir / "tracking_meta.json", "tracking metadata")
    if video is not None and Path(video).is_file():
        fp = file_fingerprint(video)
        if fp["partial_sha256"] != meta["source_video"]["partial_sha256"] or fp["size"] != meta["source_video"]["size"]:
            raise PipelineError(f"tracking cache in {out_dir} was made from a different video file; rerun `track`")
    with np.load(cache) as z:
        raw = {k: z[k] for k in z.files}
    return raw, meta
