"""`detect` command: cached tracking + calibration -> thumb trajectory + gestures JSON + summary."""
from __future__ import annotations

import csv
import datetime as _dt
from collections import Counter
from pathlib import Path

import numpy as np

from .calibration import load_calibration
from .common import SWIPE_PROXY_DISCLAIMER, PipelineError, file_fingerprint, sha256_file, write_json
from .gestures import LABEL_NAMES, STATE_NAMES, detect_strokes
from .tracking import load_tracking
from .trackproc import (JUMP_REJECTED, NO_HAND, OK, OTHER_HAND_ONLY, STATUS_NAMES, segment_track,
                        select_hand, smooth_segments)

TIME_ORIGIN = ("t = 0 s is the presentation time of the first video frame of the source recording "
               "(video stream start_time); all times are source-relative seconds from presentation timestamps")


def process(raw: dict, calib, cfg: dict) -> dict:
    """Pure processing of cached tracking -> per-frame arrays + strokes (used by detect & diagnose)."""
    t = raw["t"]
    lm = raw["landmarks_px"]
    point_lms = [int(j) for j in cfg["hand"]["point_landmarks"]]
    w, _ = calib.image_size
    center = calib.to_image(float(np.median(t)), [[0.5, 0.5]])[0]
    sel, status, new_track = select_hand(t, lm, raw["handedness"], cfg["hand"], w, center, point_lms)
    N = len(t)
    thumb_px = np.full((N, 2), np.nan)
    hscore = np.full(N, np.nan)
    ok = sel >= 0
    # tracked point = mean of the configured landmarks of the selected hand
    thumb_px[ok] = lm[np.flatnonzero(ok), sel[ok]][:, point_lms, :2].mean(axis=1)
    hscore[ok] = raw["handedness_score"][np.flatnonzero(ok), sel[ok]]
    phone_raw = calib.to_phone(t, thumb_px)
    seg, status = segment_track(t, phone_raw, status, calib.aspect, cfg["detect"], new_track)
    phone = smooth_segments(t, phone_raw, seg, calib.aspect, cfg["smoothing"])
    strokes, online, label, v_up = detect_strokes(t, phone, seg, cfg["detect"])
    return dict(t=t, pts=raw["pts"], sel=sel, status=status, new_track=new_track, thumb_px=thumb_px,
                hscore=hscore, phone_raw=phone_raw, seg=seg, phone=phone, strokes=strokes,
                online=online, label=label, v_up=v_up)


def _runs(mask: np.ndarray):
    """(start, end_exclusive) index runs where mask is True."""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def build_outputs(P: dict, cfg: dict, video_duration: float):
    t, status = P["t"], P["status"]
    N = len(t)
    accepted = [s for s in P["strokes"] if s.accepted]
    rejected = [s for s in P["strokes"] if not s.accepted]
    gestures = []
    for gid, s in enumerate(accepted, 1):
        idx = np.array(s.indices)
        t0, t1 = float(t[idx[0]]), float(t[idx[-1]])
        span = (t >= t0) & (t <= t1)
        gaps = np.diff(t[idx]) if len(idx) > 1 else np.array([0.0])
        pts = [{"t": round(float(t[i]), 5), "x": round(float(P["phone"][i, 0]), 5),
                "y": round(float(P["phone"][i, 1]), 5),
                "x_raw": round(float(P["phone_raw"][i, 0]), 5), "y_raw": round(float(P["phone_raw"][i, 1]), 5)}
               for i in idx]
        gestures.append({
            "id": gid,
            "start_s": round(t0, 5), "end_s": round(t1, 5), "duration_s": round(t1 - t0, 5),
            "end_reason": s.end_reason,
            "upward_displacement_screen_h": round(s.displacement, 4),
            "peak_upward_speed_screen_h_per_s": round(s.peak_speed, 3),
            "points": pts,
            "quality": {
                "n_points": int(len(idx)),
                "frames_in_span": int(span.sum()),
                "observed_fraction": round(len(idx) / max(1, int(span.sum())), 3),
                "max_point_gap_s": round(float(gaps.max()), 4),
                "jump_rejections_in_span": int((status[span] == JUMP_REJECTED).sum()),
                "other_hand_only_frames_in_span": int((status[span] == OTHER_HAND_ONLY).sum()),
                "mean_handedness_score": round(float(np.nanmean(P["hscore"][idx])), 3),
                "mean_smoothing_offset_screen_h": round(float(np.nanmean(np.abs(P["phone"][idx] - P["phone_raw"][idx]))), 4),
            },
        })
    rej = [{"start_s": round(float(t[s.indices[0]]), 4), "end_s": round(float(t[s.indices[-1]]), 4),
            "end_reason": s.end_reason, "upward_displacement_screen_h": round(s.displacement, 4),
            "peak_upward_speed_screen_h_per_s": round(s.peak_speed, 3), "n_points": len(s.indices)}
           for s in rejected]

    # gaps: runs of frames without an accepted observation of the selected hand
    max_gap = float(cfg["detect"]["max_gap_s"])
    gaps = []
    for a, b in _runs(status != OK):
        t_a = t[a - 1] if a > 0 else 0.0
        t_b = t[b] if b < N else video_duration
        if t_b - t_a > max_gap:
            c = Counter(STATUS_NAMES[int(x)] for x in status[a:b])
            gaps.append({"from_s": round(float(t_a), 3), "to_s": round(float(t_b), 3),
                         "duration_s": round(float(t_b - t_a), 3), "causes": dict(c)})
    gaps.sort(key=lambda g: -g["duration_s"])
    starts = np.array([g["start_s"] for g in gestures])
    ends = np.array([g["end_s"] for g in gestures])
    pauses = (starts[1:] - ends[:-1]) if len(gestures) > 1 else np.array([])
    summary = {
        "video_duration_s": round(video_duration, 3),
        "tracked_range_s": [round(float(t[0]), 3), round(float(t[-1]), 3)],
        "frames": N,
        "coverage": {
            "frames_with_any_hand": int((status != NO_HAND).sum()),
            "frames_with_selected_hand_ok": int((status == OK).sum()),
            "fraction_selected_hand_ok": round(float((status == OK).mean()), 4),
            "status_counts": {STATUS_NAMES[k]: int((status == k).sum()) for k in STATUS_NAMES},
            "hand_reacquisitions": int(P["new_track"].sum()),
            "segments": int(P["seg"].max() + 1) if (P["seg"] >= 0).any() else 0,
        },
        "gaps_longer_than_max_gap": {"count": len(gaps), "total_s": round(sum(g["duration_s"] for g in gaps), 2),
                                     "longest": gaps[:10]},
        "accepted_gestures": len(gestures),
        "rejected_candidates": len(rej),
        "end_reasons_accepted": dict(Counter(g["end_reason"] for g in gestures)),
        "gesture_duration_s": _stats([g["duration_s"] for g in gestures]),
        "pause_between_gestures_s": _stats(pauses.tolist()),
        "accuracy_note": "No ground-truth labels: detection accuracy is NOT measured. Review the diagnostic video.",
        "disclaimer": SWIPE_PROXY_DISCLAIMER,
    }
    return gestures, rej, summary


def _stats(vals):
    if not vals:
        return None
    a = np.array(vals)
    return {"min": round(float(a.min()), 3), "median": round(float(np.median(a)), 3), "max": round(float(a.max()), 3)}


def write_track_csv(path: Path, P: dict, raw: dict) -> None:
    lm = raw["landmarks_px"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "pts", "t_s", "status", "hand_slot", "new_track", "handedness_score",
                    "wrist_px_x", "wrist_px_y", "thumb_ip_px_x", "thumb_ip_px_y", "thumb_tip_px_x", "thumb_tip_px_y",
                    "phone_raw_x", "phone_raw_y", "segment", "phone_smooth_x", "phone_smooth_y",
                    "v_up_screen_h_per_s", "online_state", "final_label"])
        for i in range(len(P["t"])):
            k = P["sel"][i]
            wr = lm[i, k, 0, :2] if k >= 0 else (np.nan, np.nan)
            ip = lm[i, k, 3, :2] if k >= 0 else (np.nan, np.nan)
            f4 = lambda v: "" if not np.isfinite(v) else f"{v:.4f}"
            f1 = lambda v: "" if not np.isfinite(v) else f"{v:.1f}"
            w.writerow([i, int(P["pts"][i]), f"{P['t'][i]:.5f}", STATUS_NAMES[int(P["status"][i])], int(k),
                        int(P["new_track"][i]), f4(P["hscore"][i]), f1(wr[0]), f1(wr[1]), f1(ip[0]), f1(ip[1]),
                        f1(P["thumb_px"][i, 0]), f1(P["thumb_px"][i, 1]),
                        f4(P["phone_raw"][i, 0]), f4(P["phone_raw"][i, 1]), int(P["seg"][i]),
                        f4(P["phone"][i, 0]), f4(P["phone"][i, 1]), f4(P["v_up"][i]),
                        STATE_NAMES[int(P["online"][i])], LABEL_NAMES[int(P["label"][i])]])


def print_summary(summary: dict, gestures: list) -> None:
    c = summary["coverage"]
    print("\n=== detection summary ===")
    print(f"duration            {summary['video_duration_s']:.2f} s   frames {summary['frames']}")
    print(f"any hand detected   {c['frames_with_any_hand']} frames ({100*c['frames_with_any_hand']/summary['frames']:.1f}%)")
    print(f"selected thumb ok   {c['frames_with_selected_hand_ok']} frames ({100*c['fraction_selected_hand_ok']:.1f}%)  "
          f"jump-rejected {c['status_counts']['jump_rejected']}  other-hand-only {c['status_counts']['other_hand_only']}  "
          f"re-acquisitions {c['hand_reacquisitions']}")
    g = summary["gaps_longer_than_max_gap"]
    print(f"gaps > max_gap      {g['count']} totalling {g['total_s']:.1f} s; longest:")
    for gg in g["longest"][:5]:
        print(f"    {gg['from_s']:8.2f} -> {gg['to_s']:8.2f} s  ({gg['duration_s']:.2f} s)  {gg['causes']}")
    print(f"accepted gestures   {summary['accepted_gestures']}   (rejected candidates {summary['rejected_candidates']})")
    if gestures:
        print(f"gesture durations   {summary['gesture_duration_s']}   pauses between {summary['pause_between_gestures_s']}")
        print(f"end reasons         {summary['end_reasons_accepted']}")
    print("note: accuracy is not measured (no labeled ground truth).")
    print(f"note: {SWIPE_PROXY_DISCLAIMER}")


def run_detect(video: str, data_dir: Path, cfg: dict, calibration_path: str | None = None) -> dict:
    raw, meta = load_tracking(data_dir, video)
    calib_path = Path(calibration_path) if calibration_path else data_dir / "calibration.json"
    calib = load_calibration(calib_path)
    vi = meta["video_info"]
    if list(calib.image_size) != [vi["width"], vi["height"]]:
        raise PipelineError(f"calibration image size {calib.image_size} != video {vi['width']}x{vi['height']}")
    P = process(raw, calib, cfg)
    gestures, rejected, summary = build_outputs(P, cfg, float(vi["duration_s"]))
    doc = {
        "schema": "doomscroll-gestures/1",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "disclaimer": SWIPE_PROXY_DISCLAIMER,
        "time_origin": TIME_ORIGIN,
        "coordinates": "normalized phone screen: x 0..1 left->right, y 0..1 top->bottom; "
                       "x,y smoothed; x_raw,y_raw unsmoothed homography output",
        "source_video": meta["source_video"],
        "video_duration_s": vi["duration_s"],
        "tracked_range_s": summary["tracked_range_s"],
        "calibration": {**calib.summary(), "sha256": sha256_file(calib_path)},
        "tracking_cache": {"created": meta["created"], "model_sha256": meta["model"]["sha256"]},
        "config": {k: cfg[k] for k in ("hand", "smoothing", "detect")},
        "summary": summary,
        "gestures": gestures,
        "rejected_candidates": rejected,
    }
    write_json(data_dir / "gestures.json", doc)
    write_json(data_dir / "detect_summary.json", summary)
    write_track_csv(data_dir / "thumb_track.csv", P, raw)
    print_summary(summary, gestures)
    print(f"\nwrote {data_dir/'gestures.json'}\n      {data_dir/'thumb_track.csv'}\n      {data_dir/'detect_summary.json'}")
    return doc
