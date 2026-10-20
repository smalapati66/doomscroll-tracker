"""Stage 2a: turn raw per-frame hand detections into one clean thumb trajectory.

Steps (all cheap; rerun freely without MediaPipe):
  1. select_hand      pick the intended hand each frame, keeping identity by wrist continuity
  2. map to phone     image px -> normalized phone coords via the calibration homography
  3. segment_track    reject implausible jumps; split into segments at every gap > max_gap_s
  4. smooth_segments  time-aware One-Euro smoothing, restarted at every segment

Missing observations stay NaN throughout; nothing is interpolated or bridged.
"""
from __future__ import annotations

import numpy as np

# per-frame status codes
OK, NO_HAND, OTHER_HAND_ONLY, JUMP_REJECTED, UNMAPPABLE = 0, 1, 2, 3, 4
STATUS_NAMES = {OK: "ok", NO_HAND: "no_hand", OTHER_HAND_ONLY: "other_hand_only",
                JUMP_REJECTED: "jump_rejected", UNMAPPABLE: "unmappable"}

WRIST = 0


def select_hand(t: np.ndarray, landmarks_px: np.ndarray, handedness: np.ndarray, hcfg: dict,
                image_w: float, screen_center_px=None, point_lms=(4,)):
    """Choose one hand per frame.

    Returns (sel (N,) int hand slot or -1, status (N,), new_track (N,) bool marking re-acquisitions).
    While a track is alive (seen within reacquire_after_s), the hand whose wrist is nearest the
    previous wrist is kept, if within max_assoc_distance; otherwise the frame is "other_hand_only".
    With no live track the hand is (re)acquired with the `select` rule.
    """
    N, H = landmarks_px.shape[:2]
    sel = np.full(N, -1, np.int64)
    status = np.full(N, NO_HAND, np.int8)
    new_track = np.zeros(N, bool)
    rule = hcfg["select"]
    max_d = float(hcfg["max_assoc_distance"]) * image_w
    reacq = float(hcfg["reacquire_after_s"])
    last_wrist, last_t = None, -np.inf
    for i in range(N):
        cands = [k for k in range(H) if np.isfinite(landmarks_px[i, k, WRIST, 0])]
        if not cands:
            continue
        wrists = landmarks_px[i, cands, WRIST, :2]
        if last_wrist is not None and t[i] - last_t <= reacq:
            d = np.linalg.norm(wrists - last_wrist, axis=1)
            j = int(np.argmin(d))
            if d[j] > max_d:
                status[i] = OTHER_HAND_ONLY
                continue
            k = cands[j]
        else:
            if rule == "rightmost":
                k = cands[int(np.argmax(wrists[:, 0]))]
            elif rule == "leftmost":
                k = cands[int(np.argmin(wrists[:, 0]))]
            elif rule in ("label_left", "label_right"):
                want = 0 if rule == "label_left" else 1
                ok = [c for c in cands if handedness[i, c] == want]
                if not ok:
                    status[i] = OTHER_HAND_ONLY
                    continue
                k = ok[0]
            elif rule == "nearest_screen":
                if screen_center_px is None:
                    raise ValueError("select='nearest_screen' needs a calibration")
                tips = landmarks_px[i, cands][:, list(point_lms), :2].mean(axis=1)
                k = cands[int(np.argmin(np.linalg.norm(tips - screen_center_px, axis=1)))]
            else:
                raise ValueError(f"unknown hand.select rule '{rule}'")
            new_track[i] = True
        sel[i] = k
        status[i] = OK
        last_wrist, last_t = landmarks_px[i, k, WRIST, :2].copy(), t[i]
    return sel, status, new_track


def iso_dist(d_xy: np.ndarray, aspect: float) -> np.ndarray:
    """Length of phone-coordinate displacements in screen heights (x scaled by width/height)."""
    d = np.asarray(d_xy, dtype=np.float64)
    return np.hypot(d[..., 0] * aspect, d[..., 1])


def segment_track(t: np.ndarray, xy: np.ndarray, status: np.ndarray, aspect: float, dcfg: dict,
                  new_track: np.ndarray | None = None):
    """Reject implausible jumps and assign segment ids (-1 = not a valid observation).

    A sample is compared with the last *accepted* sample. If the step exceeds jump_distance or
    jump_speed it is rejected (status JUMP_REJECTED, treated as missing). Once more than
    max_gap_s passes without an accepted sample, the next observation starts a new segment, so a
    real relocation is picked up again without ever drawing a connecting line.
    """
    N = len(t)
    status = status.copy()
    seg = np.full(N, -1, np.int64)
    max_gap = float(dcfg["max_gap_s"])
    jd, js = float(dcfg["jump_distance"]), float(dcfg["jump_speed"])
    last, cur = None, -1
    for i in range(N):
        if status[i] != OK:
            continue
        if not np.all(np.isfinite(xy[i])):
            status[i] = UNMAPPABLE
            continue
        if last is None or t[i] - t[last] > max_gap or (new_track is not None and new_track[i]):
            cur += 1
        else:
            dt = max(t[i] - t[last], 1e-6)
            d = float(iso_dist(xy[i] - xy[last], aspect))
            if d > jd or d / dt > js:
                status[i] = JUMP_REJECTED
                continue
        seg[i] = cur
        last = i
    return seg, status


def smooth_segments(t: np.ndarray, xy: np.ndarray, seg: np.ndarray, aspect: float, scfg: dict) -> np.ndarray:
    """One-Euro filter (Casiez et al. 2012), time-aware, restarted at each segment."""
    out = np.full_like(xy, np.nan, dtype=np.float64)
    fmin, beta, fd = float(scfg["min_cutoff_hz"]), float(scfg["beta"]), float(scfg["d_cutoff_hz"])

    def alpha(fc, dt):
        tau = 1.0 / (2 * np.pi * fc)
        return 1.0 / (1.0 + tau / dt)

    prev_i, x_hat, dx_hat = None, None, None
    for i in np.flatnonzero(seg >= 0):
        x = xy[i].astype(np.float64)
        if prev_i is None or seg[i] != seg[prev_i]:
            x_hat, dx_hat = x, np.zeros(2)
        else:
            dt = max(t[i] - t[prev_i], 1e-6)
            dx = (x - x_hat) / dt
            ad = alpha(fd, dt)
            dx_hat = ad * dx + (1 - ad) * dx_hat
            speed = float(np.hypot(dx_hat[0] * aspect, dx_hat[1]))
            a = alpha(fmin + beta * speed, dt)
            x_hat = a * x + (1 - a) * x_hat
        out[i] = x_hat
        prev_i = i
    return out
