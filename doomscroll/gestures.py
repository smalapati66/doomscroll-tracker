"""Stage 2b: upward-gesture state machine on the smoothed thumb trajectory (phone coords).

Upward movement inside the screen region is only a PROXY for a swipe. It cannot prove that the
thumb touched the glass or that the phone scrolled.

States:
  IDLE       watching. Downward returns, stalls and jitter keep us here.
  CANDIDATE  upward speed rose above start_speed. Points are buffered from the stroke's true
             beginning (the lowest point just before the rise, found by looking back up to backtrack_s).
  STROKE     the candidate qualified (upward travel >= min_displacement and peak upward speed
             >= min_speed). Accepted strokes keep their candidate points.

A CANDIDATE/STROKE ends on:
  reversal       downward retrace > reversal_distance, or downward motion lasting > reversal_tolerance_s
  pause          stalled (|speed| <= end_speed) for > pause_tolerance_s
  tracking_lost  gap > max_gap_s, a rejected jump, or a hand re-acquisition (segment change)
  left_region    thumb more than boundary_tolerance outside the calibrated screen
  end_of_data
On ending, points after the highest point are dropped (the start of a downward return is never
part of a stroke). A candidate that never qualified is discarded. There is no cooldown: the next
upward motion can open a new candidate on the very next sample.

All thresholds are in seconds and screen fractions. Speeds are measured over a fixed time
window (velocity_window_s), never per frame, so results do not depend on the frame rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# online state per sample
S_MISSING, S_IDLE, S_CANDIDATE, S_STROKE = 0, 1, 2, 3
STATE_NAMES = {S_MISSING: "missing", S_IDLE: "idle", S_CANDIDATE: "candidate", S_STROKE: "stroke"}
# final label per sample (after the fact)
L_MISSING, L_IDLE, L_REJECTED, L_ACCEPTED, L_OUTSIDE = 0, 1, 2, 3, 4
LABEL_NAMES = {L_MISSING: "missing", L_IDLE: "idle", L_REJECTED: "candidate_rejected",
               L_ACCEPTED: "accepted", L_OUTSIDE: "outside_region"}


@dataclass
class Stroke:
    indices: list[int]
    accepted: bool
    end_reason: str
    displacement: float = 0.0   # screen-h upward travel (start y - peak y)
    peak_speed: float = 0.0     # screen-h/s
    opened_at: int = -1         # sample index where the candidate opened
    accepted_at: int = -1       # sample index where it qualified (-1 if never online)
    extra: dict = field(default_factory=dict)


def upward_speed(t: np.ndarray, y: np.ndarray, seg: np.ndarray, window: float) -> np.ndarray:
    """Upward speed (screen-h/s, positive = up) at each valid sample, measured over `window` seconds
    back in time within the same segment, interpolating y at t - window. NaN where invalid."""
    N = len(t)
    v = np.full(N, np.nan)
    valid = np.flatnonzero(seg >= 0)
    if len(valid) == 0:
        return v
    # process each segment independently
    seg_v = seg[valid]
    bounds = np.flatnonzero(np.diff(seg_v)) + 1
    for part in np.split(valid, bounds):
        ts, ys = t[part], y[part]
        target = ts - window
        j = np.searchsorted(ts, target, side="right") - 1  # last sample at or before target
        for n in range(len(part)):
            if n == 0:
                v[part[n]] = 0.0
                continue
            jj = j[n]
            if jj < 0:            # segment too young: use its first sample
                y0, t0 = ys[0], ts[0]
            elif ts[jj] == target[n] or jj + 1 >= len(part):
                y0, t0 = ys[jj], ts[jj]
            else:                 # interpolate between jj and jj+1
                a = (target[n] - ts[jj]) / (ts[jj + 1] - ts[jj])
                y0, t0 = ys[jj] + a * (ys[jj + 1] - ys[jj]), target[n]
            dt = ts[n] - t0
            v[part[n]] = (y0 - ys[n]) / dt if dt > 0 else 0.0
    return v


def detect_strokes(t: np.ndarray, xy: np.ndarray, seg: np.ndarray, dcfg: dict):
    """Run the state machine. Returns (strokes, online_state (N,), final_label (N,), v_up (N,))."""
    N = len(t)
    y = xy[:, 1]
    tol = float(dcfg["boundary_tolerance"])
    start_v, end_v = float(dcfg["start_speed"]), float(dcfg["end_speed"])
    min_disp, min_v = float(dcfg["min_displacement"]), float(dcfg["min_speed"])
    rb_win, min_net = float(dcfg["rebound_window_s"]), float(dcfg["min_net_rise"])
    max_start_y = float(dcfg["max_start_y"])
    pause_tol, rev_d, rev_t = float(dcfg["pause_tolerance_s"]), float(dcfg["reversal_distance"]), float(dcfg["reversal_tolerance_s"])
    max_gap, backtrack = float(dcfg["max_gap_s"]), float(dcfg["backtrack_s"])

    v = upward_speed(t, y, seg, float(dcfg["velocity_window_s"]))
    online = np.full(N, S_MISSING, np.int8)
    label = np.full(N, L_MISSING, np.int8)
    valid = seg >= 0
    label[valid] = L_IDLE
    in_region = valid & np.all((xy >= -tol) & (xy <= 1 + tol), axis=1)
    label[valid & ~in_region] = L_OUTSIDE

    strokes: list[Stroke] = []
    state = S_IDLE
    buf: list[int] = []
    cur: Stroke | None = None
    stall = down = 0.0
    last_valid = -1
    last_end = -1  # samples up to here belong to a finished stroke; never reused

    def pre_level(i0):
        """y where the thumb was rb_win seconds before sample i0 (same segment; earliest available)."""
        k, best = i0, i0
        while k > 0:
            k -= 1
            if not valid[k]:
                if t[i0] - t[k] > rb_win:
                    break
                continue
            if seg[k] != seg[i0]:
                break
            best = k
            if t[i0] - t[k] >= rb_win:
                break
        return y[best]

    def qualifies(idx):
        ys = y[idx]
        # end at the FIRST arrival at the top (holding still afterwards is not part of the stroke)
        p = int(np.flatnonzero(ys <= ys.min() + 2e-3)[0])
        disp = ys[0] - ys[p]
        pv = float(np.nanmax(v[idx[: p + 1]])) if p >= 0 else 0.0
        # rebound check: the stroke must rise above where the thumb was shortly BEFORE it began,
        # so a brief downward dip (tracking glitch or twitch) and its snap-back is not a swipe
        net = pre_level(int(idx[0])) - ys[p]
        ok = disp >= min_disp and pv >= min_v and net >= min_net and ys[0] <= max_start_y
        return ok, p, disp, pv

    def finish(reason: str, end_at: int):
        nonlocal state, buf, cur, last_end
        idx = np.array(buf)
        ok, p, disp, pv = qualifies(idx)
        kept = idx[: p + 1]
        cur.indices, cur.accepted, cur.end_reason = kept.tolist(), bool(ok), reason
        cur.displacement, cur.peak_speed = float(disp), float(pv)
        label[kept] = L_ACCEPTED if ok else L_REJECTED
        strokes.append(cur)
        last_end = int(kept[-1])
        state, buf, cur = S_IDLE, [], None

    for i in range(N):
        if not valid[i]:
            if state != S_IDLE and last_valid >= 0 and t[i] - t[last_valid] > max_gap:
                finish("tracking_lost", i)
            online[i] = S_MISSING
            continue
        if state != S_IDLE and seg[i] != seg[last_valid]:
            finish("tracking_lost", i)
        dt = t[i] - t[last_valid] if last_valid >= 0 and seg[last_valid] == seg[i] else 0.0
        last_valid = i
        if not in_region[i]:
            if state != S_IDLE:
                finish("left_region", i)
            online[i] = S_IDLE
            continue

        if state == S_IDLE:
            if v[i] > start_v:
                # look back for the lowest point (largest y) where this rise began
                j = i
                while True:
                    k = j - 1
                    while k > last_end and not valid[k]:
                        k -= 1
                    if (k <= last_end or seg[k] != seg[i] or not in_region[k]
                            or t[i] - t[k] > backtrack or y[k] < y[j] - 1e-3):
                        break
                    j = k
                cand = [n for n in range(j, i + 1) if valid[n]]
                # begin at the LAST sample at the bottom (resting samples before the rise are dropped)
                y_low = max(y[n] for n in cand)
                first = max(n for n in cand if y[n] >= y_low - 2e-3)
                buf = [n for n in cand if n >= first]
                cur = Stroke(indices=[], accepted=False, end_reason="", opened_at=i)
                state, stall, down = S_CANDIDATE, 0.0, 0.0
                online[i] = S_CANDIDATE
                if qualifies(np.array(buf))[0]:
                    state, cur.accepted_at = S_STROKE, i
                    online[i] = S_STROKE
            else:
                online[i] = S_IDLE
            continue

        # CANDIDATE or STROKE
        buf.append(i)
        if v[i] < -end_v:
            down += dt
            stall = 0.0
        elif v[i] <= end_v:
            stall += dt
            down = 0.0
        else:
            stall = down = 0.0
        peak_y = float(np.min(y[buf]))
        if y[i] - peak_y > rev_d or down > rev_t:
            online[i] = state
            finish("reversal", i)
            online[i] = S_IDLE
            continue
        if stall > pause_tol:
            online[i] = state
            finish("pause", i)
            continue
        if state == S_CANDIDATE and qualifies(np.array(buf))[0]:
            state, cur.accepted_at = S_STROKE, i
        online[i] = state

    if state != S_IDLE:
        finish("end_of_data", N - 1)
    return strokes, online, label, v
