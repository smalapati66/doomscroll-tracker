"""SYNTHETIC gesture generator, only for exercising the renderer (e.g. long-term saturation).

Output is clearly labeled synthetic and is never mixed with tracked data.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

from .common import write_json


def synthetic_gestures(duration_s: float, mean_interval_s: float, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out, t, gid = [], 1.0, 1
    while True:
        t += rng.exponential(mean_interval_s)
        if rng.random() < 0.08:                 # occasional long pause
            t += rng.uniform(15, 60)
        if t > duration_s - 1:
            break
        # distribution modeled on tracked swipes in footage/Untitled.mov (narrow column, x ~ 0.75)
        dur = rng.uniform(0.2, 0.45)
        x0, y0 = rng.normal(0.75, 0.03), rng.uniform(0.74, 0.86)
        dy = rng.uniform(0.25, 0.48)
        dx = rng.normal(0.06, 0.06)
        n = max(3, int(dur * 30))
        u = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
        pts = [{"t": round(t + dur * k / (n - 1), 4), "x": round(float(x0 + dx * u[k]), 4),
                "y": round(float(y0 - dy * u[k]), 4)} for k in range(n)]
        out.append({"id": gid, "start_s": pts[0]["t"], "end_s": pts[-1]["t"], "points": pts,
                    "synthetic": True})
        gid += 1
        t += dur
    return out


def write_synthetic(out: Path, duration_s: float, mean_interval_s: float, seed: int) -> Path:
    g = synthetic_gestures(duration_s, mean_interval_s, seed)
    write_json(out, {
        "schema": "doomscroll-gestures/1",
        "SYNTHETIC": "THIS FILE IS SYNTHETIC TEST DATA - NOT TRACKED FROM FOOTAGE",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "video_duration_s": duration_s,
        "calibration": {"screen_aspect": 9 / 19.5},
        "generator": {"mean_interval_s": mean_interval_s, "seed": seed},
        "gestures": g,
    })
    print(f"wrote {out}: {len(g)} SYNTHETIC gestures over {duration_s:.0f} s")
    return out
