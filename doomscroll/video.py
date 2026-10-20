"""Video inspection and decoding with real presentation timestamps (PyAV).

All source times in this project are *source-relative seconds*: the presentation timestamp of a
video frame minus the video stream's start_time. t = 0 is the first video frame of the recording.
Frames are never assumed to be evenly spaced.

Orientation: frames are returned in display orientation. If the stream carries a display-matrix
rotation (common for iPhone footage) it is applied here, so every later stage (calibration,
tracking, diagnostics) works in the same upright image coordinates. Mirroring is not altered:
the calibration corner order (phone TL, TR, BR, BL) absorbs any mirroring in the homography.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import numpy as np

from .common import PipelineError, require_file


@dataclass
class VideoInfo:
    path: str
    width: int                 # display (rotated) width
    height: int                # display (rotated) height
    coded_width: int
    coded_height: int
    rotation_deg: int          # FFmpeg display-matrix rotation (counter-clockwise degrees)
    time_base: Fraction
    start_pts: int
    n_frames_header: int
    avg_fps: float
    duration_s: float          # end of last video frame, source-relative (from header)
    has_audio: bool
    audio_start_s: float | None  # audio start relative to the video start (s)
    codec: str
    container_metadata: dict

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["time_base"] = f"{self.time_base.numerator}/{self.time_base.denominator}"
        return d


def open_video(path: str | Path) -> av.container.InputContainer:
    p = require_file(path, "source video", "put recordings in footage/ and pass --video footage/<file>.mov")
    try:
        return av.open(str(p))
    except av.error.FFmpegError as e:
        raise PipelineError(f"could not open video {p}: {e}") from e


def _rotation_of(frame: av.VideoFrame) -> int:
    rot = getattr(frame, "rotation", 0) or 0
    return int(round(rot / 90.0)) * 90 % 360


def orient(img: np.ndarray, rotation_deg: int) -> np.ndarray:
    """Rotate a decoded frame into display orientation.

    FFmpeg reports the display matrix rotation as counter-clockwise degrees; displaying the
    frame correctly means rotating the pixels counter-clockwise by that amount (this matches
    ffmpeg's own autorotate: rotation -90 -> transpose clockwise).
    """
    k = (rotation_deg // 90) % 4
    return np.ascontiguousarray(np.rot90(img, k)) if k else img


def probe(path: str | Path) -> VideoInfo:
    c = open_video(path)
    try:
        if not c.streams.video:
            raise PipelineError(f"{path} has no video stream")
        v = c.streams.video[0]
        tb = v.time_base
        start = v.start_time or 0
        rotation = 0
        for frame in c.decode(v):
            rotation = _rotation_of(frame)
            break
        cw, ch = v.codec_context.width, v.codec_context.height
        w, h = (ch, cw) if rotation in (90, 270) else (cw, ch)
        dur = float(v.duration * tb) if v.duration else (c.duration / 1e6 if c.duration else float("nan"))
        audio_start = None
        if c.streams.audio:
            a = c.streams.audio[0]
            if a.start_time is not None:
                audio_start = float(a.start_time * a.time_base) - float(start * tb)
        return VideoInfo(
            path=str(path), width=w, height=h, coded_width=cw, coded_height=ch,
            rotation_deg=rotation, time_base=tb, start_pts=start,
            n_frames_header=v.frames, avg_fps=float(v.average_rate) if v.average_rate else float("nan"),
            duration_s=dur, has_audio=bool(c.streams.audio), audio_start_s=audio_start,
            codec=v.codec_context.name, container_metadata=dict(c.metadata),
        )
    finally:
        c.close()


def scan_timestamps(path: str | Path) -> np.ndarray:
    """Source-relative presentation times (s) of every video frame, from packets (fast, no decode)."""
    c = open_video(path)
    try:
        v = c.streams.video[0]
        start = v.start_time or 0
        pts = [p.pts for p in c.demux(v) if p.pts is not None]
        return (np.sort(np.array(pts, dtype=np.int64)) - start) * float(v.time_base)
    finally:
        c.close()


@dataclass
class Frame:
    index: int        # decode order index within this iteration (0 = first frame yielded)
    pts: int          # raw stream pts
    t: float          # source-relative seconds
    image: np.ndarray # HxWx3 uint8, display orientation, RGB or BGR as requested


def iter_frames(path: str | Path, start_s: float = 0.0, end_s: float | None = None,
                fmt: str = "rgb24") -> Iterator[Frame]:
    """Decode frames with start_s <= t < end_s in presentation order, in display orientation."""
    c = open_video(path)
    try:
        v = c.streams.video[0]
        v.thread_type = "AUTO"
        tb = v.time_base
        start = v.start_time or 0
        if start_s > 0:
            c.seek(int(start_s / tb) + start, stream=v, backward=True, any_frame=False)
        i = 0
        for frame in c.decode(v):
            if frame.pts is None:
                continue
            t = float((frame.pts - start) * tb)
            if t < start_s - 1e-9:
                continue
            if end_s is not None and t >= end_s:
                break
            img = orient(frame.to_ndarray(format=fmt), _rotation_of(frame))
            yield Frame(index=i, pts=frame.pts, t=t, image=img)
            i += 1
    except av.error.FFmpegError as e:
        raise PipelineError(f"decoding failed for {path}: {e}") from e
    finally:
        c.close()


def frame_at(path: str | Path, t: float, fmt: str = "bgr24") -> Frame:
    """First frame at or after source time t."""
    for fr in iter_frames(path, start_s=t, fmt=fmt):
        return fr
    raise PipelineError(f"no frame at or after t={t:.3f}s in {path}")
