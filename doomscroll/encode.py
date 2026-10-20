"""H.264 encoding with PyAV (libx264), streaming frame by frame; optional audio mux via ffmpeg."""
from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from .common import PipelineError


class VideoWriter:
    """Write RGB frames to an .mp4.

    Constant rate:  VideoWriter(path, w, h, fps=30); write(img)
    Variable rate:  VideoWriter(path, w, h, fps=avg, time_base=Fraction(1, 100000)); write(img, pts=...)
    """

    def __init__(self, path: str | Path, width: int, height: int, fps: float,
                 time_base: Fraction | None = None, crf: int = 18, preset: str = "medium"):
        if width % 2 or height % 2:
            raise PipelineError(f"output size must be even for yuv420p, got {width}x{height}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.c = av.open(str(self.path), mode="w")
            rate = Fraction(fps).limit_denominator(100000)
            self.s = self.c.add_stream("libx264", rate=rate)
            self.s.width, self.s.height, self.s.pix_fmt = width, height, "yuv420p"
            self.s.options = {"crf": str(crf), "preset": preset}
            self.tb = time_base or Fraction(1) / rate
            self.s.codec_context.time_base = self.tb
            self.s.time_base = self.tb
        except av.error.FFmpegError as e:
            raise PipelineError(f"could not create encoder for {self.path}: {e}") from e
        self.n = 0
        self.last_pts = -1

    def write(self, rgb: np.ndarray, pts: int | None = None) -> None:
        if pts is None:
            pts = self.n
        if pts <= self.last_pts:
            pts = self.last_pts + 1
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        frame.pts, frame.time_base = pts, self.tb
        try:
            for pkt in self.s.encode(frame):
                self.c.mux(pkt)
        except av.error.FFmpegError as e:
            raise PipelineError(f"encoding failed for {self.path}: {e}") from e
        self.last_pts = pts
        self.n += 1

    def close(self) -> None:
        try:
            for pkt in self.s.encode():
                self.c.mux(pkt)
            self.c.close()
        except av.error.FFmpegError as e:
            raise PipelineError(f"finalizing {self.path} failed: {e}") from e


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise PipelineError(f"bundled ffmpeg not available (pip install imageio-ffmpeg): {e}") from e


def mux_source_audio(video_only: Path, source: str, start_s: float, end_s: float, out: Path) -> bool:
    """Copy the source audio for [start_s, end_s) under an encoded video. Returns False on failure."""
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video_only),
           "-ss", f"{start_s:.6f}", "-to", f"{end_s:.6f}", "-i", str(source),
           "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
           "-shortest", "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"warning: could not add source audio ({r.stderr.strip()[:300]}); keeping silent diagnostic")
        return False
    return True
