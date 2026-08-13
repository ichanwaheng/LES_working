"""Write simulation video from recorded continuous-motion frames.

Primary output is MP4 via ffmpeg. Falls back to animated GIF when ffmpeg
or image codecs are unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _render_frames(
    nodes_over_time: Sequence[np.ndarray],
    elements: np.ndarray,
    nodes0: np.ndarray,
    times: Sequence[float],
    *,
    amplify: float = 1.0,
    title: str = "Formed membrane in fluid flow",
    fluid_slices: Optional[Sequence[np.ndarray]] = None,
    grid_x: Optional[np.ndarray] = None,
    grid_z: Optional[np.ndarray] = None,
    mesh_nx: Optional[int] = None,
    mesh_ny: Optional[int] = None,
) -> List:
    """Render PIL RGB frames (flutter layout when fluid slices are provided)."""
    from viz import render_deformation_frame, render_flutter_frame

    nodes0 = np.asarray(nodes0, dtype=float)
    elements = np.asarray(elements, dtype=int)
    amp = float(amplify)
    n = len(nodes_over_time)
    if len(times) != n:
        raise ValueError("times and nodes_over_time length mismatch")

    shaped: List[np.ndarray] = []
    disp_max = 0.0
    z_vals = [float(nodes0[:, 2].min()), float(nodes0[:, 2].max())]
    for nodes in nodes_over_time:
        nodes = np.asarray(nodes, dtype=float)
        disp = nodes - nodes0
        s = nodes0 + amp * disp
        shaped.append(s)
        disp_max = max(disp_max, float(np.linalg.norm(disp, axis=1).max()))
        z_vals.append(float(s[:, 2].min()))
        z_vals.append(float(s[:, 2].max()))

    z_min, z_max = min(z_vals), max(z_vals)
    z_pad = max(0.05 * max(z_max - z_min, disp_max, 1e-6), 1e-4)
    z_limits = (z_min - z_pad, z_max + z_pad)
    if disp_max <= 0.0:
        disp_max = 1e-6

    use_fluid = (
        fluid_slices is not None
        and grid_x is not None
        and grid_z is not None
        and mesh_nx is not None
        and mesh_ny is not None
        and len(fluid_slices) == n
    )

    frames = []
    if use_fluid:
        speed_max = max(
            float(np.max(s)) for s in fluid_slices  # type: ignore[arg-type]
        )
        speed_max = max(speed_max, 1e-6)
        for i in range(n):
            frames.append(
                render_flutter_frame(
                    nodes=shaped[i],
                    elements=elements,
                    nodes0=nodes0,
                    speed_slice=np.asarray(fluid_slices[i], dtype=float),
                    grid_x=np.asarray(grid_x, dtype=float),
                    grid_z=np.asarray(grid_z, dtype=float),
                    time=float(times[i]),
                    mesh_nx=int(mesh_nx),
                    mesh_ny=int(mesh_ny),
                    speed_max=speed_max,
                    disp_max=disp_max * max(amp, 1.0),
                    z_limits=z_limits,
                    title=title,
                )
            )
    else:
        for i in range(n):
            frames.append(
                render_deformation_frame(
                    nodes=shaped[i],
                    elements=elements,
                    nodes0=nodes0,
                    time=float(times[i]),
                    disp_max=disp_max * max(amp, 1.0),
                    z_limits=z_limits,
                    title=title,
                )
            )
    return frames


def _write_mp4_ffmpeg(frames: Sequence, out_path: Path, fps: int) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cm_video_") as tmp:
        tmp_dir = Path(tmp)
        for i, frame in enumerate(frames):
            frame.save(tmp_dir / f"frame_{i:06d}.png")
        pattern = str(tmp_dir / "frame_%06d.png")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(max(int(fps), 1)),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        if proc.returncode != 0 or not out_path.is_file():
            raise RuntimeError(
                "ffmpeg failed to write MP4:\n"
                + (proc.stderr or proc.stdout or "unknown error")
            )
    return out_path


def _write_gif_fallback(frames: Sequence, out_path: Path, fps: int) -> Path:
    from viz import save_gif

    gif_path = out_path.with_suffix(".gif")
    return save_gif(frames, gif_path, fps=fps)


def write_simulation_video(
    nodes_over_time: Sequence[np.ndarray],
    elements: np.ndarray,
    nodes0: np.ndarray,
    out_path: str | Path,
    times: Optional[Sequence[float]] = None,
    fps: int = 12,
    amplify: float = 1.0,
    title: str = "Formed membrane in fluid flow",
    fluid_slices: Optional[Sequence[np.ndarray]] = None,
    grid_x: Optional[np.ndarray] = None,
    grid_z: Optional[np.ndarray] = None,
    mesh_nx: Optional[int] = None,
    mesh_ny: Optional[int] = None,
) -> Tuple[Path, str]:
    """Render continuous-motion frames and write MP4 (GIF fallback).

    Returns
    -------
    path, kind
        ``kind`` is ``\"mp4\"`` or ``\"gif\"``.
    """
    if not nodes_over_time:
        raise ValueError("no frames to write — run capture first")

    out_path = Path(out_path)
    if times is None:
        times = list(range(len(nodes_over_time)))

    frames = _render_frames(
        nodes_over_time,
        elements,
        nodes0,
        times,
        amplify=amplify,
        title=title,
        fluid_slices=fluid_slices,
        grid_x=grid_x,
        grid_z=grid_z,
        mesh_nx=mesh_nx,
        mesh_ny=mesh_ny,
    )

    if out_path.suffix.lower() not in {".mp4", ".gif"}:
        out_path = out_path.with_suffix(".mp4")

    if out_path.suffix.lower() == ".gif":
        path = _write_gif_fallback(frames, out_path, fps=fps)
        return path, "gif"

    try:
        path = _write_mp4_ffmpeg(frames, out_path, fps=fps)
        return path, "mp4"
    except Exception as exc:  # noqa: BLE001 — fall back for portability
        print(f"[continuous_motion] MP4 failed ({exc}); writing GIF instead")
        path = _write_gif_fallback(frames, out_path, fps=fps)
        return path, "gif"
