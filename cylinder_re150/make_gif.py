#!/usr/bin/env python3
"""Build an animated wake GIF from ``wake_t*.png`` frames.

Does **not** re-run the CFD by default. Point ``--out-dir`` at a finished
``cylinder_re150`` run folder (or use the default ``output/``).

Usage
-----
    cd cylinder_re150
    python3 make_gif.py
    python3 make_gif.py --out-dir output --fps 8
    python3 make_gif.py --out-dir output --every 2 --max-width 700
    python3 make_gif.py --run-quick   # run a short sim first, then GIF

After a normal solve you can also get a GIF from::

    python3 run.py --quick
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from viz import save_wake_gif, wake_frames_from_dir


def parse_args():
    p = argparse.ArgumentParser(
        description="Assemble cylinder wake PNGs into cylinder_wake.gif"
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(HERE / "output"),
        help="Directory containing wake_t*.png (default: cylinder_re150/output)",
    )
    p.add_argument(
        "--gif",
        type=str,
        default=None,
        help="Output GIF path (default: <out-dir>/cylinder_wake.gif)",
    )
    p.add_argument("--fps", type=int, default=8, help="Frames per second")
    p.add_argument(
        "--every",
        type=int,
        default=1,
        help="Keep every N-th frame (e.g. 2 or 4 for a smaller GIF)",
    )
    p.add_argument(
        "--max-width",
        type=int,
        default=900,
        help="Max GIF width in pixels (0 = no resize)",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Also copy the GIF to demo/cylinder_wake.gif",
    )
    p.add_argument(
        "--run-quick",
        action="store_true",
        help="Run `python run.py --quick` into --out-dir before building the GIF",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (HERE / out_dir).resolve()

    if args.run_quick:
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(HERE / "run.py"),
            "--quick",
            "--out-dir",
            str(out_dir),
            "--gif-fps",
            str(args.fps),
        ]
        print(f"[make_gif] running: {' '.join(cmd)}")
        subprocess.check_call(cmd, cwd=str(HERE))

    frames = wake_frames_from_dir(out_dir)
    if not frames:
        print(
            f"[make_gif] ERROR: no wake_t*.png in {out_dir}\n"
            "  Run a simulation first:\n"
            "    python3 run.py --quick\n"
            "  or:\n"
            "    python3 make_gif.py --run-quick"
        )
        return 1

    every = max(int(args.every), 1)
    if every > 1:
        frames = frames[::every]
        print(f"[make_gif] subsampled every {every} → {len(frames)} frames")

    gif_path = Path(args.gif) if args.gif else out_dir / "cylinder_wake.gif"
    if not gif_path.is_absolute():
        gif_path = (HERE / gif_path).resolve()

    print(f"[make_gif] {len(frames)} frames  →  {gif_path}  @ {args.fps} fps")
    save_wake_gif(
        frames,
        gif_path,
        fps=int(args.fps),
        max_width=int(args.max_width) if args.max_width > 0 else 0,
    )
    size_mb = gif_path.stat().st_size / 1e6
    print(f"[make_gif] wrote {gif_path}  ({size_mb:.2f} MB)")

    if args.demo:
        demo = HERE / "demo" / "cylinder_wake.gif"
        demo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gif_path, demo)
        print(f"[make_gif] demo copy → {demo}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
