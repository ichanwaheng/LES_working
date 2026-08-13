#!/usr/bin/env python3
"""Form-find a membrane, then capture continuous submerged motion as video.

Stage 1 uses ``Coupling.QuasiStaticFSI`` (unchanged) until the membrane has
found its formed shape. Stage 2 lives entirely under ``continuous_motion/``
and writes a simulation video of the membrane moving in the fluid flow.

Usage
-----
    python -m continuous_motion.run_video --quick
    python -m continuous_motion.run_video --t-end 1.5 --fps 12
    python continuous_motion/run_video.py --config continuous_motion/default_config.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Coupling import QuasiStaticFSI
from io_helper import ensure_dir, load_config, save_snapshot
from continuous_motion.capture import ContinuousMotionCapture


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=(
            "Form-find membrane, then record continuous motion video "
            "while submerged in fluid flow"
        )
    )
    p.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(here / "default_config.yaml"),
        help="YAML config (default: continuous_motion/default_config.yaml)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Coarse mesh / short interval smoke video",
    )
    p.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="Continuous-motion duration [s] after form-find (overrides config)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Video frames per second (default: config simulation.video_fps)",
    )
    p.add_argument(
        "--amplify",
        type=float,
        default=None,
        help="Amplified video scale (1.0 skips amplified file)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: config simulation.output_dir)",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Cap continuous-motion outer frames (useful with --quick)",
    )
    args, _unknown = p.parse_known_args()
    return args


def _apply_quick(cfg: dict) -> None:
    cfg["membrane"]["nx"] = 8
    cfg["membrane"]["ny"] = 6
    cfg["fluid"]["nx"] = 16
    cfg["fluid"]["ny"] = 8
    cfg["fluid"]["nz"] = 8
    cfg["fluid"]["nu"] = 5.0e-3
    cfg["time"]["dt"] = 0.01
    cfg.setdefault("quasi_static", {})
    cfg["quasi_static"]["max_iters"] = 3
    cfg["quasi_static"]["fluid_substeps"] = 4
    cfg["quasi_static"]["load_scale"] = 0.2
    cfg.setdefault("continuous_motion", {})
    cfg["continuous_motion"]["t_end"] = 0.4
    cfg["continuous_motion"]["fluid_substeps"] = 2
    cfg["continuous_motion"]["structure_substeps"] = 2
    cfg["continuous_motion"]["load_scale"] = 0.35
    cfg["les"]["enabled"] = True
    cfg["simulation"]["save_interval"] = 1


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.quick:
        _apply_quick(cfg)

    cmcfg = cfg.setdefault("continuous_motion", {})
    if args.t_end is not None:
        cmcfg["t_end"] = float(args.t_end)

    out_rel = (
        args.out_dir
        if args.out_dir is not None
        else cfg["simulation"].get("output_dir", "continuous_motion/output")
    )
    out = ensure_dir(ROOT / out_rel if not Path(out_rel).is_absolute() else out_rel)

    print(
        f"[CM] form-find UWM {cfg['membrane']['nx']}x{cfg['membrane']['ny']}, "
        f"fluid {cfg['fluid']['nx']}x{cfg['fluid']['ny']}x{cfg['fluid']['nz']}"
    )

    # ----- Stage 1: form-find (membrane finds its formed shape) ----------
    sim = QuasiStaticFSI(cfg)

    def on_form(sim_obj, info, k):
        print(
            f"[CM:form] iter={int(info['iteration']):3d}  "
            f"t={info['time']:.3f}s  max|u|={info['max_disp']:.4e}  "
            f"shape_res={info['shape_residual']:.3e}"
        )

    sim.run(callback=on_form)
    print(
        f"[CM] form found at t={sim.time:.3f}s, "
        f"max|u|={np.max(np.linalg.norm(sim.nodes - sim.x_bc, axis=1)):.4e} m"
    )
    save_snapshot(
        out,
        step=0,
        time=sim.time,
        membrane_nodes=sim.nodes,
        membrane_elements=sim.mesh.elements,
        meta={"stage": "form_found"},
    )

    # ----- Stage 2: continuous motion while submerged in flow -----------
    cap = ContinuousMotionCapture(sim)
    save_every = int(cfg["simulation"].get("save_interval", 1))

    def on_motion(sim_obj, info, k):
        print(
            f"[CM:motion] frame={int(info['iteration']):4d}  "
            f"t={info['time']:.3f}s  "
            f"Δform={info['max_disp_from_form']:.4e}  "
            f"|v|_max={info['max_speed']:.4e}  "
            f"KE={info['kinetic']:.4e}"
        )
        if k % save_every == 0:
            save_snapshot(
                out,
                step=int(info["iteration"]),
                time=info["time"],
                membrane_nodes=sim_obj.nodes,
                membrane_elements=sim_obj.mesh.elements,
                fluid_u=sim_obj.fluid.state.u,
                meta={"stage": "continuous_motion"},
            )

    hist = cap.run(
        t_end=float(cmcfg.get("t_end", cfg["time"]["t_end"])),
        n_steps=args.n_steps,
        callback=on_motion,
    )

    # History CSV
    csv_path = out / "continuous_motion_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "iteration",
                "time_s",
                "max_disp_m",
                "max_disp_from_form_m",
                "max_speed_m_s",
                "kinetic_J",
                "pressure_max_Pa",
                "cfl",
            ]
        )
        for i in range(len(hist.time)):
            w.writerow(
                [
                    hist.iteration[i],
                    hist.time[i],
                    hist.max_disp[i],
                    hist.max_disp_from_form[i],
                    hist.max_speed[i],
                    hist.kinetic[i],
                    hist.pressure_max[i],
                    hist.cfl[i],
                ]
            )
    print(f"[CM] history → {csv_path}")

    # ----- Simulation video (primary output) -----------------------------
    fps = int(
        args.fps
        if args.fps is not None
        else cfg["simulation"].get("video_fps", 12)
    )
    amplify = float(
        args.amplify
        if args.amplify is not None
        else cfg["simulation"].get("video_amplify", 5.0)
    )
    video_name = cmcfg.get("video_name", "membrane_continuous_motion.mp4")
    video_path = out / video_name

    path, kind = cap.write_video(
        video_path,
        fps=fps,
        amplify=1.0,
        title="Formed membrane in fluid flow",
    )
    print(f"[CM] simulation video ({kind}) → {path}")

    if amplify != 1.0:
        amp_path = video_path.with_name(
            video_path.stem + "_amplified" + video_path.suffix
        )
        path_a, kind_a = cap.write_video(
            amp_path,
            fps=fps,
            amplify=amplify,
            title=f"Formed membrane in fluid flow (×{amplify:g})",
        )
        print(f"[CM] amplified video ({kind_a}) → {path_a}")

    print(f"[CM] done — {len(hist.nodes)} frames in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
