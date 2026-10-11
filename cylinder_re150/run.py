#!/usr/bin/env python3
"""Laminar PISO flow past a circular cylinder at Re = 150 (2-D MAC grid).

NASA GRC Wind-US validation Study #1 analogue:
https://www.grc.nasa.gov/www/wind/valid/lamcyl/Study1_files/Study1.html

Self-contained folder — independent of ``cylinder_re3900`` and the membrane FSI code.

Usage
-----
    cd cylinder_re150
    python run.py --quick              # coarse grid, short time + GIF
    python run.py                      # config.yaml defaults
    python run.py --t-end 120 --gif-fps 8
    python run.py --no-gif
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cylinder import Cylinder
from mesh import FluidGrid
from piso import FluidSolver
from validate import (
    CD_EXP,
    ST_EXP_MID,
    format_report,
    validate_history,
    write_report,
)
from viz import plot_force_history, plot_wake, save_wake_gif, wake_frames_from_dir


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(
        description="2-D laminar PISO: circular cylinder at Re=150 (NASA Study #1)"
    )
    p.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(HERE / "config.yaml"),
        help="YAML config path",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Coarse grid / short time from config.quick",
    )
    p.add_argument("--t-end", type=float, default=None, help="Override t_end")
    p.add_argument("--dt", type=float, default=None, help="Override dt")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: config simulation.output_dir)",
    )
    p.add_argument(
        "--gif-fps",
        type=int,
        default=None,
        help="Wake GIF frames per second (default: config simulation.gif_fps)",
    )
    p.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip writing wake GIF at the end",
    )
    args, _ = p.parse_known_args()
    return args


def build_case(cfg: dict, quick: bool = False):
    phys = cfg["physical"]
    dom = cfg["domain"]
    grd = dict(cfg["grid"])
    tcfg = dict(cfg["time"])
    scfg = dict(cfg["simulation"])
    sol = cfg.get("solver", {})

    if quick:
        q = cfg.get("quick", {})
        for k in ("nx", "ny"):
            if k in q:
                grd[k] = int(q[k])
        for k in ("dt", "t_end"):
            if k in q:
                tcfg[k] = float(q[k])
        for k in ("save_interval", "plot_interval"):
            if k in q:
                scfg[k] = int(q[k])

    D = float(phys["D"])
    U = float(phys["U_inf"])
    Re = float(phys["Re"])
    nu = U * D / Re
    rho = float(phys.get("rho", 1.0))

    grid = FluidGrid(
        L=float(dom["L"]),
        W=float(dom["W"]),
        nx=int(grd["nx"]),
        ny=int(grd["ny"]),
    )
    cyl = Cylinder(D=D, cx=float(dom["cx"]), cy=float(dom["cy"]))
    mask = cyl.cell_mask(grid)
    band = cyl.surface_band(grid, n_cells=1.5)

    fluid = FluidSolver(
        grid,
        rho=rho,
        nu=nu,
        U_inf=U,
        n_correctors=int(sol.get("piso_correctors", 2)),
        conv_scheme=str(sol.get("conv_scheme", "central")),
    )
    fluid.set_immersed_boundary(mask)

    # Seed a clear cross-flow asymmetry so vortex shedding starts promptly
    # (random noise alone is slow to amplify on coarse grids).
    cy = float(dom["cy"])
    cx = float(dom["cx"])
    y_v = np.arange(grid.ny + 1) * grid.dy
    Xv, Yv = np.meshgrid(grid.x, y_v, indexing="ij")
    wake = (Xv > cx) & (Xv < cx + 4.0 * D) & (np.abs(Yv - cy) < 2.0 * D)
    fluid.state.v[wake] += 0.05 * U * np.sin(np.pi * (Yv[wake] - cy) / (2.0 * D))
    rng = np.random.default_rng(0)
    fluid.state.v += 2e-3 * U * rng.standard_normal(fluid.state.v.shape)
    fluid.state.u, fluid.state.v = fluid._apply_bc(fluid.state.u, fluid.state.v)

    vcfg = cfg.get("validation", {})
    meta = {
        "D": D,
        "U": U,
        "Re": Re,
        "nu": nu,
        "dt": float(tcfg["dt"]),
        "t_end": float(tcfg["t_end"]),
        "save_interval": int(scfg.get("save_interval", 50)),
        "plot_interval": int(scfg.get("plot_interval", 50)),
        "cfl_max": float(tcfg.get("cfl_max", 0.5)),
        "settle_fraction": float(vcfg.get("settle_fraction", 0.60)),
    }
    return grid, cyl, mask, band, fluid, meta


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config))
    grid, cyl, mask, band, fluid, meta = build_case(cfg, quick=args.quick)

    if args.t_end is not None:
        meta["t_end"] = float(args.t_end)
    if args.dt is not None:
        meta["dt"] = float(args.dt)

    out = Path(args.out_dir) if args.out_dir else Path(cfg["simulation"]["output_dir"])
    if not out.is_absolute():
        if out.parts and out.parts[0] == "cylinder_re150":
            out = (HERE.parent / out).resolve()
        else:
            out = (HERE / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    dt = meta["dt"]
    t_end = meta["t_end"]
    n_steps = int(np.ceil(t_end / dt))
    D, U = meta["D"], meta["U"]

    print(
        f"[cylinder_re150] Re={meta['Re']:g}  D={D:g}  U={U:g}  nu={meta['nu']:.6e}\n"
        f"[cylinder_re150] grid {grid.nx}x{grid.ny}  "
        f"dx={grid.dx:.4f} dy={grid.dy:.4f}\n"
        f"[cylinder_re150] solid cells={int(mask.sum())}  "
        f"dt={dt:g}  steps={n_steps}  t_end={t_end:g}\n"
        f"[cylinder_re150] NASA Study #1 targets: Cd≈{CD_EXP}, St≈{ST_EXP_MID:.3f}\n"
        f"[cylinder_re150] output → {out}"
    )

    hist_path = out / "force_history.csv"
    times, Cds, Cls = [], [], []
    wake_frames: list[Path] = []
    gif_fps = (
        int(args.gif_fps)
        if args.gif_fps is not None
        else int(cfg["simulation"].get("gif_fps", 8))
    )

    with open(hist_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["step", "t", "tUD", "Cd", "Cl", "cfl", "max_speed"])

        for step in range(1, n_steps + 1):
            st = fluid.step(dt)
            Cd, Cl = fluid.force_coefficients(D, band)
            cfl = fluid.max_cfl(dt)
            uc, vc = st.cell_centered_velocity()
            fluid_cells = ~mask
            speed = np.sqrt(uc[fluid_cells] ** 2 + vc[fluid_cells] ** 2)
            umax = float(np.max(speed)) if speed.size else 0.0
            t = fluid.t
            tUD = t * U / D

            times.append(tUD)
            Cds.append(Cd)
            Cls.append(Cl)
            writer.writerow(
                [
                    step,
                    f"{t:.6e}",
                    f"{tUD:.6e}",
                    f"{Cd:.6e}",
                    f"{Cl:.6e}",
                    f"{cfl:.4e}",
                    f"{umax:.4e}",
                ]
            )

            if step % max(meta["save_interval"], 1) == 0 or step == n_steps:
                fcsv.flush()
                print(
                    f"  step={step:6d}/{n_steps}  tU/D={tUD:7.3f}  "
                    f"Cd={Cd:7.3f}  Cl={Cl:7.3f}  CFL={cfl:.3f}  |u|_max={umax:.3f}"
                )

            if step % max(meta["plot_interval"], 1) == 0 or step == n_steps:
                frame = out / f"wake_t{tUD:07.2f}.png"
                plot_wake(
                    grid,
                    st,
                    mask,
                    frame,
                    title=f"Re={meta['Re']:g}  tU/D={tUD:.2f}",
                    cylinder=cyl,
                )
                wake_frames.append(frame)

            if cfl > 2.0 * meta["cfl_max"]:
                print(f"[cylinder_re150] WARNING: CFL={cfl:.3f} large — consider reducing dt")

    t_arr = np.asarray(times)
    Cd_arr = np.asarray(Cds)
    Cl_arr = np.asarray(Cls)

    plot_force_history(
        t_arr,
        Cd_arr,
        Cl_arr,
        out / "force_history.png",
        Cd_ref=CD_EXP,
        St_ref=ST_EXP_MID,
    )

    t_start = meta["settle_fraction"] * float(t_arr[-1]) if t_arr.size else 0.0
    report = validate_history(t_arr, Cd_arr, Cl_arr, t_start=t_start)
    report_path = write_report(report, out / "validation_report.json")
    print(format_report(report))
    print(f"[cylinder_re150] validation → {report_path}")

    gif_path = None
    if not args.no_gif:
        frames_for_gif = [p for p in wake_frames if Path(p).is_file()]
        if not frames_for_gif:
            frames_for_gif = wake_frames_from_dir(out)
        if frames_for_gif:
            gif_path = save_wake_gif(
                frames_for_gif, out / "cylinder_wake.gif", fps=gif_fps
            )
            demo_dir = HERE / "demo"
            demo_dir.mkdir(parents=True, exist_ok=True)
            demo_gif = demo_dir / "cylinder_wake.gif"
            demo_gif.write_bytes(gif_path.read_bytes())
            print(
                f"[cylinder_re150] wake GIF → {gif_path}  "
                f"({len(frames_for_gif)} frames @ {gif_fps} fps)"
            )
            print(f"[cylinder_re150] demo GIF → {demo_gif}")
        else:
            print("[cylinder_re150] WARNING: no wake frames — GIF not written")

    np.savez_compressed(
        out / "final_state.npz",
        u=st.u,
        v=st.v,
        p=st.p,
        mask=mask,
        x=grid.x,
        y=grid.y,
        Cd=Cd_arr,
        Cl=Cl_arr,
        tUD=t_arr,
        meta=np.array([meta], dtype=object),
    )
    print(f"[cylinder_re150] done. forces → {hist_path}")
    if gif_path is not None:
        print(f"[cylinder_re150] gif → {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
