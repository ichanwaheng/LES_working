#!/usr/bin/env python3
"""LES/PISO flow past a circular cylinder at Re = 3900 (MAC staggered grid).

This folder is self-contained and independent of the membrane FSI code.

Usage
-----
    cd cylinder_re3900
    python run.py --quick          # coarse smoke run
    python run.py                  # config.yaml defaults
    python run.py -c config.yaml --t-end 50
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
from viz import plot_force_history, plot_midplane, save_wake_gif, wake_frames_from_dir


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(
        description="Staggered PISO + LES: circular cylinder at Re=3900"
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

    if quick:
        q = cfg.get("quick", {})
        for k in ("nx", "ny", "nz"):
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
        H=float(dom["H"]),
        nx=int(grd["nx"]),
        ny=int(grd["ny"]),
        nz=int(grd["nz"]),
    )
    cyl = Cylinder(D=D, cx=float(dom["cx"]), cy=float(dom["cy"]))
    mask = cyl.cell_mask(grid)
    band = cyl.surface_band(grid, n_cells=1.5)

    les_cfg = cfg.get("les", {})
    fluid = FluidSolver(
        grid,
        rho=rho,
        nu=nu,
        U_inf=U,
        use_les=bool(les_cfg.get("enabled", True)),
        Cs=float(les_cfg.get("Cs", 0.1)),
        n_correctors=int(les_cfg.get("piso_correctors", 2)),
    )
    fluid.set_immersed_boundary(mask)

    # Seed a tiny asymmetry so the wake can shed
    rng = np.random.default_rng(0)
    fluid.state.v += 1e-3 * U * rng.standard_normal(fluid.state.v.shape)

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
        # Resolve relative to repo root (parent of this folder) if path starts with cylinder_
        out = (HERE.parent / out).resolve() if out.parts[0] == "cylinder_re3900" else (HERE / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    dt = meta["dt"]
    t_end = meta["t_end"]
    n_steps = int(np.ceil(t_end / dt))
    D, U = meta["D"], meta["U"]

    print(
        f"[cylinder] Re={meta['Re']:g}  D={D:g}  U={U:g}  nu={meta['nu']:.6e}\n"
        f"[cylinder] grid {grid.nx}x{grid.ny}x{grid.nz}  "
        f"dx={grid.dx:.4f} dy={grid.dy:.4f} dz={grid.dz:.4f}\n"
        f"[cylinder] solid cells={int(mask.sum())}  "
        f"dt={dt:g}  steps={n_steps}  t_end={t_end:g}\n"
        f"[cylinder] output → {out}"
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
            uc, vc, wc = st.cell_centered_velocity()
            # Mask solid for speed stats
            fluid_cells = ~mask
            speed = np.sqrt(uc[fluid_cells] ** 2 + vc[fluid_cells] ** 2 + wc[fluid_cells] ** 2)
            umax = float(np.max(speed)) if speed.size else 0.0
            t = fluid.t
            tUD = t * U / D

            times.append(tUD)
            Cds.append(Cd)
            Cls.append(Cl)
            writer.writerow(
                [step, f"{t:.6e}", f"{tUD:.6e}", f"{Cd:.6e}", f"{Cl:.6e}", f"{cfl:.4e}", f"{umax:.4e}"]
            )

            if step % max(meta["save_interval"], 1) == 0 or step == n_steps:
                fcsv.flush()
                print(
                    f"  step={step:6d}/{n_steps}  tU/D={tUD:7.3f}  "
                    f"Cd={Cd:7.3f}  Cl={Cl:7.3f}  CFL={cfl:.3f}  |u|_max={umax:.3f}"
                )

            if step % max(meta["plot_interval"], 1) == 0 or step == n_steps:
                frame = out / f"wake_t{tUD:07.2f}.png"
                plot_midplane(
                    grid,
                    st,
                    mask,
                    frame,
                    title=f"Re={meta['Re']:g}  tU/D={tUD:.2f}",
                    cylinder=cyl,
                )
                wake_frames.append(frame)

            if cfl > 2.0 * meta["cfl_max"]:
                print(f"[cylinder] WARNING: CFL={cfl:.3f} large — consider reducing dt")

    plot_force_history(
        np.asarray(times),
        np.asarray(Cds),
        np.asarray(Cls),
        out / "force_history.png",
    )

    gif_path = None
    if not args.no_gif:
        frames_for_gif = [p for p in wake_frames if Path(p).is_file()]
        if not frames_for_gif:
            frames_for_gif = wake_frames_from_dir(out)
        if frames_for_gif:
            gif_path = save_wake_gif(
                frames_for_gif, out / "cylinder_wake.gif", fps=gif_fps
            )
            # Tracked demo copy so the GIF is visible in the repo
            demo_dir = HERE / "demo"
            demo_dir.mkdir(parents=True, exist_ok=True)
            demo_gif = demo_dir / "cylinder_wake.gif"
            demo_gif.write_bytes(gif_path.read_bytes())
            print(
                f"[cylinder] wake GIF → {gif_path}  "
                f"({len(frames_for_gif)} frames @ {gif_fps} fps)"
            )
            print(f"[cylinder] demo GIF → {demo_gif}")
        else:
            print("[cylinder] WARNING: no wake frames — GIF not written")

    np.savez_compressed(
        out / "final_state.npz",
        u=st.u,
        v=st.v,
        w=st.w,
        p=st.p,
        nu_eff=st.nu_eff,
        mask=mask,
        x=grid.x,
        y=grid.y,
        z=grid.z,
        Cd=np.asarray(Cds),
        Cl=np.asarray(Cls),
        tUD=np.asarray(times),
        meta=np.array([meta], dtype=object),
    )
    print(f"[cylinder] done. forces → {hist_path}")
    if gif_path is not None:
        print(f"[cylinder] gif → {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
