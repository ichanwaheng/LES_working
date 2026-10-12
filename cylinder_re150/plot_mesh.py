#!/usr/bin/env python3
"""Visualise geometry + mesh after BL meshing, before analysis.

Writes:
  - mesh_geometry.png  — domain mesh + cylinder (and zoom)
  - mesh_spacing.png   — Δx(x), Δy(y)

Usage
-----
    cd cylinder_re150
    python3 plot_mesh.py --quick
    python3 plot_mesh.py                 # production mesh inputs
    python3 run.py --mesh-only --quick   # same, via the solver driver
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cylinder import Cylinder
from mesh import FluidGrid
from viz import plot_mesh_geometry, plot_mesh_spacing


def build_mesh_from_config(cfg: dict, quick: bool = False):
    phys, dom = cfg["physical"], cfg["domain"]
    mcfg = dict(cfg.get("mesh", {}))
    if quick:
        q = cfg.get("quick", {})
        for k in ("n_bl", "growth", "h_far"):
            if k in q:
                mcfg[k] = q[k]

    h_far = mcfg.get("h_far", None)
    grid = FluidGrid.from_boundary_layer(
        L=float(dom["L"]),
        W=float(dom["W"]),
        cx=float(dom["cx"]),
        cy=float(dom["cy"]),
        D=float(phys["D"]),
        Re=float(phys["Re"]),
        n_bl=int(mcfg.get("n_bl", 15)),
        growth=float(mcfg.get("growth", 1.15)),
        h_far=float(h_far) if h_far is not None else None,
        verbose=True,
    )
    cyl = Cylinder(
        D=float(phys["D"]),
        cx=float(dom["cx"]),
        cy=float(dom["cy"]),
    )
    mask = cyl.cell_mask(grid)
    return grid, cyl, mask


def main() -> int:
    p = argparse.ArgumentParser(
        description="Show meshed cylinder geometry before analysis"
    )
    p.add_argument("-c", "--config", default=str(HERE / "config.yaml"))
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output folder (default: <config output_dir>)",
    )
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    grid, cyl, mask = build_mesh_from_config(cfg, quick=args.quick)

    if args.out_dir:
        out = Path(args.out_dir)
        if not out.is_absolute():
            out = (HERE / out).resolve()
    else:
        out = Path(cfg["simulation"]["output_dir"])
        if not out.is_absolute():
            if out.parts and out.parts[0] == "cylinder_re150":
                out = (HERE.parent / out).resolve()
            else:
                out = (HERE / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    geo = plot_mesh_geometry(
        grid,
        mask,
        out / "mesh_geometry.png",
        cylinder=cyl,
        title="Meshed geometry (before analysis)",
    )
    sp = plot_mesh_spacing(
        grid,
        out / "mesh_spacing.png",
        cx=cyl.cx,
        cy=cyl.cy,
        R=cyl.R,
    )
    if grid.mesh_info is not None:
        (out / "mesh_report.json").write_text(
            json.dumps(grid.mesh_info, indent=2), encoding="utf-8"
        )

    print(f"[plot_mesh] geometry → {geo}")
    print(f"[plot_mesh] spacing  → {sp}")
    print(
        f"[plot_mesh] {grid.nx}×{grid.ny} cells, "
        f"solid={int(mask.sum())}  "
        f"(no flow solve — inspect mesh before analysis)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
