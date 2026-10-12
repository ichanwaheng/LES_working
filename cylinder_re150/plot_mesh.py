#!/usr/bin/env python3
"""Visualise BL-derived cell spacings for the cylinder mesh."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mesh import FluidGrid


def main() -> int:
    p = argparse.ArgumentParser(description="Plot BL mesh spacings")
    p.add_argument("-c", "--config", default=str(HERE / "config.yaml"))
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--out",
        default=str(HERE / "output" / "mesh_spacing.png"),
    )
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    phys, dom = cfg["physical"], cfg["domain"]
    mcfg = dict(cfg.get("mesh", {}))
    if args.quick:
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
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    axes[0].plot(grid.x, grid.dx_c, color="#1f4e79", lw=1.2)
    axes[0].axvline(dom["cx"] - 0.5 * phys["D"], color="k", ls="--", lw=0.8)
    axes[0].axvline(dom["cx"] + 0.5 * phys["D"], color="k", ls="--", lw=0.8)
    axes[0].set_ylabel("Δx")
    axes[0].set_xlabel("x")
    axes[0].set_title("Streamwise cell size (BL packing at cylinder walls)")

    axes[1].plot(grid.y, grid.dy_c, color="#8b3a2a", lw=1.2)
    axes[1].axvline(dom["cy"] - 0.5 * phys["D"], color="k", ls="--", lw=0.8)
    axes[1].axvline(dom["cy"] + 0.5 * phys["D"], color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("Δy")
    axes[1].set_xlabel("y")
    axes[1].set_title("Cross-stream cell size")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot_mesh] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
