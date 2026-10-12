#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class RectDomain:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def Lx(self) -> float:
        return self.xmax - self.xmin

    @property
    def Ly(self) -> float:
        return self.ymax - self.ymin


@dataclass(frozen=True)
class Cylinder:
    cx: float
    cy: float
    r: float


@dataclass(frozen=True)
class BoundaryLayerSpec:
    min_cell: float
    max_cell: float
    growth: float = 1.2


@dataclass(frozen=True)
class NineBlockOGridLayout:
    domain: RectDomain
    cylinder: Cylinder
    bl: BoundaryLayerSpec

    # O-grid sizing (computed)
    bl_layers: int
    bl_thickness: float
    ogrid_extra_thickness: float
    ogrid_outer_radius_axis: float

    # 3×3 split lines (computed)
    x0: float
    x1: float
    y0: float
    y1: float

    # 1-D grid lines (computed)
    x_lines: np.ndarray  # nodes, increasing
    y_lines: np.ndarray  # nodes, increasing


def _geometric_series_to_max(
    ds0: float, ds_max: float, growth: float
) -> List[float]:
    if ds0 <= 0 or ds_max <= 0:
        raise ValueError("cell sizes must be > 0")
    if ds0 > ds_max:
        raise ValueError(f"min_cell ({ds0}) cannot exceed max_cell ({ds_max})")
    if growth <= 1.0:
        raise ValueError("growth must be > 1.0")

    sizes: List[float] = [float(ds0)]
    if ds0 == ds_max:
        return sizes
    while True:
        nxt = sizes[-1] * growth
        if nxt >= ds_max * (1.0 - 1e-14):
            sizes.append(float(ds_max))
            break
        sizes.append(float(nxt))
        if len(sizes) > 100000:
            raise RuntimeError("unreasonable boundary-layer layer count; check inputs")
    return sizes


def _nodes_from_wall_to_interface(
    length: float,
    ds_min: float,
    ds_max: float,
    growth: float,
) -> np.ndarray:
    """Return 1-D node positions from wall at 0 to interface at `length`.

    - Starts with `ds_min` at the wall.
    - Grows geometrically up to `ds_max`.
    - Uses uniform `ds_max` thereafter.
    - Final cell is truncated to hit the interface exactly (<= ds_max).
    """
    if length <= 0:
        raise ValueError("segment length must be > 0")
    sizes = _geometric_series_to_max(ds_min, ds_max, growth)
    x: List[float] = [0.0]
    s = 0.0
    for ds in sizes:
        if s + ds >= length:
            break
        s += ds
        x.append(s)

    while True:
        rem = length - s
        if rem <= 1e-14:
            break
        if rem <= ds_max * (1.0 + 1e-12):
            x.append(length)
            break
        s += ds_max
        x.append(s)
        if len(x) > 5_000_000:
            raise RuntimeError("too many nodes generated; check inputs")
    return np.asarray(x, dtype=float)


def _uniform_nodes(a: float, b: float, ds_max: float) -> np.ndarray:
    """Uniform-ish nodes on [a,b] with max step ds_max (last step may be smaller)."""
    L = b - a
    if L <= 0:
        raise ValueError("invalid interval")
    n = int(math.ceil(L / ds_max))
    n = max(n, 1)
    return np.linspace(a, b, n + 1, dtype=float)


def build_9block_ogrid_layout(
    domain: RectDomain,
    cylinder: Cylinder,
    bl: BoundaryLayerSpec,
    *,
    extra_ogrid_cells: int = 1,
) -> NineBlockOGridLayout:
    """Compute the 9-domain split and 1-D grid lines.

    Interpretation (matches your description):
    - A boundary-layer progression exists at *all* solid walls (outer box + cylinder)
      with `min_cell` at the wall growing up to `max_cell`.
    - Outside boundary layers, cell size is clamped to `max_cell` (or smaller only
      in the final remainder cell of a segment).
    - The O-grid outer radius (on the x/y axes) is one additional cell outside the
      cylinder boundary-layer thickness.
    """
    if extra_ogrid_cells < 1:
        raise ValueError("extra_ogrid_cells must be >= 1")

    # Cylinder boundary-layer sizes (radial)
    cyl_sizes = _geometric_series_to_max(bl.min_cell, bl.max_cell, bl.growth)
    bl_thickness = float(sum(cyl_sizes))
    ogrid_extra = float(extra_ogrid_cells * bl.max_cell)
    r_outer_axis = cylinder.r + bl_thickness + ogrid_extra

    x0 = cylinder.cx - r_outer_axis
    x1 = cylinder.cx + r_outer_axis
    y0 = cylinder.cy - r_outer_axis
    y1 = cylinder.cy + r_outer_axis

    if not (domain.xmin < x0 < x1 < domain.xmax):
        raise ValueError(
            "O-grid square does not fit in the x-direction. "
            f"Need xmin < cx-r_outer < cx+r_outer < xmax, got "
            f"xmin={domain.xmin}, x0={x0}, x1={x1}, xmax={domain.xmax}."
        )
    if not (domain.ymin < y0 < y1 < domain.ymax):
        raise ValueError(
            "O-grid square does not fit in the y-direction. "
            f"Need ymin < cy-r_outer < cy+r_outer < ymax, got "
            f"ymin={domain.ymin}, y0={y0}, y1={y1}, ymax={domain.ymax}."
        )

    # X grid lines: left BL from xmin->x0, center uniform x0->x1, right BL x1->xmax
    left = _nodes_from_wall_to_interface(
        x0 - domain.xmin, bl.min_cell, bl.max_cell, bl.growth
    )
    x_left = domain.xmin + left  # includes x0

    x_mid = _uniform_nodes(x0, x1, bl.max_cell)  # includes x0,x1

    right = _nodes_from_wall_to_interface(
        domain.xmax - x1, bl.min_cell, bl.max_cell, bl.growth
    )
    x_right = domain.xmax - right[::-1]  # includes x1

    x_lines = np.unique(np.concatenate([x_left, x_mid, x_right]))
    x_lines.sort()

    # Y grid lines
    bot = _nodes_from_wall_to_interface(
        y0 - domain.ymin, bl.min_cell, bl.max_cell, bl.growth
    )
    y_bot = domain.ymin + bot

    y_mid = _uniform_nodes(y0, y1, bl.max_cell)

    top = _nodes_from_wall_to_interface(
        domain.ymax - y1, bl.min_cell, bl.max_cell, bl.growth
    )
    y_top = domain.ymax - top[::-1]

    y_lines = np.unique(np.concatenate([y_bot, y_mid, y_top]))
    y_lines.sort()

    return NineBlockOGridLayout(
        domain=domain,
        cylinder=cylinder,
        bl=bl,
        bl_layers=len(cyl_sizes),
        bl_thickness=bl_thickness,
        ogrid_extra_thickness=ogrid_extra,
        ogrid_outer_radius_axis=r_outer_axis,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        x_lines=x_lines,
        y_lines=y_lines,
    )


def plot_layout(layout: NineBlockOGridLayout, *, out: Path) -> None:
    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)

    dom = layout.domain
    cyl = layout.cylinder

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_aspect("equal", adjustable="box")

    # Domain boundary
    ax.plot(
        [dom.xmin, dom.xmax, dom.xmax, dom.xmin, dom.xmin],
        [dom.ymin, dom.ymin, dom.ymax, dom.ymax, dom.ymin],
        "k-",
        lw=1.5,
        label="domain",
    )

    # 9-block split square (outer edge of O-grid block)
    ax.plot(
        [layout.x0, layout.x1, layout.x1, layout.x0, layout.x0],
        [layout.y0, layout.y0, layout.y1, layout.y1, layout.y0],
        "C1-",
        lw=1.5,
        label="O-grid outer square (axis radius)",
    )

    # Cylinder + BL radii (circular reference)
    for rr, style, name in [
        (cyl.r, "C3-", "cylinder"),
        (cyl.r + layout.bl_thickness, "C2--", "BL outer (cylinder)"),
        (layout.ogrid_outer_radius_axis, "C1--", "O-grid outer (axis)"),
    ]:
        th = np.linspace(0, 2 * math.pi, 361)
        ax.plot(
            cyl.cx + rr * np.cos(th),
            cyl.cy + rr * np.sin(th),
            style,
            lw=1.2,
            label=name,
        )

    # Cartesian grid lines (the 8 outer blocks)
    for x in layout.x_lines:
        ax.plot([x, x], [dom.ymin, dom.ymax], color="0.85", lw=0.6)
    for y in layout.y_lines:
        ax.plot([dom.xmin, dom.xmax], [y, y], color="0.85", lw=0.6)

    ax.set_xlim(dom.xmin, dom.xmax)
    ax.set_ylim(dom.ymin, dom.ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("9-domain split + cylinder O-grid sizing (layout view)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _to_jsonable(layout: NineBlockOGridLayout) -> dict:
    d = asdict(layout)
    d["x_lines"] = layout.x_lines.tolist()
    d["y_lines"] = layout.y_lines.tolist()
    return d


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Compute a 9-block domain partition with an O-grid-sized core around a cylinder.\n"
            "Outputs split locations and 1-D grid lines derived from min/max cell sizes."
        )
    )
    p.add_argument(
        "--domain",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        required=True,
    )
    p.add_argument(
        "--cylinder",
        nargs=3,
        type=float,
        metavar=("CX", "CY", "R"),
        required=True,
    )
    p.add_argument("--min-cell", type=float, required=True)
    p.add_argument("--bl-max", type=float, required=True)
    p.add_argument("--growth", type=float, default=1.2)
    p.add_argument(
        "--extra-ogrid-cells",
        type=int,
        default=1,
        help="Cells outside the cylinder BL (on axes) included in the O-grid block.",
    )
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--plot-out", type=Path, default=None)

    args = p.parse_args(list(argv) if argv is not None else None)

    dom = RectDomain(*args.domain)
    cyl = Cylinder(*args.cylinder)
    bl = BoundaryLayerSpec(args.min_cell, args.bl_max, args.growth)

    layout = build_9block_ogrid_layout(
        dom, cyl, bl, extra_ogrid_cells=args.extra_ogrid_cells
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(_to_jsonable(layout), indent=2), encoding="utf-8")
    else:
        print(json.dumps(_to_jsonable(layout), indent=2))

    if args.plot_out:
        plot_layout(layout, out=args.plot_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

