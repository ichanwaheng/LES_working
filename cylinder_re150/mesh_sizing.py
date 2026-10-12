"""Boundary-layer mesh sizing for the laminar cylinder case.

Theory (wall-normal from a no-slip surface)
-------------------------------------------
1. Boundary-layer thickness::

       δ ≈ D / √Re

2. Choose number of cells in the BL, ``n_bl`` (typical 15–30) — **input**.

3. Choose geometric growth rate ``growth`` (typical 1.05–1.2) — **input**.

4. Cell sizes in the BL form a geometric series that sums to δ::

       Δ_k = Δ₁ · growth^{k-1},  k = 1 … n_bl
       Δ₁ · (growth^{n_bl} − 1) / (growth − 1) = δ

5. The **largest** BL cell ``Δ_max = Δ₁ · growth^{n_bl−1}`` is the uniform
   cell size used in the rest of the fluid domain (outside the BL).

6. The same wall treatment is applied on every side of the cylinder wall
   (and optionally domain walls).

7. The number of cells in each direction is then whatever is required to
   cover the domain with this spacing rule — not an arbitrary ``nx, ny``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BoundaryLayerSizing:
    """Resolved wall-normal BL packing + outer uniform size."""

    D: float
    Re: float
    n_bl: int
    growth: float

    def __post_init__(self) -> None:
        if self.D <= 0 or self.Re <= 0:
            raise ValueError("D and Re must be positive")
        if self.n_bl < 1:
            raise ValueError("n_bl must be >= 1")
        if self.growth < 1.0:
            raise ValueError("growth rate must be >= 1")

    @property
    def delta(self) -> float:
        """Approximate BL thickness ``δ = D / √Re``."""
        return float(self.D / np.sqrt(self.Re))

    @property
    def dy_wall(self) -> float:
        """First (minimum) cell size next to the wall, ``Δ₁``."""
        return float(self.bl_cell_sizes()[0])

    @property
    def dy_max(self) -> float:
        """Largest BL cell — used as far-field fluid cell size."""
        return float(self.bl_cell_sizes()[-1])

    def bl_cell_sizes(self) -> np.ndarray:
        """Wall-normal cell sizes inside the BL: ``[Δ₁, Δ₁ r, …, Δ_max]``."""
        n = int(self.n_bl)
        r = float(self.growth)
        delta = self.delta
        if n == 1:
            return np.array([delta], dtype=float)
        if abs(r - 1.0) < 1e-14:
            return np.full(n, delta / n, dtype=float)
        d1 = delta * (r - 1.0) / (r**n - 1.0)
        return d1 * r ** np.arange(n, dtype=float)

    def summary(self) -> dict:
        sizes = self.bl_cell_sizes()
        return {
            "D": self.D,
            "Re": self.Re,
            "delta_BL": self.delta,
            "n_bl": self.n_bl,
            "growth": self.growth,
            "dy_wall": float(sizes[0]),
            "dy_max": float(sizes[-1]),
            "bl_cell_sizes": sizes.tolist(),
            "bl_sum": float(sizes.sum()),
        }


def _fill_uniform(length: float, h: float) -> np.ndarray:
    """Cover ``length`` with cells of nominal size ``h`` (exact fit)."""
    length = float(length)
    if length <= 0.0:
        return np.array([], dtype=float)
    h = max(float(h), 1e-16)
    n = max(1, int(np.ceil(length / h - 1e-12)))
    return np.full(n, length / n, dtype=float)


def _pack_partial_bl(length: float, bl_sizes: np.ndarray, from_wall: bool) -> np.ndarray:
    """Fit as many geometric BL cells as fit into ``length`` (scaled last cell)."""
    if length <= 0.0:
        return np.array([], dtype=float)
    sizes = bl_sizes if from_wall else bl_sizes[::-1]
    out: List[float] = []
    remaining = float(length)
    for s in sizes:
        if remaining <= 1e-15:
            break
        if s <= remaining + 1e-15:
            out.append(float(s))
            remaining -= float(s)
        else:
            out.append(remaining)
            remaining = 0.0
            break
    if remaining > 1e-15:
        out.append(remaining)
    return np.asarray(out, dtype=float)


def line_spacings_from_walls(
    length: float,
    wall_lo: float,
    wall_hi: float,
    bl_sizes: Sequence[float],
    h_far: float | None = None,
) -> np.ndarray:
    """1-D cell widths on ``[0, length]`` with BL packing at ``wall_lo`` / ``wall_hi``.

    Layout (cylinder chord along this axis)::

        [0 ── outer (Δ_max) ── BL→wall | solid | wall→BL ── outer (Δ_max) ── L]

    The solid strip between the two wall hits is filled at the far-field
    size ``Δ_max`` (masked solid cells do not need wall spacing).
    """
    bl = np.asarray(bl_sizes, dtype=float)
    if bl.size < 1:
        raise ValueError("bl_sizes must be non-empty")
    h_outer = float(bl[-1] if h_far is None else h_far)
    bl_len = float(bl.sum())

    L = float(length)
    w0 = float(min(wall_lo, wall_hi))
    w1 = float(max(wall_lo, wall_hi))
    w0 = min(max(w0, 0.0), L)
    w1 = min(max(w1, 0.0), L)

    pieces: List[np.ndarray] = []

    # Left fluid: [0, w0] — approach wall (large → small)
    left = w0 - 0.0
    if left > bl_len + 1e-15:
        pieces.append(_fill_uniform(left - bl_len, h_outer))
        pieces.append(bl[::-1].copy())
    else:
        pieces.append(_pack_partial_bl(left, bl, from_wall=False))

    # Solid strip: [w0, w1] — far-field size (IB mask; not a resolved wall mesh)
    solid = w1 - w0
    pieces.append(_fill_uniform(solid, h_outer))

    # Right fluid: [w1, L] — leave wall (small → large)
    right = L - w1
    if right > bl_len + 1e-15:
        pieces.append(bl.copy())
        pieces.append(_fill_uniform(right - bl_len, h_outer))
    else:
        pieces.append(_pack_partial_bl(right, bl, from_wall=True))

    widths = np.concatenate([p for p in pieces if p.size > 0])
    # Renormalise tiny drift so sum == L
    s = float(widths.sum())
    if s > 0 and abs(s - L) / L > 1e-10:
        widths *= L / s
    return widths


def count_cells_report(widths: np.ndarray, bl_sizes: Sequence[float]) -> dict:
    bl = np.asarray(bl_sizes, dtype=float)
    return {
        "n_cells": int(widths.size),
        "min_dx": float(np.min(widths)) if widths.size else 0.0,
        "max_dx": float(np.max(widths)) if widths.size else 0.0,
        "dy_wall": float(bl[0]),
        "dy_max": float(bl[-1]),
    }


def build_cylinder_domain_spacings(
    L: float,
    W: float,
    cx: float,
    cy: float,
    R: float,
    sizing: BoundaryLayerSizing,
    h_far: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(dx_widths, dy_widths, info)`` for a box with circular cylinder.

    ``h_far`` defaults to ``Δ_max`` (largest BL cell), per the meshing rule.
    Pass a larger value only for coarse smoke tests.
    """
    bl = sizing.bl_cell_sizes()
    h_outer = float(bl[-1] if h_far is None else h_far)
    if h_outer < float(bl[-1]) - 1e-15:
        raise ValueError("h_far must be >= Δ_max (largest BL cell)")
    dx = line_spacings_from_walls(L, cx - R, cx + R, bl, h_far=h_outer)
    dy = line_spacings_from_walls(W, cy - R, cy + R, bl, h_far=h_outer)
    info = {
        "sizing": sizing.summary(),
        "h_far_used": h_outer,
        "h_far_is_delta_max": bool(h_far is None),
        "x": count_cells_report(dx, bl),
        "y": count_cells_report(dy, bl),
        "nx": int(dx.size),
        "ny": int(dy.size),
        "n_cells_total": int(dx.size * dy.size),
    }
    return dx, dy, info


def format_sizing_report(info: dict) -> str:
    s = info["sizing"]
    lines = [
        "=== Boundary-layer mesh sizing ===",
        f"  δ = D/√Re = {s['delta_BL']:.6e}",
        f"  n_bl = {s['n_bl']}   growth = {s['growth']}",
        f"  Δ₁ (wall) = {s['dy_wall']:.6e}",
        f"  Δ_max (from BL) = {s['dy_max']:.6e}",
        f"  h_far (outer domain) = {info.get('h_far_used', s['dy_max']):.6e}"
        + ("  [= Δ_max]" if info.get("h_far_is_delta_max", True) else "  [override]"),
        f"  BL cell sizes = [{', '.join(f'{v:.4e}' for v in s['bl_cell_sizes'][:5])}"
        + (" …]" if len(s["bl_cell_sizes"]) > 5 else "]"),
        f"  grid nx×ny = {info['nx']}×{info['ny']}  "
        f"(total {info['n_cells_total']} cells)",
        f"  x: min={info['x']['min_dx']:.4e}  max={info['x']['max_dx']:.4e}",
        f"  y: min={info['y']['min_dx']:.4e}  max={info['y']['max_dx']:.4e}",
    ]
    return "\n".join(lines)
