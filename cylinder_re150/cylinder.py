"""Circular cylinder geometry and immersed-boundary mask (2-D)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mesh import FluidGrid


@dataclass
class Cylinder:
    """Circular cylinder of diameter ``D`` centred at ``(cx, cy)``."""

    D: float
    cx: float
    cy: float

    @property
    def R(self) -> float:
        return 0.5 * self.D

    def cell_mask(self, grid: FluidGrid) -> np.ndarray:
        """Solid cells: ``(x-cx)² + (y-cy)² ≤ R²``."""
        r2 = (grid.X - self.cx) ** 2 + (grid.Y - self.cy) ** 2
        return r2 <= self.R**2

    def surface_band(self, grid: FluidGrid, n_cells: float = 1.5) -> np.ndarray:
        """Fluid cells in a thin shell outside the cylinder (force sampling).

        Thickness is at least ``n_cells * Δ_min``, but never thinner than a
        small fraction of ``D``, so highly stretched wall meshes still yield
        enough sample cells.
        """
        r = np.sqrt((grid.X - self.cx) ** 2 + (grid.Y - self.cy) ** 2)
        h_min = float(min(np.min(grid.dx_c), np.min(grid.dy_c)))
        band = max(n_cells * h_min, 0.05 * self.D, 3.0 * h_min)
        solid = r <= self.R
        near = (r > self.R) & (r <= self.R + band)
        return near & ~solid
