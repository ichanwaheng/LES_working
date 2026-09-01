"""Circular cylinder geometry and immersed-boundary mask (extruded in z)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mesh import FluidGrid


@dataclass
class Cylinder:
    """Infinite circular cylinder of diameter ``D``, axis along *z*."""

    D: float
    cx: float
    cy: float

    @property
    def R(self) -> float:
        return 0.5 * self.D

    def cell_mask(self, grid: FluidGrid) -> np.ndarray:
        """Solid cells: ``(x-cx)² + (y-cy)² ≤ R²`` for all *z*."""
        r2 = (grid.X - self.cx) ** 2 + (grid.Y - self.cy) ** 2
        return r2 <= self.R**2

    def surface_band(self, grid: FluidGrid, n_cells: float = 1.5) -> np.ndarray:
        """Fluid cells within ``n_cells`` of the cylinder surface (force sampling)."""
        r = np.sqrt((grid.X - self.cx) ** 2 + (grid.Y - self.cy) ** 2)
        band = n_cells * min(grid.dx, grid.dy)
        solid = r <= self.R
        near = (r > self.R) & (r <= self.R + band)
        return near & ~solid
