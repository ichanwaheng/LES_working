"""Uniform Cartesian MAC staggered grid (2-D) for laminar cylinder flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class FluidGrid:
    """MAC staggered grid on a box ``[0, L] × [0, W]``.

    Pressure / scalars live at cell centres ``(nx, ny)``.
    ``u`` lives on vertical faces ``(nx+1, ny)``.
    ``v`` lives on horizontal faces ``(nx, ny+1)``.
    """

    L: float
    W: float
    nx: int
    ny: int

    def __post_init__(self) -> None:
        self.dx = self.L / self.nx
        self.dy = self.W / self.ny
        self.x = (np.arange(self.nx) + 0.5) * self.dx
        self.y = (np.arange(self.ny) + 0.5) * self.dy
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        self.x_u = np.arange(self.nx + 1) * self.dx
        self.y_v = np.arange(self.ny + 1) * self.dy
        self.n_cells = self.nx * self.ny
        self.volume = self.dx * self.dy

    @property
    def shape(self) -> Tuple[int, int]:
        return self.nx, self.ny

    @property
    def shape_u(self) -> Tuple[int, int]:
        return self.nx + 1, self.ny

    @property
    def shape_v(self) -> Tuple[int, int]:
        return self.nx, self.ny + 1

    def delta(self) -> float:
        return (self.dx * self.dy) ** 0.5
