"""Uniform Cartesian MAC staggered grid for the cylinder LES case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class FluidGrid:
    """MAC staggered grid on a box ``[0,L] × [0,W] × [0,H]``.

    Pressure / scalars: cell centres ``(nx, ny, nz)``.
    ``u``: ``(nx+1, ny, nz)``, ``v``: ``(nx, ny+1, nz)``, ``w``: ``(nx, ny, nz+1)``.
    """

    L: float
    W: float
    H: float
    nx: int
    ny: int
    nz: int

    def __post_init__(self) -> None:
        self.dx = self.L / self.nx
        self.dy = self.W / self.ny
        self.dz = self.H / self.nz
        self.x = (np.arange(self.nx) + 0.5) * self.dx
        self.y = (np.arange(self.ny) + 0.5) * self.dy
        self.z = (np.arange(self.nz) + 0.5) * self.dz
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.x_u = np.arange(self.nx + 1) * self.dx
        self.y_v = np.arange(self.ny + 1) * self.dy
        self.z_w = np.arange(self.nz + 1) * self.dz
        self.n_cells = self.nx * self.ny * self.nz
        self.volume = self.dx * self.dy * self.dz

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.nx, self.ny, self.nz

    @property
    def shape_u(self) -> Tuple[int, int, int]:
        return self.nx + 1, self.ny, self.nz

    @property
    def shape_v(self) -> Tuple[int, int, int]:
        return self.nx, self.ny + 1, self.nz

    @property
    def shape_w(self) -> Tuple[int, int, int]:
        return self.nx, self.ny, self.nz + 1

    def delta(self) -> float:
        return (self.dx * self.dy * self.dz) ** (1.0 / 3.0)
