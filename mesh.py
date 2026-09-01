from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class FluidGrid:
    """Uniform Cartesian MAC staggered grid.

    Pressure and scalars live at cell centres ``(nx, ny, nz)``.
    Velocity components live on the corresponding faces:

    - ``u``: ``(nx+1, ny, nz)``  (faces normal to *x*)
    - ``v``: ``(nx, ny+1, nz)``  (faces normal to *y*)
    - ``w``: ``(nx, ny, nz+1)``  (faces normal to *z*)
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
        # Cell-centre coordinates
        self.x = (np.arange(self.nx) + 0.5) * self.dx
        self.y = (np.arange(self.ny) + 0.5) * self.dy
        self.z = (np.arange(self.nz) + 0.5) * self.dz
        self.X, self.Y, self.Z = np.meshgrid(
            self.x, self.y, self.z, indexing="ij"
        )
        # Face-centre coordinates
        self.x_u = np.arange(self.nx + 1) * self.dx
        self.y_v = np.arange(self.ny + 1) * self.dy
        self.z_w = np.arange(self.nz + 1) * self.dz
        self.n_cells = self.nx * self.ny * self.nz
        self.volume = self.dx * self.dy * self.dz

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Cell-centred field shape ``(nx, ny, nz)``."""
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

    def cell_index(self, i: int, j: int, k: int) -> int:
        return i * (self.ny * self.nz) + j * self.nz + k

    def ijk(self, idx: int) -> Tuple[int, int, int]:
        i = idx // (self.ny * self.nz)
        rem = idx % (self.ny * self.nz)
        j = rem // self.nz
        k = rem % self.nz
        return i, j, k

    def delta(self) -> float:
        return (self.dx * self.dy * self.dz) ** (1.0 / 3.0)

    @staticmethod
    def membrane_cell_mask(
        grid: FluidGrid,
        membrane_nodes: np.ndarray,
        thickness_cells: float = 1.5,
    ) -> np.ndarray:
        mx = membrane_nodes[:, 0]
        my = membrane_nodes[:, 1]
        mz = membrane_nodes[:, 2]
        band = thickness_cells * grid.dz
        mask = np.zeros(grid.shape, dtype=bool)

        x_min, x_max = mx.min(), mx.max()
        y_min, y_max = my.min(), my.max()

        for i in range(grid.nx):
            xc = grid.x[i]
            if xc < x_min or xc > x_max:
                continue
            for j in range(grid.ny):
                yc = grid.y[j]
                if yc < y_min or yc > y_max:
                    continue

                d2 = (mx - xc) ** 2 + (my - yc) ** 2
                n = int(np.argmin(d2))
                z_m = mz[n]
                for k in range(grid.nz):
                    if abs(grid.z[k] - z_m) <= band:
                        mask[i, j, k] = True
        return mask
