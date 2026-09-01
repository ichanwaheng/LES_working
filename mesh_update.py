from __future__ import annotations

import numpy as np

from mesh import FluidGrid
from PISO import FluidSolver


def update_immsersed_boundary(
    fluid: FluidSolver,
    grid: FluidGrid,
    mesh,
    nodes: np.ndarray,
    velocities: np.ndarray,
    thickness_cells: float = 1.5,
) -> np.ndarray:
    """Mark solid cells and set staggered face solid velocities from the membrane."""
    mask = FluidGrid.membrane_cell_mask(
        grid, nodes, thickness_cells=thickness_cells
    )
    # Cell-centred solid velocity (interpolated onto faces inside FluidSolver)
    u_s = np.zeros(grid.shape)
    v_s = np.zeros(grid.shape)
    w_s = np.zeros(grid.shape)

    mx, my, mz = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    vx, vy, vz = velocities[:, 0], velocities[:, 1], velocities[:, 2]
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
            if not np.any(mask[i, j, :]):
                continue
            d2 = (mx - xc) ** 2 + (my - yc) ** 2
            n = int(np.argmin(d2))
            for k in range(grid.nz):
                if mask[i, j, k]:
                    u_s[i, j, k] = vx[n]
                    v_s[i, j, k] = vy[n]
                    w_s[i, j, k] = vz[n]

    fluid.set_immersed_boundary(mask, u_s, v_s, w_s)
    return mask


def under_relax(new: np.ndarray, old: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * new + (1.0 - alpha) * old
