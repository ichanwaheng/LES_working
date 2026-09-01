"""Smagorinsky eddy viscosity on a MAC staggered grid.

Velocities are interpolated to cell centres, the strain-rate magnitude is
evaluated with central differences, and ν_eff is returned at cell centres.
"""

from __future__ import annotations

import numpy as np

from mesh import FluidGrid


def _to_cell_centered(
    u: np.ndarray, v: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average face velocities onto pressure / cell centres."""
    uc = 0.5 * (u[:-1] + u[1:])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    return uc, vc, wc


def smagorinsky_viscosity(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    grid: FluidGrid,
    Cs: float = 0.17,
    nu: float = 1.5e-5,
) -> np.ndarray:
    """Return ν_eff = ν + ν_t at cell centres from staggered (u, v, w)."""
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    uc, vc, wc = _to_cell_centered(u, v, w)

    dudx = np.gradient(uc, dx, axis=0, edge_order=1)
    dudy = np.gradient(uc, dy, axis=1, edge_order=1)
    dudz = np.gradient(uc, dz, axis=2, edge_order=1)
    dvdx = np.gradient(vc, dx, axis=0, edge_order=1)
    dvdy = np.gradient(vc, dy, axis=1, edge_order=1)
    dvdz = np.gradient(vc, dz, axis=2, edge_order=1)
    dwdx = np.gradient(wc, dx, axis=0, edge_order=1)
    dwdy = np.gradient(wc, dy, axis=1, edge_order=1)
    dwdz = np.gradient(wc, dz, axis=2, edge_order=1)

    S2 = (
        2.0 * dudx**2
        + 2.0 * dvdy**2
        + 2.0 * dwdz**2
        + (dudy + dvdx) ** 2
        + (dudz + dwdx) ** 2
        + (dvdz + dwdy) ** 2
    )
    S_mag = np.sqrt(np.maximum(S2, 0.0))
    delta = grid.delta()
    nu_t = (Cs * delta) ** 2 * S_mag
    return nu + nu_t
