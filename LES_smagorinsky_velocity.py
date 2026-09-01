"""Smagorinsky eddy viscosity on a MAC staggered grid.

Strain rates use face-velocity differences at native staggered locations
(no averaging of prognostic u,v,w onto cell centres). ν_eff is returned at
pressure centres for the viscous term.
"""

from __future__ import annotations

import numpy as np

from mesh import FluidGrid


def _to_cell_centered(
    u: np.ndarray, v: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average face velocities onto cell centres (plots / sampling only)."""
    uc = 0.5 * (u[:-1] + u[1:])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    return uc, vc, wc


def _diff_y_on_u(u: np.ndarray, dy: float) -> np.ndarray:
    out = np.zeros_like(u)
    out[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / (2.0 * dy)
    out[:, 0, :] = (u[:, 1, :] - u[:, 0, :]) / dy
    out[:, -1, :] = (u[:, -1, :] - u[:, -2, :]) / dy
    return out


def _diff_z_on_u(u: np.ndarray, dz: float) -> np.ndarray:
    out = np.zeros_like(u)
    out[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * dz)
    out[:, :, 0] = (u[:, :, 1] - u[:, :, 0]) / dz
    out[:, :, -1] = (u[:, :, -1] - u[:, :, -2]) / dz
    return out


def _diff_x_on_v(v: np.ndarray, dx: float) -> np.ndarray:
    out = np.zeros_like(v)
    out[1:-1, :, :] = (v[2:, :, :] - v[:-2, :, :]) / (2.0 * dx)
    out[0, :, :] = (v[1, :, :] - v[0, :, :]) / dx
    out[-1, :, :] = (v[-1, :, :] - v[-2, :, :]) / dx
    return out


def _diff_z_on_v(v: np.ndarray, dz: float) -> np.ndarray:
    out = np.zeros_like(v)
    out[:, :, 1:-1] = (v[:, :, 2:] - v[:, :, :-2]) / (2.0 * dz)
    out[:, :, 0] = (v[:, :, 1] - v[:, :, 0]) / dz
    out[:, :, -1] = (v[:, :, -1] - v[:, :, -2]) / dz
    return out


def _diff_x_on_w(w: np.ndarray, dx: float) -> np.ndarray:
    out = np.zeros_like(w)
    out[1:-1, :, :] = (w[2:, :, :] - w[:-2, :, :]) / (2.0 * dx)
    out[0, :, :] = (w[1, :, :] - w[0, :, :]) / dx
    out[-1, :, :] = (w[-1, :, :] - w[-2, :, :]) / dx
    return out


def _diff_y_on_w(w: np.ndarray, dy: float) -> np.ndarray:
    out = np.zeros_like(w)
    out[:, 1:-1, :] = (w[:, 2:, :] - w[:, :-2, :]) / (2.0 * dy)
    out[:, 0, :] = (w[:, 1, :] - w[:, 0, :]) / dy
    out[:, -1, :] = (w[:, -1, :] - w[:, -2, :]) / dy
    return out


def strain_rate_magnitude_staggered(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """``|S| = sqrt(2 S_ij S_ij)`` at cell centres from staggered operators."""
    dudx = (u[1:] - u[:-1]) / dx
    dvdy = (v[:, 1:] - v[:, :-1]) / dy
    dwdz = (w[:, :, 1:] - w[:, :, :-1]) / dz

    dudy_u = _diff_y_on_u(u, dy)
    dudz_u = _diff_z_on_u(u, dz)
    dvdx_v = _diff_x_on_v(v, dx)
    dvdz_v = _diff_z_on_v(v, dz)
    dwdx_w = _diff_x_on_w(w, dx)
    dwdy_w = _diff_y_on_w(w, dy)

    dudy_c = 0.5 * (dudy_u[:-1] + dudy_u[1:])
    dudz_c = 0.5 * (dudz_u[:-1] + dudz_u[1:])
    dvdx_c = 0.5 * (dvdx_v[:, :-1] + dvdx_v[:, 1:])
    dvdz_c = 0.5 * (dvdz_v[:, :-1] + dvdz_v[:, 1:])
    dwdx_c = 0.5 * (dwdx_w[:, :, :-1] + dwdx_w[:, :, 1:])
    dwdy_c = 0.5 * (dwdy_w[:, :, :-1] + dwdy_w[:, :, 1:])

    Sxy = 0.5 * (dudy_c + dvdx_c)
    Sxz = 0.5 * (dudz_c + dwdx_c)
    Syz = 0.5 * (dvdz_c + dwdy_c)

    S2 = (
        2.0 * dudx**2
        + 2.0 * dvdy**2
        + 2.0 * dwdz**2
        + 4.0 * Sxy**2
        + 4.0 * Sxz**2
        + 4.0 * Syz**2
    )
    return np.sqrt(np.maximum(S2, 0.0))


def smagorinsky_viscosity(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    grid: FluidGrid,
    Cs: float = 0.17,
    nu: float = 1.5e-5,
) -> np.ndarray:
    """Return ν_eff = ν + ν_t at cell centres from staggered face velocities."""
    S_mag = strain_rate_magnitude_staggered(u, v, w, grid.dx, grid.dy, grid.dz)
    nu_t = (Cs * grid.delta()) ** 2 * S_mag
    return nu + nu_t
