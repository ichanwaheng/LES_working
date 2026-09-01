"""Smagorinsky eddy viscosity on a MAC staggered grid.

Strain rates are formed from face-velocity differences at their natural
locations (no averaging of u,v,w onto cell centres):

- ``S_xx, S_yy, S_zz`` at pressure centres from face jumps
- shear components from derivatives on the corresponding faces, then
  averaged to centres only as scalars for ``|S|`` / ν_t
"""

from __future__ import annotations

import numpy as np

from mesh import FluidGrid


def cell_centered_velocity(
    u: np.ndarray, v: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average face velocities onto cell centres (plots / CFL only)."""
    uc = 0.5 * (u[:-1] + u[1:])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    return uc, vc, wc


def _diff_y_on_u(u: np.ndarray, dy: float) -> np.ndarray:
    """∂u/∂y at u-face locations; ``u`` has shape (nx+1, ny, nz)."""
    out = np.zeros_like(u)
    out[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / (2.0 * dy)
    out[:, 0, :] = (u[:, 1, :] - u[:, 0, :]) / dy
    out[:, -1, :] = (u[:, -1, :] - u[:, -2, :]) / dy
    return out


def _diff_z_on_u(u: np.ndarray, dz: float) -> np.ndarray:
    """∂u/∂z at u-face locations."""
    out = np.zeros_like(u)
    out[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * dz)
    out[:, :, 0] = (u[:, :, 1] - u[:, :, 0]) / dz
    out[:, :, -1] = (u[:, :, -1] - u[:, :, -2]) / dz
    return out


def _diff_x_on_v(v: np.ndarray, dx: float) -> np.ndarray:
    """∂v/∂x at v-face locations; ``v`` has shape (nx, ny+1, nz)."""
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
    """∂w/∂x at w-face locations; ``w`` has shape (nx, ny, nz+1)."""
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


def _faces_to_centres_u(phi_u: np.ndarray) -> np.ndarray:
    """Average a u-face field onto cell centres."""
    return 0.5 * (phi_u[:-1] + phi_u[1:])


def _faces_to_centres_v(phi_v: np.ndarray) -> np.ndarray:
    return 0.5 * (phi_v[:, :-1] + phi_v[:, 1:])


def _faces_to_centres_w(phi_w: np.ndarray) -> np.ndarray:
    return 0.5 * (phi_w[:, :, :-1] + phi_w[:, :, 1:])


def strain_rate_magnitude_staggered(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """``|S| = sqrt(2 S_ij S_ij)`` at cell centres from staggered operators."""
    # Normal strains at pressure centres (face jumps — no velocity average)
    dudx = (u[1:] - u[:-1]) / dx
    dvdy = (v[:, 1:] - v[:, :-1]) / dy
    dwdz = (w[:, :, 1:] - w[:, :, :-1]) / dz

    # Shear: derivatives on native faces, then average those scalars to centres
    dudy_c = _faces_to_centres_u(_diff_y_on_u(u, dy))
    dudz_c = _faces_to_centres_u(_diff_z_on_u(u, dz))
    dvdx_c = _faces_to_centres_v(_diff_x_on_v(v, dx))
    dvdz_c = _faces_to_centres_v(_diff_z_on_v(v, dz))
    dwdx_c = _faces_to_centres_w(_diff_x_on_w(w, dx))
    dwdy_c = _faces_to_centres_w(_diff_y_on_w(w, dy))

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
    # 2 S_ij S_ij = 2(Sxx²+Syy²+Szz²) + 4(Sxy²+Sxz²+Syz²)
    return np.sqrt(np.maximum(S2, 0.0))


def smagorinsky_viscosity(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    grid: FluidGrid,
    Cs: float = 0.1,
    nu: float = 1.0 / 3900.0,
) -> np.ndarray:
    """Return ν_eff = ν + ν_t at cell centres from staggered face velocities."""
    S_mag = strain_rate_magnitude_staggered(u, v, w, grid.dx, grid.dy, grid.dz)
    nu_t = (Cs * grid.delta()) ** 2 * S_mag
    return nu + nu_t
