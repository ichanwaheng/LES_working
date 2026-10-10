"""2-D incompressible Navier–Stokes on a MAC staggered grid — laminar PISO.

External-flow BCs for a cylinder in a box:
- inlet (west): uniform ``U_inf``
- outlet (east): Neumann
- far-field *y*: freestream / slip
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve

from mesh import FluidGrid


@dataclass
class FluidState:
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray

    def cell_centered_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        uc = 0.5 * (self.u[:-1] + self.u[1:])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        return uc, vc


def _laplacian_field(phi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    lap = np.zeros_like(phi)
    if phi.shape[0] > 1:
        lap[1:-1] += (phi[2:] - 2 * phi[1:-1] + phi[0:-2]) / dx**2
        lap[0] += (phi[1] - phi[0]) / dx**2
        lap[-1] += (phi[-2] - phi[-1]) / dx**2
    if phi.shape[1] > 1:
        lap[:, 1:-1] += (phi[:, 2:] - 2 * phi[:, 1:-1] + phi[:, 0:-2]) / dy**2
        lap[:, 0] += (phi[:, 1] - phi[:, 0]) / dy**2
        lap[:, -1] += (phi[:, -2] - phi[:, -1]) / dy**2
    return lap


def _upwind_1d(phi: np.ndarray, vel: np.ndarray, d: float, axis: int) -> np.ndarray:
    out = np.zeros_like(phi)
    if axis == 0:
        out[1:-1] = np.where(
            vel[1:-1] >= 0.0,
            (phi[1:-1] - phi[:-2]) / d,
            (phi[2:] - phi[1:-1]) / d,
        )
    else:
        out[:, 1:-1] = np.where(
            vel[:, 1:-1] >= 0.0,
            (phi[:, 1:-1] - phi[:, :-2]) / d,
            (phi[:, 2:] - phi[:, 1:-1]) / d,
        )
    return out


class FluidSolver:
    """MAC-staggered laminar PISO for 2-D flow past a cylinder."""

    def __init__(
        self,
        grid: FluidGrid,
        rho: float = 1.0,
        nu: float = 1.0 / 150.0,
        U_inf: float = 1.0,
        u_clip: Optional[float] = None,
        n_correctors: int = 2,
    ) -> None:
        self.grid = grid
        self.rho = float(rho)
        self.nu = float(nu)
        self.U_inf = float(U_inf)
        self.u_clip = float(u_clip) if u_clip is not None else 8.0 * abs(self.U_inf)
        self.n_correctors = max(int(n_correctors), 1)
        self.t = 0.0

        self.state = FluidState(
            u=np.full(grid.shape_u, self.U_inf),
            v=np.zeros(grid.shape_v),
            p=np.zeros(grid.shape),
        )
        self._obstacle = np.zeros(grid.shape, dtype=bool)
        self._u_solid = np.zeros(grid.shape_u)
        self._v_solid = np.zeros(grid.shape_v)
        self._lap_cache: Optional[sparse.csr_matrix] = None

    def set_immersed_boundary(self, mask: np.ndarray) -> None:
        """Stationary solid cylinder (zero face velocity on solid-adjacent faces)."""
        self._obstacle = np.asarray(mask, dtype=bool)
        self._u_solid = np.zeros(self.grid.shape_u)
        self._v_solid = np.zeros(self.grid.shape_v)

    def _face_mask_u(self) -> np.ndarray:
        m = self._obstacle
        out = np.zeros(self.grid.shape_u, dtype=bool)
        out[0] = m[0]
        out[-1] = m[-1]
        out[1:-1] = m[:-1] | m[1:]
        return out

    def _face_mask_v(self) -> np.ndarray:
        m = self._obstacle
        out = np.zeros(self.grid.shape_v, dtype=bool)
        out[:, 0] = m[:, 0]
        out[:, -1] = m[:, -1]
        out[:, 1:-1] = m[:, :-1] | m[:, 1:]
        return out

    def _apply_bc(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        U = self.U_inf

        # Inlet
        u[0, :] = U
        v[0, :] = 0.0

        # Outlet Neumann
        u[-1, :] = u[-2, :]
        v[-1, :] = v[-2, :]

        # Far-field y: freestream / slip
        u[:, 0] = U
        u[:, -1] = U
        v[:, 0] = 0.0
        v[:, -1] = 0.0

        if np.any(self._obstacle):
            mu, mv = self._face_mask_u(), self._face_mask_v()
            u[mu] = self._u_solid[mu]
            v[mv] = self._v_solid[mv]
        return u, v

    def _sanitize(self, *fields: np.ndarray) -> Tuple[np.ndarray, ...]:
        out = []
        for f in fields:
            g = np.nan_to_num(f, nan=0.0, posinf=self.u_clip, neginf=-self.u_clip)
            out.append(np.clip(g, -self.u_clip, self.u_clip))
        return tuple(out)

    def _interp_v_to_u(self, v: np.ndarray) -> np.ndarray:
        out = np.zeros(self.grid.shape_u)
        vc = 0.5 * (v[:, :-1] + v[:, 1:])
        out[1:-1] = 0.5 * (vc[:-1] + vc[1:])
        out[0] = vc[0]
        out[-1] = vc[-1]
        return out

    def _interp_u_to_v(self, u: np.ndarray) -> np.ndarray:
        out = np.zeros(self.grid.shape_v)
        uc = 0.5 * (u[:-1, :] + u[1:, :])
        out[:, 1:-1] = 0.5 * (uc[:, :-1] + uc[:, 1:])
        out[:, 0] = uc[:, 0]
        out[:, -1] = uc[:, -1]
        return out

    def _momentum_H_u(self, u, v) -> np.ndarray:
        g = self.grid
        v_u = self._interp_v_to_u(v)
        conv = u * _upwind_1d(u, u, g.dx, 0) + v_u * _upwind_1d(u, v_u, g.dy, 1)
        return -conv + self.nu * _laplacian_field(u, g.dx, g.dy)

    def _momentum_H_v(self, u, v) -> np.ndarray:
        g = self.grid
        u_v = self._interp_u_to_v(u)
        conv = u_v * _upwind_1d(v, u_v, g.dx, 0) + v * _upwind_1d(v, v, g.dy, 1)
        return -conv + self.nu * _laplacian_field(v, g.dx, g.dy)

    def _pressure_gradient_faces(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g = self.grid
        dpdx = np.zeros(g.shape_u)
        dpdy = np.zeros(g.shape_v)
        dpdx[1:-1] = (p[1:] - p[:-1]) / g.dx
        dpdy[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / g.dy
        return dpdx, dpdy

    def _divergence(self, u, v) -> np.ndarray:
        g = self.grid
        div = (u[1:] - u[:-1]) / g.dx + (v[:, 1:] - v[:, :-1]) / g.dy
        div[self._obstacle] = 0.0
        return div

    def _build_laplacian(self) -> sparse.csr_matrix:
        g = self.grid
        nx, ny = g.shape
        n = nx * ny
        dx2, dy2 = g.dx**2, g.dy**2
        rows, cols, data = [], [], []

        def add(r, c, val):
            rows.append(r)
            cols.append(c)
            data.append(val)

        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                diag = 0.0
                if i > 0:
                    add(idx, idx - ny, 1.0 / dx2)
                    diag -= 1.0 / dx2
                if i < nx - 1:
                    add(idx, idx + ny, 1.0 / dx2)
                    diag -= 1.0 / dx2
                if j > 0:
                    add(idx, idx - 1, 1.0 / dy2)
                    diag -= 1.0 / dy2
                if j < ny - 1:
                    add(idx, idx + 1, 1.0 / dy2)
                    diag -= 1.0 / dy2
                add(idx, idx, diag if abs(diag) > 0 else 1.0)

        A = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tolil()
        A[0, :] = 0.0
        A[0, 0] = 1.0
        return A.tocsr()

    def _solve_pressure_correction(self, div: np.ndarray, dt: float) -> np.ndarray:
        if self._lap_cache is None:
            self._lap_cache = self._build_laplacian()
        rhs = (self.rho / max(dt, 1e-12)) * div.ravel(order="C")
        rhs = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)
        rhs -= rhs.mean()
        rhs[0] = 0.0
        p_flat, info = cg(self._lap_cache, rhs, rtol=1e-6, maxiter=800)
        if info != 0 or not np.all(np.isfinite(p_flat)):
            p_flat = spsolve(self._lap_cache, rhs)
        p_corr = np.asarray(p_flat, dtype=float).reshape(self.grid.shape, order="C")
        p_corr = np.nan_to_num(p_corr, nan=0.0, posinf=0.0, neginf=0.0)
        p_corr -= p_corr.mean()
        return p_corr

    def step(self, dt: float) -> FluidState:
        self.t += dt
        u, v = self._sanitize(self.state.u, self.state.v)
        p = np.nan_to_num(self.state.p, nan=0.0)

        Hu = self._momentum_H_u(u, v)
        Hv = self._momentum_H_v(u, v)
        dpdx, dpdy = self._pressure_gradient_faces(p)
        u_s = u + dt * (Hu - dpdx / self.rho)
        v_s = v + dt * (Hv - dpdy / self.rho)
        u_s, v_s = self._sanitize(u_s, v_s)
        u_s, v_s = self._apply_bc(u_s, v_s)

        for corr in range(self.n_correctors):
            if corr > 0:
                Hu = self._momentum_H_u(u_s, v_s)
                Hv = self._momentum_H_v(u_s, v_s)
                dpdx, dpdy = self._pressure_gradient_faces(p)
                u_s = u + dt * (Hu - dpdx / self.rho)
                v_s = v + dt * (Hv - dpdy / self.rho)
                u_s, v_s = self._sanitize(u_s, v_s)
                u_s, v_s = self._apply_bc(u_s, v_s)

            div = self._divergence(u_s, v_s)
            p_corr = self._solve_pressure_correction(div, dt)
            cdx, cdy = self._pressure_gradient_faces(p_corr)
            u_s = u_s - (dt / self.rho) * cdx
            v_s = v_s - (dt / self.rho) * cdy
            p = p + p_corr
            u_s, v_s = self._sanitize(u_s, v_s)
            u_s, v_s = self._apply_bc(u_s, v_s)

        p = np.nan_to_num(p, nan=0.0)
        p -= p.mean()
        self.state = FluidState(u=u_s, v=v_s, p=p)
        return self.state

    def force_coefficients(
        self, cylinder_D: float, surface_mask: np.ndarray
    ) -> Tuple[float, float]:
        """Approximate Cd, Cl from pressure on a near-surface fluid band.

        Uses ``F ≈ -p n ΔA`` with outward normal from cylinder centre;
        normalised by ``½ ρ U² D`` (2-D unit depth).
        """
        g = self.grid
        p = self.state.p
        solid = self._obstacle
        band = surface_mask
        if not np.any(solid) or not np.any(band):
            return 0.0, 0.0

        xs = g.X[solid]
        ys = g.Y[solid]
        cx, cy = float(xs.mean()), float(ys.mean())

        rx = g.X[band] - cx
        ry = g.Y[band] - cy
        r = np.sqrt(rx * rx + ry * ry) + 1e-12
        nx_hat = rx / r
        ny_hat = ry / r
        area = min(g.dx, g.dy)  # unit depth
        px = p[band]
        Fx = float(np.sum(-px * nx_hat * area))
        Fy = float(np.sum(-px * ny_hat * area))
        q = 0.5 * self.rho * self.U_inf**2
        ref = q * cylinder_D + 1e-12
        return Fx / ref, Fy / ref

    def max_cfl(self, dt: float) -> float:
        g = self.grid
        uc, vc = self.state.cell_centered_velocity()
        umax = max(float(np.max(np.abs(uc))), float(np.max(np.abs(vc))), 1e-12)
        return umax * dt / min(g.dx, g.dy)
