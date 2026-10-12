"""2-D incompressible Navier–Stokes on a MAC staggered grid — laminar PISO.

Supports non-uniform cell widths from boundary-layer mesh sizing.

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


def _laplacian_variable(
    phi: np.ndarray,
    d_centre_x: np.ndarray,
    d_centre_y: np.ndarray,
) -> np.ndarray:
    """Five-point Laplacian on a rectilinear (possibly stretched) grid."""
    lap = np.zeros_like(phi)
    nx, ny = phi.shape

    if nx > 1 and d_centre_x.size >= 1:
        hm = d_centre_x[:-1, None]
        hp = d_centre_x[1:, None]
        if nx > 2:
            lap[1:-1, :] += (
                2.0
                * (
                    (phi[2:, :] - phi[1:-1, :]) / hp
                    - (phi[1:-1, :] - phi[:-2, :]) / hm
                )
                / (hm + hp)
            )
        h0 = float(d_centre_x[0])
        lap[0, :] += (phi[1, :] - phi[0, :]) / (h0 * h0)
        hN = float(d_centre_x[-1])
        lap[-1, :] += (phi[-2, :] - phi[-1, :]) / (hN * hN)

    if ny > 1 and d_centre_y.size >= 1:
        hm = d_centre_y[:-1][None, :]
        hp = d_centre_y[1:][None, :]
        if ny > 2:
            lap[:, 1:-1] += (
                2.0
                * (
                    (phi[:, 2:] - phi[:, 1:-1]) / hp
                    - (phi[:, 1:-1] - phi[:, :-2]) / hm
                )
                / (hm + hp)
            )
        h0 = float(d_centre_y[0])
        lap[:, 0] += (phi[:, 1] - phi[:, 0]) / (h0 * h0)
        hN = float(d_centre_y[-1])
        lap[:, -1] += (phi[:, -2] - phi[:, -1]) / (hN * hN)

    return lap


def _upwind_1d_var(
    phi: np.ndarray, vel: np.ndarray, d_c: np.ndarray, axis: int
) -> np.ndarray:
    """First-order upwind using neighbour spacings ``d_c`` (len = n-1)."""
    out = np.zeros_like(phi)
    if axis == 0:
        hm = d_c[:-1, None]
        hp = d_c[1:, None]
        out[1:-1] = np.where(
            vel[1:-1] >= 0.0,
            (phi[1:-1] - phi[:-2]) / hm,
            (phi[2:] - phi[1:-1]) / hp,
        )
    else:
        hm = d_c[:-1][None, :]
        hp = d_c[1:][None, :]
        out[:, 1:-1] = np.where(
            vel[:, 1:-1] >= 0.0,
            (phi[:, 1:-1] - phi[:, :-2]) / hm,
            (phi[:, 2:] - phi[:, 1:-1]) / hp,
        )
    return out


def _central_1d_var(phi: np.ndarray, d_c: np.ndarray, axis: int) -> np.ndarray:
    out = np.zeros_like(phi)
    if axis == 0:
        denom = (d_c[:-1] + d_c[1:])[:, None]
        out[1:-1] = (phi[2:] - phi[:-2]) / denom
        out[0] = (phi[1] - phi[0]) / d_c[0]
        out[-1] = (phi[-1] - phi[-2]) / d_c[-1]
    else:
        denom = (d_c[:-1] + d_c[1:])[None, :]
        out[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / denom
        out[:, 0] = (phi[:, 1] - phi[:, 0]) / d_c[0]
        out[:, -1] = (phi[:, -1] - phi[:, -2]) / d_c[-1]
    return out


def _convective_grad_var(
    phi: np.ndarray, vel: np.ndarray, d_c: np.ndarray, axis: int, scheme: str
) -> np.ndarray:
    if scheme == "upwind":
        return _upwind_1d_var(phi, vel, d_c, axis)
    return _central_1d_var(phi, d_c, axis)


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
        conv_scheme: str = "central",
    ) -> None:
        self.grid = grid
        self.rho = float(rho)
        self.nu = float(nu)
        self.U_inf = float(U_inf)
        self.u_clip = float(u_clip) if u_clip is not None else 8.0 * abs(self.U_inf)
        self.n_correctors = max(int(n_correctors), 1)
        scheme = str(conv_scheme).lower().strip()
        self.conv_scheme = scheme if scheme in ("central", "upwind") else "central"
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

        # Precompute neighbour spacings on each staggered family
        g = grid
        # u-faces at x_u; spacing between u-faces ≈ cell widths (interior)
        self._du_x = np.diff(g.x_u)  # length nx  (== dx_c)
        self._du_y = g.dy_p.copy() if g.dy_p.size else np.array([g.dy])
        # v-faces
        self._dv_x = g.dx_p.copy() if g.dx_p.size else np.array([g.dx])
        self._dv_y = np.diff(g.y_v)  # length ny (== dy_c)
        # pressure centres
        self._dp_x = g.dx_p
        self._dp_y = g.dy_p

    def set_immersed_boundary(self, mask: np.ndarray) -> None:
        """Stationary solid cylinder (zero face velocity on solid-adjacent faces)."""
        self._obstacle = np.asarray(mask, dtype=bool)
        self._u_solid = np.zeros(self.grid.shape_u)
        self._v_solid = np.zeros(self.grid.shape_v)
        self._lap_cache = None

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

        u[0, :] = U
        v[0, :] = 0.0

        u[-1, :] = u[-2, :]
        v[-1, :] = v[-2, :]

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
        v_u = self._interp_v_to_u(v)
        s = self.conv_scheme
        d_c_x = np.diff(self.grid.x_u)
        d_c_y = (
            self.grid.dy_p if self.grid.dy_p.size else np.array([self.grid.dy])
        )
        conv = (
            u * _convective_grad_var(u, u, d_c_x, 0, s)
            + v_u * _convective_grad_var(u, v_u, d_c_y, 1, s)
        )
        return -conv + self.nu * _laplacian_variable(u, d_c_x, d_c_y)

    def _momentum_H_v(self, u, v) -> np.ndarray:
        u_v = self._interp_u_to_v(u)
        s = self.conv_scheme
        d_c_x = (
            self.grid.dx_p if self.grid.dx_p.size else np.array([self.grid.dx])
        )
        d_c_y = np.diff(self.grid.y_v)
        conv = (
            u_v * _convective_grad_var(v, u_v, d_c_x, 0, s)
            + v * _convective_grad_var(v, v, d_c_y, 1, s)
        )
        return -conv + self.nu * _laplacian_variable(v, d_c_x, d_c_y)

    def _pressure_gradient_faces(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g = self.grid
        dpdx = np.zeros(g.shape_u)
        dpdy = np.zeros(g.shape_v)
        # u-face i (1..nx-1) between cells i-1 and i: use centre spacing
        if g.dx_p.size:
            dpdx[1:-1] = (p[1:] - p[:-1]) / g.dx_p[:, None]
        if g.dy_p.size:
            dpdy[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / g.dy_p[None, :]
        return dpdx, dpdy

    def _divergence(self, u, v) -> np.ndarray:
        g = self.grid
        div = (u[1:] - u[:-1]) / g.dx_c[:, None] + (v[:, 1:] - v[:, :-1]) / g.dy_c[
            None, :
        ]
        div[self._obstacle] = 0.0
        return div

    def _build_laplacian(self) -> sparse.csr_matrix:
        """Variable-spacing Neumann Laplacian consistent with div(grad)."""
        g = self.grid
        nx, ny = g.shape
        n = nx * ny
        rows, cols, data = [], [], []

        def add(r, c, val):
            rows.append(r)
            cols.append(c)
            data.append(val)

        dx_c, dy_c = g.dx_c, g.dy_c
        dx_p, dy_p = g.dx_p, g.dy_p

        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                diag = 0.0
                # x-fluxes through left/right faces
                if i > 0:
                    inv = 1.0 / (dx_p[i - 1] * dx_c[i])
                    add(idx, idx - ny, inv)
                    diag -= inv
                if i < nx - 1:
                    inv = 1.0 / (dx_p[i] * dx_c[i])
                    add(idx, idx + ny, inv)
                    diag -= inv
                if j > 0:
                    inv = 1.0 / (dy_p[j - 1] * dy_c[j])
                    add(idx, idx - 1, inv)
                    diag -= inv
                if j < ny - 1:
                    inv = 1.0 / (dy_p[j] * dy_c[j])
                    add(idx, idx + 1, inv)
                    diag -= inv
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
        # local cell size for band cells
        ii, jj = np.where(band)
        area = np.minimum(g.dx_c[ii], g.dy_c[jj])
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
