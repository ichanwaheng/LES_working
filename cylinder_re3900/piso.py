"""Incompressible NS on a MAC staggered grid — PISO + Smagorinsky LES.

External-flow BCs for a cylinder in a box:
- inlet (west): uniform ``U_inf``
- outlet (east): Neumann
- far-field *y* / *z*: freestream / slip (not channel no-slip walls)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve

import les
from mesh import FluidGrid


@dataclass
class FluidState:
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    p: np.ndarray
    nu_eff: np.ndarray

    def cell_centered_velocity(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return les.cell_centered_velocity(self.u, self.v, self.w)


def _laplacian_field(phi: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    lap = np.zeros_like(phi)
    if phi.shape[0] > 1:
        lap[1:-1] += (phi[2:] - 2 * phi[1:-1] + phi[0:-2]) / dx**2
        lap[0] += (phi[1] - phi[0]) / dx**2
        lap[-1] += (phi[-2] - phi[-1]) / dx**2
    if phi.shape[1] > 1:
        lap[:, 1:-1] += (phi[:, 2:] - 2 * phi[:, 1:-1] + phi[:, 0:-2]) / dy**2
        lap[:, 0] += (phi[:, 1] - phi[:, 0]) / dy**2
        lap[:, -1] += (phi[:, -2] - phi[:, -1]) / dy**2
    if phi.shape[2] > 1:
        lap[:, :, 1:-1] += (
            phi[:, :, 2:] - 2 * phi[:, :, 1:-1] + phi[:, :, 0:-2]
        ) / dz**2
        lap[:, :, 0] += (phi[:, :, 1] - phi[:, :, 0]) / dz**2
        lap[:, :, -1] += (phi[:, :, -2] - phi[:, :, -1]) / dz**2
    return lap


def _upwind_1d(phi: np.ndarray, vel: np.ndarray, d: float, axis: int) -> np.ndarray:
    out = np.zeros_like(phi)
    if axis == 0:
        out[1:-1] = np.where(
            vel[1:-1] >= 0.0,
            (phi[1:-1] - phi[:-2]) / d,
            (phi[2:] - phi[1:-1]) / d,
        )
    elif axis == 1:
        out[:, 1:-1] = np.where(
            vel[:, 1:-1] >= 0.0,
            (phi[:, 1:-1] - phi[:, :-2]) / d,
            (phi[:, 2:] - phi[:, 1:-1]) / d,
        )
    else:
        out[:, :, 1:-1] = np.where(
            vel[:, :, 1:-1] >= 0.0,
            (phi[:, :, 1:-1] - phi[:, :, :-2]) / d,
            (phi[:, :, 2:] - phi[:, :, 1:-1]) / d,
        )
    return out


class FluidSolver:
    """MAC-staggered PISO + LES for flow past a cylinder."""

    def __init__(
        self,
        grid: FluidGrid,
        rho: float = 1.0,
        nu: float = 1.0 / 3900.0,
        U_inf: float = 1.0,
        use_les: bool = True,
        Cs: float = 0.1,
        u_clip: Optional[float] = None,
        n_correctors: int = 2,
    ) -> None:
        self.grid = grid
        self.rho = float(rho)
        self.nu = float(nu)
        self.U_inf = float(U_inf)
        self.use_les = bool(use_les)
        self.Cs = float(Cs)
        self.u_clip = float(u_clip) if u_clip is not None else 8.0 * abs(self.U_inf)
        self.n_correctors = max(int(n_correctors), 1)
        self.t = 0.0

        self.state = FluidState(
            u=np.full(grid.shape_u, self.U_inf),
            v=np.zeros(grid.shape_v),
            w=np.zeros(grid.shape_w),
            p=np.zeros(grid.shape),
            nu_eff=np.full(grid.shape, self.nu),
        )
        self._obstacle = np.zeros(grid.shape, dtype=bool)
        self._u_solid = np.zeros(grid.shape_u)
        self._v_solid = np.zeros(grid.shape_v)
        self._w_solid = np.zeros(grid.shape_w)
        self._lap_cache: Optional[sparse.csr_matrix] = None

    def set_immersed_boundary(self, mask: np.ndarray) -> None:
        """Stationary solid cylinder (zero face velocity on solid-adjacent faces)."""
        self._obstacle = np.asarray(mask, dtype=bool)
        self._u_solid = np.zeros(self.grid.shape_u)
        self._v_solid = np.zeros(self.grid.shape_v)
        self._w_solid = np.zeros(self.grid.shape_w)

    def _cells_to_u_faces(self, phi: np.ndarray) -> np.ndarray:
        out = np.zeros(self.grid.shape_u)
        out[1:-1] = 0.5 * (phi[:-1] + phi[1:])
        out[0] = phi[0]
        out[-1] = phi[-1]
        return out

    def _cells_to_v_faces(self, phi: np.ndarray) -> np.ndarray:
        out = np.zeros(self.grid.shape_v)
        out[:, 1:-1] = 0.5 * (phi[:, :-1] + phi[:, 1:])
        out[:, 0] = phi[:, 0]
        out[:, -1] = phi[:, -1]
        return out

    def _cells_to_w_faces(self, phi: np.ndarray) -> np.ndarray:
        out = np.zeros(self.grid.shape_w)
        out[:, :, 1:-1] = 0.5 * (phi[:, :, :-1] + phi[:, :, 1:])
        out[:, :, 0] = phi[:, :, 0]
        out[:, :, -1] = phi[:, :, -1]
        return out

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

    def _face_mask_w(self) -> np.ndarray:
        m = self._obstacle
        out = np.zeros(self.grid.shape_w, dtype=bool)
        out[:, :, 0] = m[:, :, 0]
        out[:, :, -1] = m[:, :, -1]
        out[:, :, 1:-1] = m[:, :, :-1] | m[:, :, 1:]
        return out

    def _apply_bc(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        U = self.U_inf

        # Inlet
        u[0, :, :] = U
        v[0, :, :] = 0.0
        w[0, :, :] = 0.0

        # Outlet Neumann
        u[-1, :, :] = u[-2, :, :]
        v[-1, :, :] = v[-2, :, :]
        w[-1, :, :] = w[-2, :, :]

        # Far-field y: freestream / slip
        u[:, 0, :] = U
        u[:, -1, :] = U
        v[:, 0, :] = 0.0
        v[:, -1, :] = 0.0
        w[:, 0, :] = w[:, 1, :]
        w[:, -1, :] = w[:, -2, :]

        # Spanwise z: slip (approximation to periodic for short spans)
        u[:, :, 0] = u[:, :, 1]
        u[:, :, -1] = u[:, :, -2]
        v[:, :, 0] = v[:, :, 1]
        v[:, :, -1] = v[:, :, -2]
        w[:, :, 0] = 0.0
        w[:, :, -1] = 0.0

        if np.any(self._obstacle):
            mu, mv, mw = self._face_mask_u(), self._face_mask_v(), self._face_mask_w()
            u[mu] = self._u_solid[mu]
            v[mv] = self._v_solid[mv]
            w[mw] = self._w_solid[mw]
        return u, v, w

    def _sanitize(self, *fields: np.ndarray) -> Tuple[np.ndarray, ...]:
        out = []
        for f in fields:
            g = np.nan_to_num(f, nan=0.0, posinf=self.u_clip, neginf=-self.u_clip)
            out.append(np.clip(g, -self.u_clip, self.u_clip))
        return tuple(out)

    def _interp_v_to_u(self, v: np.ndarray) -> np.ndarray:
        out = np.zeros(self.grid.shape_u)
        vc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
        out[1:-1] = 0.5 * (vc[:-1] + vc[1:])
        out[0] = vc[0]
        out[-1] = vc[-1]
        return out

    def _interp_w_to_u(self, w: np.ndarray) -> np.ndarray:
        out = np.zeros(self.grid.shape_u)
        wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
        out[1:-1] = 0.5 * (wc[:-1] + wc[1:])
        out[0] = wc[0]
        out[-1] = wc[-1]
        return out

    def _interp_u_to_v(self, u: np.ndarray) -> np.ndarray:
        out = np.zeros(self.grid.shape_v)
        uc = 0.5 * (u[:-1, :, :] + u[1:, :, :])
        out[:, 1:-1] = 0.5 * (uc[:, :-1] + uc[:, 1:])
        out[:, 0] = uc[:, 0]
        out[:, -1] = uc[:, -1]
        return out

    def _interp_w_to_v(self, w: np.ndarray) -> np.ndarray:
        out = np.zeros(self.grid.shape_v)
        wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
        out[:, 1:-1] = 0.5 * (wc[:, :-1] + wc[:, 1:])
        out[:, 0] = wc[:, 0]
        out[:, -1] = wc[:, -1]
        return out

    def _interp_u_to_w(self, u: np.ndarray) -> np.ndarray:
        out = np.zeros(self.grid.shape_w)
        uc = 0.5 * (u[:-1, :, :] + u[1:, :, :])
        out[:, :, 1:-1] = 0.5 * (uc[:, :, :-1] + uc[:, :, 1:])
        out[:, :, 0] = uc[:, :, 0]
        out[:, :, -1] = uc[:, :, -1]
        return out

    def _interp_v_to_w(self, v: np.ndarray) -> np.ndarray:
        out = np.zeros(self.grid.shape_w)
        vc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
        out[:, :, 1:-1] = 0.5 * (vc[:, :, :-1] + vc[:, :, 1:])
        out[:, :, 0] = vc[:, :, 0]
        out[:, :, -1] = vc[:, :, -1]
        return out

    def _momentum_H_u(self, u, v, w, nu_u) -> np.ndarray:
        g = self.grid
        v_u = self._interp_v_to_u(v)
        w_u = self._interp_w_to_u(w)
        conv = (
            u * _upwind_1d(u, u, g.dx, 0)
            + v_u * _upwind_1d(u, v_u, g.dy, 1)
            + w_u * _upwind_1d(u, w_u, g.dz, 2)
        )
        return -conv + nu_u * _laplacian_field(u, g.dx, g.dy, g.dz)

    def _momentum_H_v(self, u, v, w, nu_v) -> np.ndarray:
        g = self.grid
        u_v = self._interp_u_to_v(u)
        w_v = self._interp_w_to_v(w)
        conv = (
            u_v * _upwind_1d(v, u_v, g.dx, 0)
            + v * _upwind_1d(v, v, g.dy, 1)
            + w_v * _upwind_1d(v, w_v, g.dz, 2)
        )
        return -conv + nu_v * _laplacian_field(v, g.dx, g.dy, g.dz)

    def _momentum_H_w(self, u, v, w, nu_w) -> np.ndarray:
        g = self.grid
        u_w = self._interp_u_to_w(u)
        v_w = self._interp_v_to_w(v)
        conv = (
            u_w * _upwind_1d(w, u_w, g.dx, 0)
            + v_w * _upwind_1d(w, v_w, g.dy, 1)
            + w * _upwind_1d(w, w, g.dz, 2)
        )
        return -conv + nu_w * _laplacian_field(w, g.dx, g.dy, g.dz)

    def _pressure_gradient_faces(
        self, p: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = self.grid
        dpdx = np.zeros(g.shape_u)
        dpdy = np.zeros(g.shape_v)
        dpdz = np.zeros(g.shape_w)
        dpdx[1:-1] = (p[1:] - p[:-1]) / g.dx
        dpdy[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / g.dy
        dpdz[:, :, 1:-1] = (p[:, :, 1:] - p[:, :, :-1]) / g.dz
        return dpdx, dpdy, dpdz

    def _divergence(self, u, v, w) -> np.ndarray:
        g = self.grid
        div = (
            (u[1:] - u[:-1]) / g.dx
            + (v[:, 1:] - v[:, :-1]) / g.dy
            + (w[:, :, 1:] - w[:, :, :-1]) / g.dz
        )
        div[self._obstacle] = 0.0
        return div

    def _build_laplacian(self) -> sparse.csr_matrix:
        g = self.grid
        nx, ny, nz = g.shape
        n = nx * ny * nz
        dx2, dy2, dz2 = g.dx**2, g.dy**2, g.dz**2
        rows, cols, data = [], [], []

        def add(r, c, val):
            rows.append(r)
            cols.append(c)
            data.append(val)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = i * ny * nz + j * nz + k
                    diag = 0.0
                    if i > 0:
                        add(idx, idx - ny * nz, 1.0 / dx2)
                        diag -= 1.0 / dx2
                    if i < nx - 1:
                        add(idx, idx + ny * nz, 1.0 / dx2)
                        diag -= 1.0 / dx2
                    if j > 0:
                        add(idx, idx - nz, 1.0 / dy2)
                        diag -= 1.0 / dy2
                    if j < ny - 1:
                        add(idx, idx + nz, 1.0 / dy2)
                        diag -= 1.0 / dy2
                    if k > 0:
                        add(idx, idx - 1, 1.0 / dz2)
                        diag -= 1.0 / dz2
                    if k < nz - 1:
                        add(idx, idx + 1, 1.0 / dz2)
                        diag -= 1.0 / dz2
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
        p_flat, info = cg(self._lap_cache, rhs, rtol=1e-6, maxiter=500)
        if info != 0 or not np.all(np.isfinite(p_flat)):
            p_flat = spsolve(self._lap_cache, rhs)
        p_corr = np.asarray(p_flat, dtype=float).reshape(self.grid.shape, order="C")
        p_corr = np.nan_to_num(p_corr, nan=0.0, posinf=0.0, neginf=0.0)
        p_corr -= p_corr.mean()
        return p_corr

    def _les_nu_eff(self, u, v, w, dt: float) -> np.ndarray:
        g = self.grid
        if self.use_les:
            nu_eff = les.smagorinsky_viscosity(u, v, w, g, self.Cs, self.nu)
            nu_eff = np.clip(nu_eff, self.nu, 50.0 * self.nu + 1.0)
        else:
            nu_eff = np.full(g.shape, self.nu)
        nu_eff = np.maximum(nu_eff, self.nu)
        nu_stab = 0.15 / (
            dt * (1.0 / g.dx**2 + 1.0 / g.dy**2 + 1.0 / g.dz**2)
        )
        return np.minimum(nu_eff, max(nu_stab, self.nu))

    def _nu_on_faces(self, nu_eff):
        return (
            self._cells_to_u_faces(nu_eff),
            self._cells_to_v_faces(nu_eff),
            self._cells_to_w_faces(nu_eff),
        )

    def step(self, dt: float) -> FluidState:
        self.t += dt
        u, v, w = self._sanitize(self.state.u, self.state.v, self.state.w)
        p = np.nan_to_num(self.state.p, nan=0.0)

        nu_eff = self._les_nu_eff(u, v, w, dt)
        nu_u, nu_v, nu_w = self._nu_on_faces(nu_eff)

        Hu = self._momentum_H_u(u, v, w, nu_u)
        Hv = self._momentum_H_v(u, v, w, nu_v)
        Hw = self._momentum_H_w(u, v, w, nu_w)
        dpdx, dpdy, dpdz = self._pressure_gradient_faces(p)
        u_s = u + dt * (Hu - dpdx / self.rho)
        v_s = v + dt * (Hv - dpdy / self.rho)
        w_s = w + dt * (Hw - dpdz / self.rho)
        u_s, v_s, w_s = self._sanitize(u_s, v_s, w_s)
        u_s, v_s, w_s = self._apply_bc(u_s, v_s, w_s)

        for corr in range(self.n_correctors):
            if corr > 0:
                Hu = self._momentum_H_u(u_s, v_s, w_s, nu_u)
                Hv = self._momentum_H_v(u_s, v_s, w_s, nu_v)
                Hw = self._momentum_H_w(u_s, v_s, w_s, nu_w)
                dpdx, dpdy, dpdz = self._pressure_gradient_faces(p)
                u_s = u + dt * (Hu - dpdx / self.rho)
                v_s = v + dt * (Hv - dpdy / self.rho)
                w_s = w + dt * (Hw - dpdz / self.rho)
                u_s, v_s, w_s = self._sanitize(u_s, v_s, w_s)
                u_s, v_s, w_s = self._apply_bc(u_s, v_s, w_s)

            div = self._divergence(u_s, v_s, w_s)
            p_corr = self._solve_pressure_correction(div, dt)
            cdx, cdy, cdz = self._pressure_gradient_faces(p_corr)
            u_s = u_s - (dt / self.rho) * cdx
            v_s = v_s - (dt / self.rho) * cdy
            w_s = w_s - (dt / self.rho) * cdz
            p = p + p_corr
            u_s, v_s, w_s = self._sanitize(u_s, v_s, w_s)
            u_s, v_s, w_s = self._apply_bc(u_s, v_s, w_s)

        p = np.nan_to_num(p, nan=0.0)
        p -= p.mean()
        self.state = FluidState(u=u_s, v=v_s, w=w_s, p=p, nu_eff=nu_eff)
        return self.state

    def force_coefficients(
        self, cylinder_D: float, surface_mask: np.ndarray
    ) -> Tuple[float, float]:
        """Approximate Cd, Cl from pressure on a near-surface fluid band.

        Uses ``F ≈ -p n ΔA`` with outward normal from cylinder centre projected
        in the *xy*-plane; spanwise-integrated and normalised by
        ``½ ρ U² D H``.
        """
        g = self.grid
        p = self.state.p
        # Cylinder centre inferred from solid mask centroid in xy
        solid = self._obstacle
        if not np.any(solid):
            return 0.0, 0.0
        # Use provided surface band
        band = surface_mask
        if not np.any(band):
            return 0.0, 0.0

        # Recover cx, cy from solid cells
        xs = g.X[solid]
        ys = g.Y[solid]
        cx, cy = float(xs.mean()), float(ys.mean())

        dx, dy, dz = g.dx, g.dy, g.dz
        dA = np.sqrt(dx * dy) * dz  # rough face area scale per cell column slice
        # Better: treat each band cell as contributing n̂ * p * (dx*dz or dy*dz)
        rx = g.X[band] - cx
        ry = g.Y[band] - cy
        r = np.sqrt(rx * rx + ry * ry) + 1e-12
        nx_hat = rx / r
        ny_hat = ry / r
        # Area per sampled cell ~ min(dx,dy) * dz (radial ring thickness × span)
        area = min(dx, dy) * dz
        px = p[band]
        Fx = float(np.sum(-px * nx_hat * area))
        Fy = float(np.sum(-px * ny_hat * area))
        q = 0.5 * self.rho * self.U_inf**2
        ref = q * cylinder_D * g.H + 1e-12
        return Fx / ref, Fy / ref

    def max_cfl(self, dt: float) -> float:
        g = self.grid
        uc, vc, wc = self.state.cell_centered_velocity()
        umax = max(
            float(np.max(np.abs(uc))),
            float(np.max(np.abs(vc))),
            float(np.max(np.abs(wc))),
            1e-12,
        )
        return umax * dt / min(g.dx, g.dy, g.dz)
