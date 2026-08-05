"""Incompressible Navier–Stokes on a collocated Cartesian grid — PISO + LES.

Discretisation
--------------
1. **LES (Smagorinsky)** — evaluate subgrid eddy viscosity ν_t from the
   resolved strain-rate tensor and form ν_eff = ν + ν_t.
2. **Momentum** — first-order upwind convection + 7-point viscous Laplacian
   with ν_eff (explicit neighbour fluxes H(u)).
3. **PISO** (Issa, 1986) — momentum predictor with the old pressure gradient,
   then ``n_correctors`` pressure–velocity correctors. Correctors after the
   first rebuild H(u) from the latest velocity (the PISO neighbour update),
   then re-correct pressure.

This replaces the earlier single-projection “PISO-like” fractional step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve

import LES_smagorinsky_velocity
from mesh import FluidGrid


@dataclass
class FluidState:
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    p: np.ndarray
    nu_eff: np.ndarray


def _diff_central(phi: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """Central difference with one-sided edges (no periodic wrap)."""
    out = np.zeros_like(phi)
    sl_c = [slice(None)] * phi.ndim
    sl_m = [slice(None)] * phi.ndim
    sl_p = [slice(None)] * phi.ndim
    sl_c[axis] = slice(1, -1)
    sl_m[axis] = slice(0, -2)
    sl_p[axis] = slice(2, None)
    out[tuple(sl_c)] = (phi[tuple(sl_p)] - phi[tuple(sl_m)]) / (2.0 * dx)

    sl0 = [slice(None)] * phi.ndim
    sl1 = [slice(None)] * phi.ndim
    sl0[axis] = 0
    sl1[axis] = 1
    out[tuple(sl0)] = (phi[tuple(sl1)] - phi[tuple(sl0)]) / dx

    sln = [slice(None)] * phi.ndim
    slnm = [slice(None)] * phi.ndim
    sln[axis] = -1
    slnm[axis] = -2
    out[tuple(sln)] = (phi[tuple(sln)] - phi[tuple(slnm)]) / dx
    return out


def _laplacian(phi: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """7-point Laplacian with Neumann (copy) boundaries — no wrap."""
    lap = np.zeros_like(phi)
    lap[1:-1] += (phi[2:] - 2 * phi[1:-1] + phi[0:-2]) / dx**2
    lap[0] += (phi[1] - phi[0]) / dx**2
    lap[-1] += (phi[-2] - phi[-1]) / dx**2
    lap[:, 1:-1] += (phi[:, 2:] - 2 * phi[:, 1:-1] + phi[:, 0:-2]) / dy**2
    lap[:, 0] += (phi[:, 1] - phi[:, 0]) / dy**2
    lap[:, -1] += (phi[:, -2] - phi[:, -1]) / dy**2
    lap[:, :, 1:-1] += (phi[:, :, 2:] - 2 * phi[:, :, 1:-1] + phi[:, :, 0:-2]) / dz**2
    lap[:, :, 0] += (phi[:, :, 1] - phi[:, :, 0]) / dz**2
    lap[:, :, -1] += (phi[:, :, -2] - phi[:, :, -1]) / dz**2
    return lap


class FluidSolver:
    """Collocated incompressible NS with Smagorinsky LES + PISO.

    Parameters
    ----------
    n_correctors :
        Number of PISO pressure–velocity correctors (Issa: typically 2).
        The first corrector uses the momentum predictor; later correctors
        rebuild the explicit momentum residual H(u) from the latest velocity
        before solving the pressure equation again.
    """

    def __init__(
        self,
        grid: FluidGrid,
        rho: float = 1.225,
        nu: float = 1.5e-5,
        U_inlet: float = 10.0,
        use_les: bool = True,
        Cs: float = 0.17,
        u_clip: Optional[float] = None,
        gust_amp: float = 0.0,
        gust_freq: float = 1.0,
        n_correctors: int = 2,
    ) -> None:
        self.grid = grid
        self.rho = float(rho)
        self.nu = float(nu)
        self.U_inlet = float(U_inlet)
        self.use_les = bool(use_les)
        self.Cs = float(Cs)
        self.u_clip = float(u_clip) if u_clip is not None else 5.0 * abs(self.U_inlet)
        self.gust_amp = float(gust_amp)
        self.gust_freq = float(gust_freq)
        self.n_correctors = max(int(n_correctors), 1)
        self.t = 0.0
        py = np.sin(np.pi * (np.arange(grid.ny) + 0.5) / grid.ny)
        pz = np.sin(np.pi * (np.arange(grid.nz) + 0.5) / grid.nz)
        self._inlet_profile = (py[:, None] * pz[None, :]) ** 0.25

        sh = grid.shape
        self.state = FluidState(
            u=np.full(sh, self.U_inlet * 0.5),
            v=np.zeros(sh),
            w=np.zeros(sh),
            p=np.zeros(sh),
            nu_eff=np.full(sh, self.nu),
        )
        self.state.u[0, :, :] = self.U_inlet * self._inlet_profile
        self._obstacle = np.zeros(sh, dtype=bool)
        self._u_solid = np.zeros(sh)
        self._v_solid = np.zeros(sh)
        self._w_solid = np.zeros(sh)
        self._lap_cache: Optional[sparse.csr_matrix] = None

    def set_immersed_boundary(
        self,
        mask: np.ndarray,
        u_s: Optional[np.ndarray] = None,
        v_s: Optional[np.ndarray] = None,
        w_s: Optional[np.ndarray] = None,
    ) -> None:
        self._obstacle = mask.astype(bool)
        z = np.zeros(self.grid.shape)
        self._u_solid = z if u_s is None else np.asarray(u_s, dtype=float)
        self._v_solid = z if v_s is None else np.asarray(v_s, dtype=float)
        self._w_solid = z if w_s is None else np.asarray(w_s, dtype=float)

    def _apply_bc(self, u, v, w) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for arr in (u, v, w):
            arr[:, 0, :] = 0.0
            arr[:, -1, :] = 0.0
            arr[:, :, 0] = 0.0
            arr[:, :, -1] = 0.0
        u[-1, :, :] = u[-2, :, :]
        v[-1, :, :] = v[-2, :, :]
        w[-1, :, :] = w[-2, :, :]
        if self.gust_amp > 0.0:
            om = 2.0 * np.pi * self.gust_freq
            u_in = self.U_inlet * (1.0 + 0.5 * self.gust_amp * np.sin(om * self.t))
            w_in = self.U_inlet * self.gust_amp * (
                np.sin(om * self.t)
                + 0.4 * np.sin(0.37 * om * self.t + 1.3)
            )
        else:
            u_in = self.U_inlet
            w_in = 0.0
        u[0, :, :] = u_in * self._inlet_profile
        v[0, :, :] = 0.0
        w[0, :, :] = w_in * self._inlet_profile
        m = self._obstacle
        if np.any(m):
            u[m] = self._u_solid[m]
            v[m] = self._v_solid[m]
            w[m] = self._w_solid[m]
        return u, v, w

    def _sanitize(self, *fields: np.ndarray) -> Tuple[np.ndarray, ...]:
        out = []
        for f in fields:
            g = np.nan_to_num(f, nan=0.0, posinf=self.u_clip, neginf=-self.u_clip)
            out.append(np.clip(g, -self.u_clip, self.u_clip))
        return tuple(out)

    def _convective(
        self,
        phi: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
    ) -> np.ndarray:
        """First-order upwind convective derivative u·∇φ."""
        g = self.grid
        dx, dy, dz = g.dx, g.dy, g.dz
        dudx = np.zeros_like(phi)
        dudy = np.zeros_like(phi)
        dudz = np.zeros_like(phi)
        dudx[1:-1] = np.where(
            u[1:-1] >= 0,
            (phi[1:-1] - phi[:-2]) / dx,
            (phi[2:] - phi[1:-1]) / dx,
        )
        dudy[:, 1:-1] = np.where(
            v[:, 1:-1] >= 0,
            (phi[:, 1:-1] - phi[:, :-2]) / dy,
            (phi[:, 2:] - phi[:, 1:-1]) / dy,
        )
        dudz[:, :, 1:-1] = np.where(
            w[:, :, 1:-1] >= 0,
            (phi[:, :, 1:-1] - phi[:, :, :-2]) / dz,
            (phi[:, :, 2:] - phi[:, :, 1:-1]) / dz,
        )
        return u * dudx + v * dudy + w * dudz

    def _momentum_H(
        self,
        phi: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        nu_eff: np.ndarray,
    ) -> np.ndarray:
        """Explicit momentum residual without pressure: H = -conv + ν_eff ∇²φ."""
        g = self.grid
        conv = self._convective(phi, u, v, w)
        lap = _laplacian(phi, g.dx, g.dy, g.dz)
        return -conv + nu_eff * lap

    def _pressure_gradient(
        self, p: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = self.grid
        return (
            _diff_central(p, g.dx, 0),
            _diff_central(p, g.dy, 1),
            _diff_central(p, g.dz, 2),
        )

    def _divergence(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        g = self.grid
        div = (
            _diff_central(u, g.dx, 0)
            + _diff_central(v, g.dy, 1)
            + _diff_central(w, g.dz, 2)
        )
        div[self._obstacle] = 0.0
        return div

    def _build_laplacian(self) -> sparse.csr_matrix:
        g = self.grid
        nx, ny, nz = g.shape
        n = nx * ny * nz
        dx2, dy2, dz2 = g.dx**2, g.dy**2, g.dz**2
        rows, cols, data = [], [], []

        def add(r, c, v):
            rows.append(r)
            cols.append(c)
            data.append(v)

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
        """Solve ∇² p' = (ρ / Δt) ∇·u*  (Neumann, pinned cell)."""
        if self._lap_cache is None:
            self._lap_cache = self._build_laplacian()
        rhs = (self.rho / max(dt, 1e-12)) * div.ravel(order="C")
        rhs = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)
        rhs -= rhs.mean()
        rhs[0] = 0.0
        p_flat, info = cg(self._lap_cache, rhs, tol=1e-6, maxiter=400)   ### check this it is supposed to be rtol not tol, misspelled by mistake and cannot find other rtol 
        if info != 0 or not np.all(np.isfinite(p_flat)):
            p_flat = spsolve(self._lap_cache, rhs)
        p_corr = np.asarray(p_flat, dtype=float).reshape(self.grid.shape, order="C")
        p_corr = np.nan_to_num(p_corr, nan=0.0, posinf=0.0, neginf=0.0)
        p_corr -= p_corr.mean()
        return p_corr

    def _les_nu_eff(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray, dt: float
    ) -> np.ndarray:
        """Smagorinsky LES discretisation → ν_eff, capped for explicit stability."""
        g = self.grid
        if self.use_les:
            nu_eff = LES_smagorinsky_velocity.smagorinsky_viscosity(u, v, w, g, self.Cs, self.nu)
            nu_eff = np.clip(nu_eff, self.nu, 50.0 * self.nu + 1.0)
        else:
            nu_eff = np.full(g.shape, self.nu)
        nu_eff = np.maximum(nu_eff, self.nu)
        nu_stab = 0.2 / (
            dt * (1.0 / g.dx**2 + 1.0 / g.dy**2 + 1.0 / g.dz**2)
        )
        return np.minimum(nu_eff, max(nu_stab, self.nu))

    def step(self, dt: float) -> FluidState:
        """One PISO time step after Smagorinsky LES discretisation."""
        self.t += dt
        g = self.grid
        u, v, w = self._sanitize(self.state.u, self.state.v, self.state.w)
        p = np.nan_to_num(self.state.p, nan=0.0)

        # ------------------------------------------------------------------
        # 1) LES Smagorinsky — discretise subgrid viscosity from strain rate
        # ------------------------------------------------------------------
        nu_eff = self._les_nu_eff(u, v, w, dt)

        # ------------------------------------------------------------------
        # 2) Momentum predictor (Issa): H(u^n) − ∇p^n / ρ
        # ------------------------------------------------------------------
        Hu = self._momentum_H(u, u, v, w, nu_eff)
        Hv = self._momentum_H(v, u, v, w, nu_eff)
        Hw = self._momentum_H(w, u, v, w, nu_eff)
        dpdx, dpdy, dpdz = self._pressure_gradient(p)
        u_s = u + dt * (Hu - dpdx / self.rho)
        v_s = v + dt * (Hv - dpdy / self.rho)
        w_s = w + dt * (Hw - dpdz / self.rho)
        u_s, v_s, w_s = self._sanitize(u_s, v_s, w_s)
        u_s, v_s, w_s = self._apply_bc(u_s, v_s, w_s)

        # ------------------------------------------------------------------
        # 3) PISO pressure–velocity correctors
        # ------------------------------------------------------------------
        for corr in range(self.n_correctors):
            if corr > 0:
                # Neighbour / flux update with latest corrected velocity
                # (distinguishes PISO from a single fractional-step projection).
                Hu = self._momentum_H(u_s, u_s, v_s, w_s, nu_eff)
                Hv = self._momentum_H(v_s, u_s, v_s, w_s, nu_eff)
                Hw = self._momentum_H(w_s, u_s, v_s, w_s, nu_eff)
                dpdx, dpdy, dpdz = self._pressure_gradient(p)
                u_s = u + dt * (Hu - dpdx / self.rho)
                v_s = v + dt * (Hv - dpdy / self.rho)
                w_s = w + dt * (Hw - dpdz / self.rho)
                u_s, v_s, w_s = self._sanitize(u_s, v_s, w_s)
                u_s, v_s, w_s = self._apply_bc(u_s, v_s, w_s)

            div = self._divergence(u_s, v_s, w_s)
            p_corr = self._solve_pressure_correction(div, dt)
            cdx, cdy, cdz = self._pressure_gradient(p_corr)
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

    def sample_at(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g = self.grid
        pts = np.asarray(points, dtype=float)
        pts = np.nan_to_num(pts, nan=0.0)
        fi = np.clip(pts[:, 0] / g.dx - 0.5, 0.0, g.nx - 1.001)
        fj = np.clip(pts[:, 1] / g.dy - 0.5, 0.0, g.ny - 1.001)
        fk = np.clip(pts[:, 2] / g.dz - 0.5, 0.0, g.nz - 1.001)
        i0 = np.floor(fi).astype(int)
        j0 = np.floor(fj).astype(int)
        k0 = np.floor(fk).astype(int)
        i1 = np.minimum(i0 + 1, g.nx - 1)
        j1 = np.minimum(j0 + 1, g.ny - 1)
        k1 = np.minimum(k0 + 1, g.nz - 1)
        wx, wy, wz = fi - i0, fj - j0, fk - k0

        def trilin(field):
            field = np.nan_to_num(field, nan=0.0)
            c000 = field[i0, j0, k0]
            c100 = field[i1, j0, k0]
            c010 = field[i0, j1, k0]
            c110 = field[i1, j1, k0]
            c001 = field[i0, j0, k1]
            c101 = field[i1, j0, k1]
            c011 = field[i0, j1, k1]
            c111 = field[i1, j1, k1]
            c00 = c000 * (1 - wx) + c100 * wx
            c01 = c001 * (1 - wx) + c101 * wx
            c10 = c010 * (1 - wx) + c110 * wx
            c11 = c011 * (1 - wx) + c111 * wx
            c0 = c00 * (1 - wy) + c10 * wy
            c1 = c01 * (1 - wy) + c11 * wy
            return c0 * (1 - wz) + c1 * wz

        vel = np.column_stack(
            [trilin(self.state.u), trilin(self.state.v), trilin(self.state.w)]
        )
        pressure = trilin(self.state.p)
        return vel, pressure

    def max_cfl(self, dt: float) -> float:
        g = self.grid
        umax = max(
            float(np.max(np.abs(self.state.u))),
            float(np.max(np.abs(self.state.v))),
            float(np.max(np.abs(self.state.w))),
            1e-12,
        )
        hmin = min(g.dx, g.dy, g.dz)
        return umax * dt / hmin