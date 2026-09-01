"""Incompressible Navier–Stokes on a MAC staggered Cartesian grid — PISO + LES.

Discretisation
--------------
1. **MAC staggering** — pressure / ν_eff at cell centres ``(nx, ny, nz)``;
   ``u`` on *x*-faces ``(nx+1, ny, nz)``, ``v`` on *y*-faces ``(nx, ny+1, nz)``,
   ``w`` on *z*-faces ``(nx, ny, nz+1)``.
2. **LES (Smagorinsky)** — interpolate face velocities to cell centres, form
   ν_eff = ν + ν_t, then average ν_eff back to faces for the viscous term.
3. **Momentum** — first-order upwind convection + 7-point viscous Laplacian
   with face ν_eff (explicit neighbour fluxes H(u)).
4. **PISO** (Issa, 1986) — momentum predictor with the old face pressure
   gradient, then ``n_correctors`` pressure–velocity correctors. Correctors
   after the first rebuild H(u) from the latest velocity, then re-correct
   pressure. Divergence and the pressure Poisson equation are cell-centred;
   velocity corrections use one-sided face pressure differences.
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
    """Staggered velocity + cell-centred pressure / eddy viscosity."""

    u: np.ndarray  # (nx+1, ny, nz)
    v: np.ndarray  # (nx, ny+1, nz)
    w: np.ndarray  # (nx, ny, nz+1)
    p: np.ndarray  # (nx, ny, nz)
    nu_eff: np.ndarray  # (nx, ny, nz)

    def cell_centered_velocity(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Average face velocities onto cell centres (for plotting / CFL)."""
        uc = 0.5 * (self.u[:-1] + self.u[1:])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        wc = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return uc, vc, wc


def _laplacian_field(
    phi: np.ndarray, dx: float, dy: float, dz: float
) -> np.ndarray:
    """7-point Laplacian with Neumann (copy) boundaries — no wrap."""
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


def _upwind_1d(
    phi: np.ndarray, vel: np.ndarray, d: float, axis: int
) -> np.ndarray:
    """First-order upwind derivative of ``phi`` advected by ``vel`` along axis."""
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
    """MAC-staggered incompressible NS with Smagorinsky LES + PISO.

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

        # Inlet profile on cell-centred (y, z); applied on the west u-face.
        py = np.sin(np.pi * (np.arange(grid.ny) + 0.5) / grid.ny)
        pz = np.sin(np.pi * (np.arange(grid.nz) + 0.5) / grid.nz)
        self._inlet_profile = (py[:, None] * pz[None, :]) ** 0.25

        self.state = FluidState(
            u=np.full(grid.shape_u, self.U_inlet * 0.5),
            v=np.zeros(grid.shape_v),
            w=np.zeros(grid.shape_w),
            p=np.zeros(grid.shape),
            nu_eff=np.full(grid.shape, self.nu),
        )
        self.state.u[0, :, :] = self.U_inlet * self._inlet_profile

        self._obstacle = np.zeros(grid.shape, dtype=bool)
        self._u_solid = np.zeros(grid.shape_u)
        self._v_solid = np.zeros(grid.shape_v)
        self._w_solid = np.zeros(grid.shape_w)
        self._lap_cache: Optional[sparse.csr_matrix] = None

    def set_immersed_boundary(
        self,
        mask: np.ndarray,
        u_s: Optional[np.ndarray] = None,
        v_s: Optional[np.ndarray] = None,
        w_s: Optional[np.ndarray] = None,
    ) -> None:
        """Set solid *cell* mask and optional face solid velocities.

        ``mask`` is cell-centred. Face solid velocities may be passed already
        on the staggered shapes, or as cell-centred arrays (interpolated).
        """
        self._obstacle = np.asarray(mask, dtype=bool)
        g = self.grid
        if u_s is None:
            self._u_solid = np.zeros(g.shape_u)
        else:
            u_s = np.asarray(u_s, dtype=float)
            self._u_solid = (
                u_s if u_s.shape == g.shape_u else self._cells_to_u_faces(u_s)
            )
        if v_s is None:
            self._v_solid = np.zeros(g.shape_v)
        else:
            v_s = np.asarray(v_s, dtype=float)
            self._v_solid = (
                v_s if v_s.shape == g.shape_v else self._cells_to_v_faces(v_s)
            )
        if w_s is None:
            self._w_solid = np.zeros(g.shape_w)
        else:
            w_s = np.asarray(w_s, dtype=float)
            self._w_solid = (
                w_s if w_s.shape == g.shape_w else self._cells_to_w_faces(w_s)
            )

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
        """True on u-faces that bound at least one solid cell."""
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
        # No-slip on y / z walls (tangential + normal)
        u[:, 0, :] = 0.0
        u[:, -1, :] = 0.0
        u[:, :, 0] = 0.0
        u[:, :, -1] = 0.0
        v[:, 0, :] = 0.0
        v[:, -1, :] = 0.0
        v[:, :, 0] = 0.0
        v[:, :, -1] = 0.0
        v[0, :, :] = 0.0
        v[-1, :, :] = 0.0
        w[:, 0, :] = 0.0
        w[:, -1, :] = 0.0
        w[:, :, 0] = 0.0
        w[:, :, -1] = 0.0
        w[0, :, :] = 0.0
        w[-1, :, :] = 0.0

        # Outlet: Neumann on east faces
        u[-1, :, :] = u[-2, :, :]
        v[-1, :, :] = v[-2, :, :]
        w[-1, :, :] = w[-2, :, :]

        if self.gust_amp > 0.0:
            om = 2.0 * np.pi * self.gust_freq
            u_in = self.U_inlet * (1.0 + 0.5 * self.gust_amp * np.sin(om * self.t))
            w_in = self.U_inlet * self.gust_amp * (
                np.sin(om * self.t) + 0.4 * np.sin(0.37 * om * self.t + 1.3)
            )
        else:
            u_in = self.U_inlet
            w_in = 0.0

        # Inlet on west u-face; tangential velocities near inlet
        u[0, :, :] = u_in * self._inlet_profile
        v[0, :, :] = 0.0
        if abs(w_in) > 0.0:
            # Map cell-centred (ny, nz) profile onto west w-faces (ny, nz+1)
            prof = self._inlet_profile
            prof_w = np.zeros((self.grid.ny, self.grid.nz + 1))
            prof_w[:, 1:-1] = 0.5 * (prof[:, :-1] + prof[:, 1:])
            prof_w[:, 0] = prof[:, 0]
            prof_w[:, -1] = prof[:, -1]
            w[0, :, :] = w_in * prof_w
        else:
            w[0, :, :] = 0.0

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
        """v (nx, ny+1, nz) → u-faces (nx+1, ny, nz)."""
        out = np.zeros(self.grid.shape_u)
        # Interior u-faces: average of 4 surrounding v faces in y, then in x
        vc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])  # (nx, ny, nz) at cell centres
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

    def _momentum_H_u(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        nu_u: np.ndarray,
    ) -> np.ndarray:
        g = self.grid
        v_u = self._interp_v_to_u(v)
        w_u = self._interp_w_to_u(w)
        conv = (
            u * _upwind_1d(u, u, g.dx, 0)
            + v_u * _upwind_1d(u, v_u, g.dy, 1)
            + w_u * _upwind_1d(u, w_u, g.dz, 2)
        )
        return -conv + nu_u * _laplacian_field(u, g.dx, g.dy, g.dz)

    def _momentum_H_v(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        nu_v: np.ndarray,
    ) -> np.ndarray:
        g = self.grid
        u_v = self._interp_u_to_v(u)
        w_v = self._interp_w_to_v(w)
        conv = (
            u_v * _upwind_1d(v, u_v, g.dx, 0)
            + v * _upwind_1d(v, v, g.dy, 1)
            + w_v * _upwind_1d(v, w_v, g.dz, 2)
        )
        return -conv + nu_v * _laplacian_field(v, g.dx, g.dy, g.dz)

    def _momentum_H_w(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        nu_w: np.ndarray,
    ) -> np.ndarray:
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
        """∇p on velocity faces from cell-centred pressure."""
        g = self.grid
        dpdx = np.zeros(g.shape_u)
        dpdy = np.zeros(g.shape_v)
        dpdz = np.zeros(g.shape_w)
        dpdx[1:-1] = (p[1:] - p[:-1]) / g.dx
        dpdy[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / g.dy
        dpdz[:, :, 1:-1] = (p[:, :, 1:] - p[:, :, :-1]) / g.dz
        return dpdx, dpdy, dpdz

    def _divergence(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        """Cell-centred divergence from face fluxes."""
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
        p_flat, info = cg(self._lap_cache, rhs, rtol=1e-6, maxiter=400)
        if info != 0 or not np.all(np.isfinite(p_flat)):
            p_flat = spsolve(self._lap_cache, rhs)
        p_corr = np.asarray(p_flat, dtype=float).reshape(self.grid.shape, order="C")
        p_corr = np.nan_to_num(p_corr, nan=0.0, posinf=0.0, neginf=0.0)
        p_corr -= p_corr.mean()
        return p_corr

    def _les_nu_eff(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray, dt: float
    ) -> np.ndarray:
        g = self.grid
        if self.use_les:
            nu_eff = LES_smagorinsky_velocity.smagorinsky_viscosity(
                u, v, w, g, self.Cs, self.nu
            )
            nu_eff = np.clip(nu_eff, self.nu, 50.0 * self.nu + 1.0)
        else:
            nu_eff = np.full(g.shape, self.nu)
        nu_eff = np.maximum(nu_eff, self.nu)
        nu_stab = 0.2 / (
            dt * (1.0 / g.dx**2 + 1.0 / g.dy**2 + 1.0 / g.dz**2)
        )
        return np.minimum(nu_eff, max(nu_stab, self.nu))

    def _nu_on_faces(
        self, nu_eff: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self._cells_to_u_faces(nu_eff),
            self._cells_to_v_faces(nu_eff),
            self._cells_to_w_faces(nu_eff),
        )

    def step(self, dt: float) -> FluidState:
        """One PISO time step on the staggered grid after Smagorinsky LES."""
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

    def sample_at(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Trilinear sample of cell-centred velocity and pressure."""
        g = self.grid
        pts = np.asarray(points, dtype=float)
        pts = np.nan_to_num(pts, nan=0.0)
        uc, vc, wc = self.state.cell_centered_velocity()

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

        vel = np.column_stack([trilin(uc), trilin(vc), trilin(wc)])
        pressure = trilin(self.state.p)
        return vel, pressure

    def max_cfl(self, dt: float) -> float:
        g = self.grid
        uc, vc, wc = self.state.cell_centered_velocity()
        umax = max(
            float(np.max(np.abs(uc))),
            float(np.max(np.abs(vc))),
            float(np.max(np.abs(wc))),
            1e-12,
        )
        hmin = min(g.dx, g.dy, g.dz)
        return umax * dt / hmin
