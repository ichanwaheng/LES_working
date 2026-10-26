#!/usr/bin/env python3
"""
ns_staggered_euler_3d.py
========================
3-D incompressible Navier–Stokes on the spanwise extrusion of the 2-D O–H
staggered mesh (``mesh_from_known_sizes_3d.py``). Existing 2-D files are
imported for the xy section only; they are not modified.

Per step (Griebel / CaptainProton42, same order as the 2-D solver)::

    1. apply BCs on Un
    2. F, G, H = explicit Euler predictors (no pressure)
         F = u + Δt · ( ∇·(ν_eff ∇u) − conv_γ(u) )
         G = v + Δt · ( ∇·(ν_eff ∇v) − conv_γ(v) )
         H = w + Δt · ( ∇·(ν_eff ∇w) − conv_γ(w) )
    3. pressure Poisson  ∇²p = (ρ/Δt) ∇·U*
    4. correct staggered face flux / cell velocity

ν_eff is either molecular ν (laminar, ``--no-les``) or Smagorinsky
ν + (C_s Δ)² |S| with Δ = V^{1/3} (``--les``, the default). The viscous
operator interpolates ν_eff onto faces, so a spatially varying eddy
viscosity is consistent.

Usage
-----
    cd cylinder_re150
    python ns_staggered_euler_3d.py --quick --les --steps 20
    python ns_staggered_euler_3d.py --fine --les --scheme central --nz 8

Needs numpy, scipy, matplotlib. Optional: --Re 3900 for a high-Re LES setup
(the smoke/fine xy meshes are still the Re=150 known-sizes grids; refine
d_min before treating that as a resolved LES).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, splu

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mesh_from_known_sizes_3d as m3  # noqa: E402


def load_staggered_3d(path: Path) -> dict:
    raw = np.load(path, allow_pickle=False)
    out = dict(
        cell_centers=raw["cell_centers"].astype(np.float64),
        cell_volumes=raw["cell_volumes"].astype(np.float64),
        face_centers=raw["face_centers"].astype(np.float64),
        face_normals=raw["face_normals"].astype(np.float64),
        face_areas=raw["face_areas"].astype(np.float64),
        face_owner=raw["face_owner"].astype(np.int64),
        face_neigh=raw["face_neigh"].astype(np.int64),
        face_kind=raw["face_kind"].astype(np.int8),
        face_bc=raw["face_bc"].astype(np.int32),
        bc_names=[str(x) for x in raw["bc_names"].tolist()],
        D=float(raw["D"]),
        xc=float(raw["xc"]),
        yc=float(raw["yc"]),
        d_min=float(raw["d_min"]),
        d_max=float(raw["d_max"]),
        a=float(raw["a"]),
        nz=int(raw["nz"]),
        Lz=float(raw["Lz"]),
        dz=float(raw["dz"]),
        n_cells_2d=int(raw["n_cells_2d"]),
        periodic_z=bool(raw["periodic_z"]),
    )
    if "quads" in raw.files:
        out["quads"] = raw["quads"].astype(np.float64)
    return out


def ensure_staggered_mesh_3d(
    preset: str | None,
    nz: int,
    Lz: float,
    periodic: bool,
    build: bool,
) -> Path:
    tag = preset if preset is not None else "result"
    path = HERE / "mesh_output" / f"staggered_3d_{tag}_nz{nz}.npz"
    if path.is_file() and not build:
        print(f"[mesh-3d] reusing {path}")
        return path
    m3.main(
        preset=preset, nz=nz, Lz=Lz, periodic=periodic, build=build, out=path
    )
    return path


def build_topology(S: dict) -> dict:
    P, Fc = S["cell_centers"], S["face_centers"]
    n, A = S["face_normals"], S["face_areas"]
    own, nei = S["face_owner"], S["face_neigh"]
    Nc, Nf = len(P), len(Fc)
    vol = np.asarray(S["cell_volumes"], dtype=np.float64)
    vol = np.maximum(vol, 1e-30)

    d_pn = np.empty(Nf)
    interior = nei >= 0
    d_pn[interior] = np.linalg.norm(P[nei[interior]] - P[own[interior]], axis=1)
    bnd = ~interior
    d_pn[bnd] = 2.0 * np.linalg.norm(Fc[bnd] - P[own[bnd]], axis=1)
    d_pn = np.maximum(d_pn, 1e-30)

    bc_index = {nm: i for i, nm in enumerate(S["bc_names"])}
    face_bc = S["face_bc"]
    masks = {nm: face_bc == i for nm, i in bc_index.items()}
    masks["interior"] = interior
    masks["boundary"] = ~interior

    hmin = float(min(np.min(vol ** (1.0 / 3.0)), float(S["dz"])))
    return dict(
        vol=vol,
        d_pn=d_pn,
        masks=masks,
        bc_index=bc_index,
        Nc=Nc,
        Nf=Nf,
        hmin=hmin,
        n=n,
        A=A,
    )


def accumulate(flux, own, nei, Nc):
    out = np.zeros(Nc)
    np.add.at(out, own, flux)
    m = nei >= 0
    np.add.at(out, nei[m], -flux[m])
    return out


def donor_cell_face(phi, Un, own, nei, interior, gamma):
    """φ_f = (1−γ)·½(φ_o+φ_n) + γ·upwind."""
    phi_f = phi[own].copy()
    if not interior.any():
        return phi_f
    o, n = own[interior], nei[interior]
    central = 0.5 * (phi[o] + phi[n])
    upw = np.where(Un[interior] >= 0.0, phi[o], phi[n])
    phi_f[interior] = (1.0 - gamma) * central + gamma * upw
    return phi_f


class GriebelStaggeredNS3D:
    """3-D Griebel projection on the extruded O–H mesh, optional Smagorinsky."""

    def __init__(
        self,
        S: dict,
        Re: float = 150.0,
        U_inf: float = 1.0,
        rho: float = 1.0,
        scheme: str = "central",
        tau: float = 0.4,
        poisson: str = "lu",
        sor_max_it: int = 2000,
        sor_eps: float = 1e-8,
        perturb: float = 0.1,
        perturb_T: float = 2.0,
        les: bool = True,
        Cs: float = 0.1,
        van_driest: bool = True,
    ) -> None:
        self.S = S
        self.topo = build_topology(S)
        self.Re = float(Re)
        self.nu = float(U_inf) * float(S["D"]) / float(Re)
        self.U_inf = float(U_inf)
        self.rho = float(rho)
        self.scheme = str(scheme).lower().strip()
        if self.scheme not in ("central", "upwind", "donor"):
            raise ValueError("scheme must be central | upwind | donor")
        self.tau = float(tau)
        self.poisson = str(poisson).lower()
        self.sor_max_it = int(sor_max_it)
        self.sor_eps = float(sor_eps)
        self.perturb = float(perturb)
        self.perturb_T = float(perturb_T)
        self.les = bool(les)
        self.Cs = float(Cs)
        self.van_driest = bool(van_driest)
        self.t = 0.0
        self.sor_omega = 1.7

        Nc, Nf = self.topo["Nc"], self.topo["Nf"]
        self.u = np.full(Nc, self.U_inf)
        self.v = np.zeros(Nc)
        self.w = np.zeros(Nc)
        self.p = np.zeros(Nc)
        self.Un = np.zeros(Nf)
        self.nu_eff = np.full(Nc, self.nu)
        if self.poisson not in ("lu", "cg", "sor"):
            raise ValueError("poisson must be lu | cg | sor")
        self._A = self._build_poisson()
        self._M = self._build_preconditioner(self._A)
        self._lu = self._build_direct(self._A) if self.poisson == "lu" else None
        self._init_solenoidal()

    @staticmethod
    def _build_direct(A: sparse.csr_matrix):
        Ap = A.tolil(copy=True)
        Ap[0, :] = 0.0
        Ap[0, 0] = 1.0
        return splu(Ap.tocsc(), permc_spec="MMD_AT_PLUS_A")

    @staticmethod
    def _build_preconditioner(A: sparse.csr_matrix):
        diag = A.diagonal()
        inv = np.where(np.abs(diag) > 1e-30, 1.0 / diag, 0.0)

        def jacobi(x, inv=inv):
            return inv * x

        return LinearOperator(A.shape, matvec=jacobi)

    def _enforce_Un_bc(self, Un: np.ndarray) -> np.ndarray:
        S, m = self.S, self.topo["masks"]
        n = S["face_normals"]
        Un = Un.copy()
        if "inlet" in m:
            v_in = self._inlet_v()
            Un[m["inlet"]] = (
                self.U_inf * n[m["inlet"], 0] + v_in * n[m["inlet"], 1]
            )
        for w in ("wall_top", "wall_bottom", "cylinder", "zmin", "zmax"):
            if w in m:
                Un[m[w]] = 0.0
        return Un

    def _inlet_v(self) -> float:
        eps, T = self.perturb, self.perturb_T
        if eps <= 0.0 or self.t >= T:
            return 0.0
        return float(eps * self.U_inf * np.sin(np.pi * self.t / T))

    def _balance_outlet(self, Un: np.ndarray) -> np.ndarray:
        m = self.topo["masks"]
        if "outlet" not in m:
            return Un
        A = self.S["face_areas"]
        out = m["outlet"]
        bnd = self.S["face_neigh"] < 0
        others = bnd & ~out
        Q_other = float(np.sum(Un[others] * A[others]))
        Q_out = float(np.sum(Un[out] * A[out]))
        Un = Un.copy()
        Un[out] += -(Q_other + Q_out) / float(np.sum(A[out]))
        return Un

    def _gamma(self, dt: float) -> float:
        if self.scheme == "central":
            return 0.0
        if self.scheme == "upwind":
            return 1.0
        umax = max(float(np.max(np.abs(self.Un))), abs(self.U_inf), 1e-12)
        return float(min(1.0, umax * dt / self.topo["hmin"]))

    def stable_dt(self) -> float:
        h = self.topo["hmin"]
        umax = max(float(np.max(np.abs(self.Un))), abs(self.U_inf), 1e-12)
        nu_max = float(np.max(self.nu_eff))
        dt_conv = h / umax
        dt_diff = h * h / (6.0 * max(nu_max, 1e-30))
        return float(self.tau * min(dt_conv, dt_diff))

    def _face_velocity_cartesian(self, u, v, w, Un, gamma):
        S, topo = self.S, self.topo
        own, nei = S["face_owner"], S["face_neigh"]
        interior = topo["masks"]["interior"]
        uf = donor_cell_face(u, Un, own, nei, interior, gamma)
        vf = donor_cell_face(v, Un, own, nei, interior, gamma)
        wf = donor_cell_face(w, Un, own, nei, interior, gamma)
        m = topo["masks"]
        if "inlet" in m:
            uf[m["inlet"]], vf[m["inlet"]], wf[m["inlet"]] = self.U_inf, 0.0, 0.0
        for wall in ("wall_top", "wall_bottom", "cylinder", "zmin", "zmax"):
            if wall in m:
                uf[m[wall]] = 0.0
                vf[m[wall]] = 0.0
                wf[m[wall]] = 0.0
        return uf, vf, wf

    def _convective(self, u, v, w, Un, gamma):
        S, topo = self.S, self.topo
        A = S["face_areas"]
        Un = self._enforce_Un_bc(Un)
        uf, vf, wf = self._face_velocity_cartesian(u, v, w, Un, gamma)
        ru = accumulate(Un * uf * A, S["face_owner"], S["face_neigh"], topo["Nc"])
        rv = accumulate(Un * vf * A, S["face_owner"], S["face_neigh"], topo["Nc"])
        rw = accumulate(Un * wf * A, S["face_owner"], S["face_neigh"], topo["Nc"])
        V = topo["vol"]
        return -ru / V, -rv / V, -rw / V

    def _nu_on_faces(self, nu_eff: np.ndarray) -> np.ndarray:
        S, topo = self.S, self.topo
        own, nei = S["face_owner"], S["face_neigh"]
        interior = topo["masks"]["interior"]
        nu_f = nu_eff[own].copy()
        nu_f[interior] = 0.5 * (nu_eff[own[interior]] + nu_eff[nei[interior]])
        return nu_f

    def _laplacian(self, phi, nu_f, dirichlet_u: bool = False, dirichlet_w: bool = False):
        """∇·(ν_f ∇φ) with ν_f already on faces."""
        S, topo = self.S, self.topo
        A, d_pn = S["face_areas"], topo["d_pn"]
        own, nei = S["face_owner"], S["face_neigh"]
        interior = topo["masks"]["interior"]
        m = topo["masks"]
        Nf = topo["Nf"]

        phi_f = phi[own].copy()
        if dirichlet_u:
            if "inlet" in m:
                phi_f[m["inlet"]] = self.U_inf
            for wall in ("wall_top", "wall_bottom", "cylinder", "zmin", "zmax"):
                if wall in m:
                    phi_f[m[wall]] = 0.0
        elif dirichlet_w:
            for wall in ("wall_top", "wall_bottom", "cylinder", "inlet", "zmin", "zmax"):
                if wall in m:
                    phi_f[m[wall]] = 0.0
        else:
            for wall in ("wall_top", "wall_bottom", "cylinder", "inlet", "zmin", "zmax"):
                if wall in m:
                    phi_f[m[wall]] = 0.0

        g = np.empty(Nf)
        g[interior] = (phi[nei[interior]] - phi[own[interior]]) / d_pn[interior]
        bnd = ~interior
        g[bnd] = (phi_f[bnd] - phi[own[bnd]]) / (0.5 * d_pn[bnd])
        return accumulate(nu_f * g * A, own, nei, topo["Nc"]) / topo["vol"]

    def _div(self, Un):
        S, topo = self.S, self.topo
        return (
            accumulate(Un * S["face_areas"], S["face_owner"], S["face_neigh"], topo["Nc"])
            / topo["vol"]
        )

    def _grad(self, phi):
        S, topo = self.S, self.topo
        A, n = S["face_areas"], S["face_normals"]
        own, nei = S["face_owner"], S["face_neigh"]
        interior = topo["masks"]["interior"]
        pf = phi[own].copy()
        pf[interior] = 0.5 * (phi[own[interior]] + phi[nei[interior]])
        V = topo["vol"]
        gx = accumulate(pf * n[:, 0] * A, own, nei, topo["Nc"]) / V
        gy = accumulate(pf * n[:, 1] * A, own, nei, topo["Nc"]) / V
        gz = accumulate(pf * n[:, 2] * A, own, nei, topo["Nc"]) / V
        return gx, gy, gz

    def smagorinsky_viscosity(self) -> np.ndarray:
        """ν_eff = ν + (C_s Δ)² |S| at cell centres, optional van Driest."""
        if not self.les:
            return np.full(self.topo["Nc"], self.nu)
        ux, uy, uz = self._grad(self.u)
        vx, vy, vz = self._grad(self.v)
        wx, wy, wz = self._grad(self.w)
        Sxx, Syy, Szz = ux, vy, wz
        Sxy = 0.5 * (uy + vx)
        Sxz = 0.5 * (uz + wx)
        Syz = 0.5 * (vz + wy)
        S2 = (
            2.0 * (Sxx * Sxx + Syy * Syy + Szz * Szz)
            + 4.0 * (Sxy * Sxy + Sxz * Sxz + Syz * Syz)
        )
        S_mag = np.sqrt(np.maximum(S2, 0.0))
        delta = self.topo["vol"] ** (1.0 / 3.0)
        if self.van_driest:
            P = self.S["cell_centers"]
            r = np.hypot(P[:, 0] - self.S["xc"], P[:, 1] - self.S["yc"])
            y_wall = np.maximum(r - 0.5 * self.S["D"], 0.0)
            # cheap y+ using a cylinder-scale u_τ ≈ 0.04 U∞
            utau = 0.04 * self.U_inf
            yplus = y_wall * utau / max(self.nu, 1e-30)
            fd = 1.0 - np.exp(-yplus / 25.0)
            delta = delta * fd
        nu_t = (self.Cs * delta) ** 2 * S_mag
        nu_eff = self.nu + nu_t
        return np.clip(nu_eff, self.nu, 50.0 * self.nu + 1.0)

    def _build_poisson(self):
        S, topo = self.S, self.topo
        A_f, d_pn = S["face_areas"], topo["d_pn"]
        own, nei = S["face_owner"], S["face_neigh"]
        interior = topo["masks"]["interior"]
        Nc = topo["Nc"]
        o, c = own[interior], nei[interior]
        g = A_f[interior] / d_pn[interior]
        rows = np.concatenate([o, o, c, c])
        cols = np.concatenate([c, o, o, c])
        data = np.concatenate([g, -g, g, -g])
        return sparse.coo_matrix((data, (rows, cols)), shape=(Nc, Nc)).tocsr()

    def _solve_poisson(self, rhs: np.ndarray) -> tuple[np.ndarray, int]:
        rhs = rhs.copy()
        rhs -= float(np.mean(rhs))
        if self.poisson == "sor":
            return self._sor(rhs)
        if self.poisson == "lu":
            rhs[0] = 0.0
            p = self._lu.solve(rhs)
            p -= float(np.mean(p))
            return p, 0

        A = self._A
        n = A.shape[0]

        def matvec(x, A=A):
            y = A @ x
            return y - float(np.mean(y))

        Aop = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
        p, info = cg(
            Aop,
            rhs,
            x0=np.zeros(n),
            M=self._M,
            rtol=self.sor_eps,
            maxiter=self.sor_max_it,
        )
        p = np.nan_to_num(np.asarray(p), nan=0.0)
        p -= float(np.mean(p))
        return p, 0 if info == 0 else int(info)

    def _sor(self, rhs: np.ndarray) -> tuple[np.ndarray, int]:
        A = self._A
        p = np.zeros_like(rhs)
        omega = self.sor_omega
        indptr, indices, data = A.indptr, A.indices, A.data
        for it in range(self.sor_max_it):
            for i in range(len(p)):
                s = 0.0
                dii = 0.0
                for k in range(indptr[i], indptr[i + 1]):
                    j = indices[k]
                    if j == i:
                        dii = data[k]
                    else:
                        s += data[k] * p[j]
                if abs(dii) < 1e-30:
                    continue
                p_new = (rhs[i] - s) / dii
                p[i] = (1.0 - omega) * p[i] + omega * p_new
            p -= float(np.mean(p))
            res = A @ p - rhs
            res -= float(np.mean(res))
            if np.sqrt(np.mean(res * res)) <= self.sor_eps * (
                np.sqrt(np.mean(p * p)) + 0.01
            ):
                return p, it + 1
        return p, self.sor_max_it

    def _init_solenoidal(self):
        S = self.S
        n = S["face_normals"]
        Fc = S["face_centers"]
        P = S["cell_centers"]
        x0, y0 = S["xc"] + S["D"], S["yc"]
        r2f = (Fc[:, 0] - x0) ** 2 + (Fc[:, 1] - y0) ** 2
        r2c = (P[:, 0] - x0) ** 2 + (P[:, 1] - y0) ** 2
        gauss_f = np.exp(-r2f / 0.5**2)
        gauss_c = np.exp(-r2c / 0.5**2)
        uf = np.full(self.topo["Nf"], self.U_inf)
        vf = np.zeros(self.topo["Nf"])
        wf = np.zeros(self.topo["Nf"])
        if self.perturb > 0.0:
            vf += self.perturb * self.U_inf * gauss_f
            kz = 2.0 * np.pi / max(S["Lz"], 1e-12)
            wf += 0.5 * self.perturb * self.U_inf * gauss_f * np.sin(kz * Fc[:, 2])
            self.v += self.perturb * self.U_inf * gauss_c
            self.w += 0.5 * self.perturb * self.U_inf * gauss_c * np.sin(
                kz * P[:, 2]
            )
        Un = uf * n[:, 0] + vf * n[:, 1] + wf * n[:, 2]
        self.Un = self._project_Un(Un, dt=1.0)
        self.nu_eff = self.smagorinsky_viscosity()

    def load_state(self, path: Path) -> list[dict]:
        d = np.load(path)
        if len(d["u"]) != self.topo["Nc"] or len(d["Un"]) != self.topo["Nf"]:
            raise ValueError(f"{path} does not match this mesh")
        self.u = d["u"].astype(np.float64)
        self.v = d["v"].astype(np.float64)
        self.w = d["w"].astype(np.float64)
        self.p = d["p"].astype(np.float64)
        self.Un = d["Un"].astype(np.float64)
        self.t = float(d["t"])
        if "nu_eff" in d.files:
            self.nu_eff = d["nu_eff"].astype(np.float64)
        keys = ("hist_t", "hist_div", "hist_umax", "hist_Cd", "hist_Cl")
        if not all(k in d.files for k in keys):
            return []
        return [
            dict(t=float(t), max_div=float(dv), max_u=float(um), Cd=float(cd), Cl=float(cl))
            for t, dv, um, cd, cl in zip(*(d[k] for k in keys))
        ]

    def _project_Un(self, Un_star: np.ndarray, dt: float) -> np.ndarray:
        S, topo = self.S, self.topo
        own, nei = S["face_owner"], S["face_neigh"]
        d_pn = topo["d_pn"]
        interior = topo["masks"]["interior"]
        Un_star = self._enforce_Un_bc(Un_star)
        Un_star = self._balance_outlet(Un_star)
        div = self._div(Un_star)
        rhs = (self.rho / dt) * topo["vol"] * div
        p_corr, info = self._solve_poisson(rhs)
        self._p_corr = p_corr
        self._poisson_info = info
        Un_new = Un_star.copy()
        Un_new[interior] = Un_star[interior] - (dt / self.rho) * (
            (p_corr[nei[interior]] - p_corr[own[interior]]) / d_pn[interior]
        )
        return self._enforce_Un_bc(Un_new)

    def step(self, dt: float | None = None) -> dict:
        S = self.S
        n = S["face_normals"]
        self.nu_eff = self.smagorinsky_viscosity()
        if dt is None:
            dt = self.stable_dt()
        gamma = self._gamma(dt)
        self.Un = self._enforce_Un_bc(self.Un)
        nu_f = self._nu_on_faces(self.nu_eff)

        cu, cv, cw = self._convective(self.u, self.v, self.w, self.Un, gamma)
        lu = self._laplacian(self.u, nu_f, dirichlet_u=True)
        lv = self._laplacian(self.v, nu_f, dirichlet_u=False)
        lw = self._laplacian(self.w, nu_f, dirichlet_w=True)
        F = self.u + dt * (lu + cu)
        G = self.v + dt * (lv + cv)
        H = self.w + dt * (lw + cw)

        uf, vf, wf = self._face_velocity_cartesian(F, G, H, self.Un, gamma=0.0)
        Un_star = uf * n[:, 0] + vf * n[:, 1] + wf * n[:, 2]
        Un_new = self._project_Un(Un_star, dt)
        p_corr = self._p_corr

        gx, gy, gz = self._grad(p_corr)
        inv = dt / self.rho
        self.u = F - inv * gx
        self.v = G - inv * gy
        self.w = H - inv * gz
        self.p = p_corr
        self.Un = Un_new
        self.t += dt

        Cd, Cl = self.force_coefficients()
        return dict(
            t=self.t,
            dt=dt,
            gamma=gamma,
            max_div=float(np.max(np.abs(self._div(self.Un)))),
            max_u=float(np.max(np.abs(self.u))),
            max_w=float(np.max(np.abs(self.w))),
            nu_t_max=float(np.max(self.nu_eff) - self.nu),
            poisson=int(self._poisson_info),
            Cd=Cd,
            Cl=Cl,
        )

    def force_coefficients(self) -> tuple[float, float]:
        """Spanwise-integrated Cd, Cl from pressure + molecular wall shear."""
        S, topo = self.S, self.topo
        m = topo["masks"]
        if "cylinder" not in m:
            return 0.0, 0.0
        f = np.where(m["cylinder"])[0]
        n = S["face_normals"][f]
        A = S["face_areas"][f]
        o = S["face_owner"][f]
        d = 0.5 * topo["d_pn"][f]
        Fp_x = np.sum(self.p[o] * n[:, 0] * A)
        Fp_y = np.sum(self.p[o] * n[:, 1] * A)
        tx, ty = -n[:, 1], n[:, 0]
        ut = self.u[o] * tx + self.v[o] * ty
        tau = self.rho * self.nu * ut / d
        Fs_x = np.sum(tau * tx * A)
        Fs_y = np.sum(tau * ty * A)
        q = 0.5 * self.rho * self.U_inf**2 * S["D"] * S["Lz"]
        return float((Fp_x + Fs_x) / q), float((Fp_y + Fs_y) / q)

    def slice_field(self, phi: np.ndarray, k: int | None = None) -> np.ndarray:
        """Values of a cell field on spanwise layer k (default mid-span)."""
        Nc2, nz = self.S["n_cells_2d"], self.S["nz"]
        if k is None:
            k = nz // 2
        k = int(np.clip(k, 0, nz - 1))
        return phi[k * Nc2 : (k + 1) * Nc2]

    def vorticity_z(self) -> np.ndarray:
        _, uy, _ = self._grad(self.u)
        vx, _, _ = self._grad(self.v)
        return vx - uy


def save_fields(path: Path, solver: GriebelStaggeredNS3D, hist: list) -> None:
    np.savez_compressed(
        path,
        u=solver.u,
        v=solver.v,
        w=solver.w,
        p=solver.p,
        Un=solver.Un,
        nu_eff=solver.nu_eff,
        t=solver.t,
        hist_t=np.array([h["t"] for h in hist]),
        hist_div=np.array([h["max_div"] for h in hist]),
        hist_umax=np.array([h["max_u"] for h in hist]),
        hist_Cd=np.array([h["Cd"] for h in hist]),
        hist_Cl=np.array([h["Cl"] for h in hist]),
    )


def plot_forces(path: Path, hist: list, St_ref: float = 0.18) -> dict:
    t = np.array([h["t"] for h in hist])
    Cd = np.array([h["Cd"] for h in hist])
    Cl = np.array([h["Cl"] for h in hist])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(t, Cd, lw=1.2, label="Cd")
    ax.plot(t, Cl, lw=1.2, label="Cl")
    ax.set_xlabel(r"$t U_\infty / D$")
    ax.set_ylabel("force coefficient")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    n = max(len(t) // 5, 1)
    Cd_mean = float(np.mean(Cd[-n:]))
    Cl_amp = 0.5 * float(np.max(Cl[-n:]) - np.min(Cl[-n:])) if n > 2 else 0.0
    return dict(Cd_mean=Cd_mean, Cl_amp=Cl_amp, St=None, St_ref=St_ref)


def _draw_slice(ax, solver, field, vlim, cmap, vmin=None):
    S = solver.S
    lo = -vlim if vmin is None else vmin
    if "quads" in S:
        coll = PolyCollection(
            S["quads"],
            array=field,
            cmap=cmap,
            edgecolors="face",
            linewidths=0.3,
        )
        coll.set_clim(lo, vlim)
        ax.add_collection(coll)
        mappable = coll
    else:
        P = S["cell_centers"][: S["n_cells_2d"]]
        mappable = ax.scatter(
            P[:, 0], P[:, 1], c=field, s=3, cmap=cmap, vmin=lo, vmax=vlim
        )
    ax.add_patch(
        plt.Circle((S["xc"], S["yc"]), S["D"] / 2, fc="white", ec="k", lw=1.2, zorder=5)
    )
    ax.set_aspect("equal")
    ax.set_xlim(S["xc"] - 2, S["xc"] + 12)
    ax.set_ylim(S["yc"] - 4, S["yc"] + 4)
    return mappable


def save_wake_frame(path: Path, solver: GriebelStaggeredNS3D, vmax: float = 1.6) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wz = solver.slice_field(solver.vorticity_z())
    speed = solver.slice_field(np.sqrt(solver.u**2 + solver.v**2 + solver.w**2))
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True)
    m0 = _draw_slice(ax0, solver, wz, vlim=3.0, cmap="RdBu_r")
    ax0.set_title("Vorticity plot", loc="left", fontsize=11, fontweight="bold")
    ax0.set_ylabel("y")
    fig.colorbar(m0, ax=ax0, fraction=0.046, pad=0.02, label="vorticity")
    m1 = _draw_slice(ax1, solver, speed, vlim=vmax, cmap="viridis", vmin=0.0)
    ax1.set_title("Velocity plot", loc="left", fontsize=11, fontweight="bold")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(m1, ax=ax1, fraction=0.046, pad=0.02, label="velocity")
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_solution(path: Path, solver: GriebelStaggeredNS3D, hist: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    speed = solver.slice_field(np.sqrt(solver.u**2 + solver.v**2 + solver.w**2))
    m = _draw_slice(axes[0], solver, speed, vlim=1.6, cmap="viridis", vmin=0.0)
    axes[0].set_title("Velocity plot", loc="left", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(m, ax=axes[0], fraction=0.046, pad=0.04, label="velocity")
    axes[1].semilogy(
        [h["t"] for h in hist],
        [max(h["max_div"], 1e-16) for h in hist],
        "k-",
        lw=1.5,
    )
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("max |∇·U|")
    axes[1].set_title("Continuity residual", loc="left", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def frame_time(path: Path) -> float:
    """Parse t from ``wake_{k:05d}_t{t:08.4f}.png``."""
    try:
        return float(path.stem.split("_t")[-1])
    except ValueError:
        return 0.0


def save_wake_gif(frame_paths, out_path, fps=8, max_width=900) -> Path:
    from PIL import Image

    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if not paths:
        raise ValueError("no wake frames to write into GIF")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if max_width and im.width > max_width:
            h = int(im.height * max_width / im.width)
            im = im.resize((max_width, h), Image.Resampling.LANCZOS)
        rgb.append(im)
    palette_img = rgb[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [palette_img] + [
        im.quantize(colors=256, palette=palette_img) for im in rgb[1:]
    ]
    duration_ms = max(int(1000 / max(fps, 1)), 40)
    frames[0].save(
        out_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    for im in rgb:
        im.close()
    return out_path


def build_gif(frames, gif_path, fps=8, max_frames=60):
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in sorted(set(idx.tolist()))]
    return save_wake_gif(frames, gif_path, fps=fps, max_width=640), len(frames)


def parse_args():
    p = argparse.ArgumentParser(
        description="3-D Griebel NS + Smagorinsky on the extruded O–H mesh"
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--fine", action="store_true")
    p.add_argument("--build-mesh", action="store_true")
    p.add_argument("--nz", type=int, default=4)
    p.add_argument("--Lz", type=float, default=float(np.pi))
    p.add_argument("--no-periodic", action="store_true")
    p.add_argument("--scheme", choices=("central", "upwind", "donor"), default="central")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--Re", type=float, default=150.0)
    p.add_argument("--U-inf", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.4)
    p.add_argument("--poisson", choices=("lu", "cg", "sor"), default="lu")
    p.add_argument("--les", dest="les", action="store_true", default=True)
    p.add_argument("--no-les", dest="les", action="store_false")
    p.add_argument("--Cs", type=float, default=0.1)
    p.add_argument("--no-van-driest", action="store_true")
    p.add_argument("--perturb", type=float, default=0.1)
    p.add_argument("--sor-eps", type=float, default=1e-8)
    p.add_argument("--sor-max-it", type=int, default=2000)
    p.add_argument("--out-dir", type=str, default=str(HERE / "output_ns3d"))
    p.add_argument("--gif", action="store_true", default=True)
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--gif-fps", type=int, default=8)
    p.add_argument("--gif-max-frames", type=int, default=60)
    p.add_argument("--frame-every", type=int, default=2)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--restart", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_gif = bool(args.gif) and not bool(args.no_gif)
    preset = "fine" if args.fine else ("smoke" if args.quick else None)

    mesh_path = ensure_staggered_mesh_3d(
        preset=preset,
        nz=args.nz,
        Lz=args.Lz,
        periodic=not args.no_periodic,
        build=args.build_mesh,
    )
    S = load_staggered_3d(mesh_path)
    solver = GriebelStaggeredNS3D(
        S,
        Re=args.Re,
        U_inf=args.U_inf,
        scheme=args.scheme,
        tau=args.tau,
        poisson=args.poisson,
        sor_eps=args.sor_eps,
        sor_max_it=args.sor_max_it,
        perturb=args.perturb,
        les=args.les,
        Cs=args.Cs,
        van_driest=not args.no_van_driest,
    )

    t_end = args.t_end
    n_steps = args.steps
    every = max(int(args.frame_every), 1)
    if args.quick:
        if t_end is None:
            t_end = 40.0
        if args.steps == 50:
            n_steps = 5000
        if args.frame_every == 2:
            every = 8
    elif args.fine:
        if t_end is None:
            t_end = 40.0
        if args.steps == 50:
            n_steps = 200000
        if args.frame_every == 2:
            every = 40

    print("=" * 72)
    print("GRIEBEL NS 3-D on extruded known-sizes mesh")
    print(f"LES      : {'Smagorinsky Cs=' + str(solver.Cs) if solver.les else 'off (molecular ν)'}")
    print("=" * 72)
    print(f"mesh     : {mesh_path}")
    print(
        f"cells    : {solver.topo['Nc']:,}   faces: {solver.topo['Nf']:,}   "
        f"hmin={solver.topo['hmin']:.4g}   nz={S['nz']}"
    )
    print(
        f"scheme   : {solver.scheme}   Re={args.Re:g}  ν={solver.nu:.6g}  "
        f"poisson={args.poisson}"
    )
    print(f"steps    : ≤{n_steps}" + (f"  t_end={t_end:g}" if t_end else ""))
    print("-" * 72)

    hist: list[dict] = []
    frames: list[Path] = []
    if args.restart:
        hist = solver.load_state(Path(args.restart))
        frames = sorted(out.glob("wake_*.png"))
        print(f"restart  : {args.restart}  t={solver.t:.3f}")
    else:
        for old in out.glob("wake_*.png"):
            old.unlink()
    k0 = len(hist)
    n_steps += k0
    t0 = time.time()
    k = k0
    while True:
        k += 1
        info = solver.step(args.dt)
        hist.append(info)
        if write_gif and (k == k0 + 1 or k % every == 0):
            fp = out / f"wake_{k:05d}_t{solver.t:08.4f}.png"
            frames.append(save_wake_frame(fp, solver))
        if k == k0 + 1 or k % 20 == 0:
            print(
                f"step {k:5d}  t={info['t']:.4f}  dt={info['dt']:.4g}  "
                f"γ={info['gamma']:.3f}  max|div|={info['max_div']:.3e}  "
                f"max|u|={info['max_u']:.3f}  max|w|={info['max_w']:.3f}  "
                f"νt={info['nu_t_max']:.3e}  Cd={info['Cd']:.3f}  Cl={info['Cl']:+.3f}"
            )
        done = k >= n_steps or (t_end is not None and solver.t >= t_end)
        if done:
            if write_gif:
                fp = out / f"wake_{k:05d}_t{solver.t:08.4f}.png"
                frames.append(save_wake_frame(fp, solver))
            print(
                f"step {k:5d}  t={info['t']:.4f}  dt={info['dt']:.4g}  "
                f"γ={info['gamma']:.3f}  max|div|={info['max_div']:.3e}  "
                f"max|u|={info['max_u']:.3f}  Cd={info['Cd']:.3f}  "
                f"Cl={info['Cl']:+.3f}"
            )
            break

    elapsed = time.time() - t0
    print("-" * 72)
    print(
        f"done in {elapsed:.2f}s  ({elapsed / max(k - k0, 1) * 1000:.1f} ms/step)"
        f"  t={solver.t:.3f}"
    )

    mesh_tag = "quick" if args.quick else ("fine" if args.fine else "run")
    les_tag = "les" if solver.les else "lam"
    tag = f"{solver.scheme}_{mesh_tag}_{les_tag}"
    fields = out / f"fields_{tag}.npz"
    plot = out / f"plot_{tag}.png"
    forces = out / f"forces_{tag}.png"
    save_fields(fields, solver, hist)
    plot_solution(plot, solver, hist)
    fstat = plot_forces(forces, hist)
    print(f"wrote {fields}")
    print(f"wrote {plot}")
    print(f"wrote {forces}")
    print(f"forces   : Cd_mean={fstat['Cd_mean']:.3f}  Cl_amp={fstat['Cl_amp']:.3f}")

    gif_path = None
    if write_gif and frames:
        uniq, seen = [], set()
        for f in frames:
            if f not in seen:
                uniq.append(f)
                seen.add(f)
        if t_end is not None and t_end > 20.0:
            t0_gif = t_end - 20.0
            later = [f for f in uniq if frame_time(f) >= t0_gif]
            if later:
                uniq = later
        gif_path = out / f"cylinder_wake_3d_{solver.scheme}.gif"
        _, n_used = build_gif(
            uniq, gif_path, fps=int(args.gif_fps), max_frames=int(args.gif_max_frames)
        )
        print(
            f"wrote {gif_path}  ({n_used} of {len(uniq)} frames, "
            f"{gif_path.stat().st_size / 1e6:.2f} MB)"
        )
        if args.demo:
            demo = HERE / "demo" / "cylinder_wake_3d_smagorinsky.gif"
            demo.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gif_path, demo)
            print(f"wrote {demo}")
    return solver, hist, gif_path


if __name__ == "__main__":
    main()
