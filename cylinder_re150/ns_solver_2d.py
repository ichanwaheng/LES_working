#!/usr/bin/env python3
"""
ns_solver_2d.py
===============
2-D incompressible Navier–Stokes solver only (Griebel / MAC projection)
on the staggered O–H mesh from ``mesh_from_known_sizes.py``.

This file loads a mesh and advances the solution. It does not plot or do
any other post-processing. ``ns_staggered_euler.py`` is unchanged and still
has the figures.

Per step::

    1. apply BCs
    2. F, G = explicit Euler predictors (no pressure)
    3. pressure Poisson  ∇²p = (ρ/Δt) ∇·U*
    4. correct staggered face flux / cell velocity

    python ns_solver_2d.py --quick --scheme donor --steps 80
    python ns_solver_2d.py --fine --scheme central --t-end 120
    python ns_solver_2d.py --mesh mesh_output/staggered_fine.npz --restart fields.npz

Needs numpy and scipy. Imports the mesher only if a staggered npz is missing
or ``--build-mesh`` is set.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, splu

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load_staggered(path: Path) -> dict:
    raw = np.load(path, allow_pickle=False)
    out = dict(
        cell_centers=raw["cell_centers"].astype(np.float64),
        face_centers=raw["face_centers"].astype(np.float64),
        face_normals=raw["face_normals"].astype(np.float64),
        face_lengths=raw["face_lengths"].astype(np.float64),
        face_owner=raw["face_owner"].astype(np.int64),
        face_neigh=raw["face_neigh"].astype(np.int64),
        face_bc=raw["face_bc"].astype(np.int32),
        bc_names=[str(x) for x in raw["bc_names"].tolist()],
        D=float(raw["D"]),
        xc=float(raw["xc"]),
        yc=float(raw["yc"]),
        d_min=float(raw["d_min"]),
        d_max=float(raw["d_max"]),
        a=float(raw["a"]),
    )
    if "quads" in raw.files:
        out["quads"] = raw["quads"].astype(np.float64)
    return out


MESH_PRESETS = {
    # smoke: large d_min, few cells, fast
    "smoke": dict(d_min=0.05, d_max=0.30, growth_ratio=1.20, max_jump=2.5, a=2.0),
    # fine: small d_min, many cells — resolves shear layers for shedding
    "fine": dict(d_min=0.015, d_max=0.12, growth_ratio=1.15, max_jump=2.0, a=None),
}


def ensure_staggered_mesh(
    path: Path, build: bool, quick: bool, preset: str | None = None
) -> Path:
    """
    Return path to staggered_*.npz from the known-sizes mesher.

    ``--quick`` → preset 'smoke'; ``--fine`` → preset 'fine'. Both are the
    same O–H multiblock mesher — never a Cartesian substitute.
    """
    path = Path(path)
    if preset is None and quick:
        preset = "smoke"
    if path.is_file() and not build and preset is None:
        return path

    from mesh_from_known_sizes import INPUT, main as mesh_main

    I = dict(INPUT)
    tag = "result"
    if preset is not None:
        I.update(MESH_PRESETS[preset])
        tag = preset
        path = HERE / "mesh_output" / f"staggered_{tag}.npz"
        if path.is_file() and not build:
            print(f"[mesh] reusing {path}")
            return path
    elif build or not path.is_file():
        tag = "result"
        path = HERE / "mesh_output" / f"staggered_{tag}.npz"

    print(f"[mesh] building known-sizes staggered mesh → {path}")
    print(
        f"[mesh] d_min={I['d_min']}  d_max={I['d_max']}  "
        f"growth={I['growth_ratio']}"
    )
    mesh_main(I, tag=tag)
    return HERE / "mesh_output" / f"staggered_{tag}.npz"


def build_topology(S: dict) -> dict:
    P, Fc = S["cell_centers"], S["face_centers"]
    n, L = S["face_normals"], S["face_lengths"]
    own, nei = S["face_owner"], S["face_neigh"]
    Nc, Nf = len(P), len(Fc)

    contrib = 0.5 * (Fc[:, 0] * n[:, 0] + Fc[:, 1] * n[:, 1]) * L
    vol = np.zeros(Nc)
    np.add.at(vol, own, contrib)
    interior = nei >= 0
    np.add.at(vol, nei[interior], -contrib[interior])
    vol = np.maximum(vol, 1e-30)

    d_pn = np.empty(Nf)
    d_pn[interior] = np.linalg.norm(P[nei[interior]] - P[own[interior]], axis=1)
    bnd = ~interior
    d_pn[bnd] = 2.0 * np.linalg.norm(Fc[bnd] - P[own[bnd]], axis=1)
    d_pn = np.maximum(d_pn, 1e-30)

    bc_index = {nm: i for i, nm in enumerate(S["bc_names"])}
    face_bc = S["face_bc"]
    masks = {nm: face_bc == i for nm, i in bc_index.items()}
    masks["interior"] = interior
    masks["boundary"] = ~interior

    hmin = float(np.min(np.sqrt(vol)))
    return dict(
        vol=vol,
        d_pn=d_pn,
        masks=masks,
        bc_index=bc_index,
        Nc=Nc,
        Nf=Nf,
        hmin=hmin,
    )


def accumulate(flux, own, nei, Nc):
    out = np.zeros(Nc)
    np.add.at(out, own, flux)
    m = nei >= 0
    np.add.at(out, nei[m], -flux[m])
    return out


# =====================================================================
# Griebel donor-cell γ face interpolation  (unstructured form of
# CaptainProton42/NavierStokes integration.c)
# =====================================================================
def donor_cell_face(
    phi: np.ndarray,
    Un: np.ndarray,
    own: np.ndarray,
    nei: np.ndarray,
    interior: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    φ_f = ½(φ_o+φ_n) − ½ γ sign(U_n) (φ_n−φ_o)

    γ=0 central, γ=1 full upwind (donor-cell).
    """
    phi_f = phi[own].copy()
    if not interior.any():
        return phi_f
    o = own[interior]
    n = nei[interior]
    central = 0.5 * (phi[o] + phi[n])
    upw = np.where(Un[interior] >= 0.0, phi[o], phi[n])
    # equivalent blend: (1-γ)*central + γ*upwind
    # = ½(o+n) − ½γ sign(Un)(n−o)
    phi_f[interior] = (1.0 - gamma) * central + gamma * upw
    return phi_f


# =====================================================================
# Griebel-style F, G + projection on staggered FV mesh
# =====================================================================
class GriebelStaggeredNS:
    """
    CaptainProton42 / Griebel projection on the known-sizes staggered mesh.

    Unknowns: cell-centred (u,v,p); continuity on face-normal Un.
    """

    def __init__(
        self,
        S: dict,
        Re: float = 150.0,
        U_inf: float = 1.0,
        rho: float = 1.0,
        scheme: str = "donor",
        tau: float = 0.4,
        sor_omega: float = 1.7,
        poisson: str = "lu",
        sor_max_it: int = 2000,
        sor_eps: float = 1e-8,
        perturb: float = 0.1,
        perturb_T: float = 2.0,
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
        self.sor_omega = float(sor_omega)
        self.poisson = str(poisson).lower()
        self.sor_max_it = int(sor_max_it)
        self.sor_eps = float(sor_eps)
        self.perturb = float(perturb)
        self.perturb_T = float(perturb_T)
        self.t = 0.0

        Nc, Nf = self.topo["Nc"], self.topo["Nf"]
        self.u = np.full(Nc, self.U_inf)
        self.v = np.zeros(Nc)
        self.p = np.zeros(Nc)
        self.Un = np.zeros(Nf)
        if self.poisson not in ("lu", "cg", "sor"):
            raise ValueError("poisson must be lu | cg | sor")
        self._A = self._build_poisson()
        self._M = self._build_preconditioner(self._A)
        self._lu = self._build_direct(self._A) if self.poisson == "lu" else None
        self._init_solenoidal()

    @staticmethod
    def _build_direct(A: sparse.csr_matrix):
        """
        Sparse LU of the Neumann Laplacian with one dof pinned.

        The pin removes the constant null space; because the un-pinned rows
        sum to zero, a mean-free RHS is still solved exactly by every row.
        Factorised once, then each step costs a pair of triangular solves.
        """
        Ap = A.tolil(copy=True)
        Ap[0, :] = 0.0
        Ap[0, 0] = 1.0
        return splu(Ap.tocsc(), permc_spec="MMD_AT_PLUS_A")

    @staticmethod
    def _build_preconditioner(A: sparse.csr_matrix):
        """Jacobi preconditioner (safe for the singular Neumann Laplacian)."""
        diag = A.diagonal()
        inv = np.where(np.abs(diag) > 1e-30, 1.0 / diag, 0.0)

        def jacobi(x, inv=inv):
            return inv * x

        return LinearOperator(A.shape, matvec=jacobi)

    # ----- BCs -----
    def _enforce_Un_bc(self, Un: np.ndarray) -> np.ndarray:
        S, m = self.S, self.topo["masks"]
        n = S["face_normals"]
        Un = Un.copy()
        if "inlet" in m:
            v_in = self._inlet_v()
            Un[m["inlet"]] = (
                self.U_inf * n[m["inlet"], 0] + v_in * n[m["inlet"], 1]
            )
        for w in ("wall_top", "wall_bottom", "cylinder"):
            if w in m:
                Un[m[w]] = 0.0
        return Un

    def _inlet_v(self) -> float:
        """
        Short transverse inlet pulse v = ε U∞ sin(π t/T) for t < T.

        The O–H mesh is exactly mirror-symmetric, so without a finite
        perturbation Cl stays identically zero and the wake never sheds.
        """
        eps, T = self.perturb, self.perturb_T
        if eps <= 0.0 or self.t >= T:
            return 0.0
        return float(eps * self.U_inf * np.sin(np.pi * self.t / T))

    def _balance_outlet(self, Un: np.ndarray) -> np.ndarray:
        """
        Shift the zero-gradient outlet flux so total outflow = total inflow.

        The Neumann Poisson problem is only solvable for a mass-balanced RHS;
        without this the mean imbalance is left behind as a uniform residual
        divergence that is largest in the smallest (cylinder) cells.
        """
        m = self.topo["masks"]
        if "outlet" not in m:
            return Un
        L = self.S["face_lengths"]
        out = m["outlet"]
        bnd = self.S["face_neigh"] < 0
        others = bnd & ~out
        Q_other = float(np.sum(Un[others] * L[others]))  # net outward, non-outlet
        Q_out = float(np.sum(Un[out] * L[out]))
        Un = Un.copy()
        Un[out] += -(Q_other + Q_out) / float(np.sum(L[out]))
        return Un

    def _gamma(self, dt: float) -> float:
        if self.scheme == "central":
            return 0.0
        if self.scheme == "upwind":
            return 1.0
        umax = max(float(np.max(np.abs(self.Un))), abs(self.U_inf), 1e-12)
        return float(min(1.0, umax * dt / self.topo["hmin"]))

    def stable_dt(self) -> float:
        """Adaptive Δt with security τ (CaptainProton42 main.c idea)."""
        h = self.topo["hmin"]
        umax = max(float(np.max(np.abs(self.Un))), abs(self.U_inf), 1e-12)
        dt_conv = h / umax
        dt_diff = self.Re / 2.0 * h * h  # ~ Re/2 / (1/h²) with h²
        # match reference: Re/2/(1/dx²+1/dy²) ≈ Re/2/(2/h²) = Re h²/4
        dt_diff = self.Re * h * h / 4.0
        return float(self.tau * min(dt_conv, dt_diff))

    # ----- operators -----
    def _face_velocity_cartesian(self, u, v, Un, gamma):
        S, topo = self.S, self.topo
        own, nei = S["face_owner"], S["face_neigh"]
        interior = topo["masks"]["interior"]
        uf = donor_cell_face(u, Un, own, nei, interior, gamma)
        vf = donor_cell_face(v, Un, own, nei, interior, gamma)
        m = topo["masks"]
        if "inlet" in m:
            uf[m["inlet"]], vf[m["inlet"]] = self.U_inf, 0.0
        for w in ("wall_top", "wall_bottom", "cylinder"):
            if w in m:
                uf[m[w]] = 0.0
                vf[m[w]] = 0.0
        return uf, vf

    def _convective(self, u, v, Un, gamma):
        """−∇·(u⊗u) at cells using donor-cell face momentum (Griebel γ)."""
        S, topo = self.S, self.topo
        L = S["face_lengths"]
        Un = self._enforce_Un_bc(Un)
        uf, vf = self._face_velocity_cartesian(u, v, Un, gamma)
        ru = accumulate(Un * uf * L, S["face_owner"], S["face_neigh"], topo["Nc"])
        rv = accumulate(Un * vf * L, S["face_owner"], S["face_neigh"], topo["Nc"])
        return -ru / topo["vol"], -rv / topo["vol"]

    def _laplacian(self, phi, U_inf_dirichlet_u: bool = False):
        """ν∇²φ via face gradients (central viscous, as in Griebel d²φ)."""
        S, topo = self.S, self.topo
        L, d_pn = S["face_lengths"], topo["d_pn"]
        own, nei = S["face_owner"], S["face_neigh"]
        interior = topo["masks"]["interior"]
        m = topo["masks"]
        Nf = topo["Nf"]

        phi_f = phi[own].copy()
        if U_inf_dirichlet_u:
            if "inlet" in m:
                phi_f[m["inlet"]] = self.U_inf
            for w in ("wall_top", "wall_bottom", "cylinder"):
                if w in m:
                    phi_f[m[w]] = 0.0
        else:
            for w in ("wall_top", "wall_bottom", "cylinder", "inlet"):
                if w in m:
                    phi_f[m[w]] = 0.0  # v=0 on those faces

        g = np.empty(Nf)
        g[interior] = (phi[nei[interior]] - phi[own[interior]]) / d_pn[interior]
        bnd = ~interior
        g[bnd] = (phi_f[bnd] - phi[own[bnd]]) / (0.5 * d_pn[bnd])
        return accumulate(self.nu * g * L, own, nei, topo["Nc"]) / topo["vol"]

    def _div(self, Un):
        S, topo = self.S, self.topo
        return (
            accumulate(Un * S["face_lengths"], S["face_owner"], S["face_neigh"], topo["Nc"])
            / topo["vol"]
        )

    def _grad(self, phi):
        S, topo = self.S, self.topo
        L, n = S["face_lengths"], S["face_normals"]
        own, nei = S["face_owner"], S["face_neigh"]
        interior = topo["masks"]["interior"]
        pf = phi[own].copy()
        pf[interior] = 0.5 * (phi[own[interior]] + phi[nei[interior]])
        gx = accumulate(pf * n[:, 0] * L, own, nei, topo["Nc"]) / topo["vol"]
        gy = accumulate(pf * n[:, 1] * L, own, nei, topo["Nc"]) / topo["vol"]
        return gx, gy

    def _build_poisson(self):
        """
        Symmetric integrated Neumann Laplacian (singular).

        Null space handled in ``_solve_poisson`` via mean projection.
        """
        S, topo = self.S, self.topo
        L, d_pn = S["face_lengths"], topo["d_pn"]
        own, nei = S["face_owner"], S["face_neigh"]
        interior = topo["masks"]["interior"]
        Nc = topo["Nc"]
        o, c = own[interior], nei[interior]
        g = L[interior] / d_pn[interior]
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
        if info != 0:
            p2, info2 = cg(
                Aop,
                rhs,
                x0=p,
                M=self._M,
                rtol=max(self.sor_eps, 1e-6),
                maxiter=self.sor_max_it * 2,
            )
            p = np.nan_to_num(np.asarray(p2), nan=0.0)
            p -= float(np.mean(p))
            info = info2
        return p, 0 if info == 0 else int(info)

    def _sor(self, rhs: np.ndarray) -> tuple[np.ndarray, int]:
        """Classical SOR on the sparse Laplacian (CaptainProton42 SOR)."""
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
        uf = np.full(self.topo["Nf"], self.U_inf)
        vf = np.zeros(self.topo["Nf"])
        if self.perturb > 0.0:
            # asymmetric transverse seed in the near wake: the symmetric steady
            # branch (long Fornberg bubble) is otherwise reached and the
            # shedding mode takes >100 tU/D to grow from round-off
            Fc = S["face_centers"]
            r2 = (Fc[:, 0] - 1.0) ** 2 + Fc[:, 1] ** 2
            vf += self.perturb * self.U_inf * np.exp(-r2 / 0.5**2)
            Pc = S["cell_centers"]
            r2c = (Pc[:, 0] - 1.0) ** 2 + Pc[:, 1] ** 2
            self.v += self.perturb * self.U_inf * np.exp(-r2c / 0.5**2)
        Un = uf * n[:, 0] + vf * n[:, 1]
        self.Un = self._project_Un(Un, dt=1.0)

    def load_state(self, path: Path) -> list[dict]:
        """Restart from a ``fields_*.npz``; returns the stored history."""
        d = np.load(path)
        if len(d["u"]) != self.topo["Nc"] or len(d["Un"]) != self.topo["Nf"]:
            raise ValueError(f"{path} does not match this mesh")
        self.u = d["u"].astype(np.float64)
        self.v = d["v"].astype(np.float64)
        self.p = d["p"].astype(np.float64)
        self.Un = d["Un"].astype(np.float64)
        self.t = float(d["t"])
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
        # A p = (ρ/dt) V div   (integrated, like Griebel RHS·volume)
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
        """One Griebel step: BC → F,G → Poisson → correct."""
        S = self.S
        n = S["face_normals"]
        if dt is None:
            dt = self.stable_dt()
        gamma = self._gamma(dt)
        self.Un = self._enforce_Un_bc(self.Un)

        # --- F, G predictors (CaptainProton42 FG) ---
        cu, cv = self._convective(self.u, self.v, self.Un, gamma)
        lu = self._laplacian(self.u, U_inf_dirichlet_u=True)
        lv = self._laplacian(self.v, U_inf_dirichlet_u=False)
        F = self.u + dt * (lu + cu)  # cu already = −conv
        G = self.v + dt * (lv + cv)

        # face normal from F,G (central) then project
        uf, vf = self._face_velocity_cartesian(F, G, self.Un, gamma=0.0)
        Un_star = uf * n[:, 0] + vf * n[:, 1]
        Un_new = self._project_Un(Un_star, dt)
        p_corr = self._p_corr

        gx, gy = self._grad(p_corr)
        self.u = F - (dt / self.rho) * gx
        self.v = G - (dt / self.rho) * gy
        # F,G carry no old pressure gradient, so the Poisson solution *is*
        # the pressure at the new level (Griebel eq. 3.38), not an increment
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
            max_v=float(np.max(np.abs(self.v))),
            poisson=int(self._poisson_info),
            Cd=Cd,
            Cl=Cl,
        )

    def force_coefficients(self) -> tuple[float, float]:
        """
        Cd, Cl from pressure + wall shear on the cylinder faces.

        Cl oscillating in time is the signature of vortex shedding.
        """
        S, topo = self.S, self.topo
        m = topo["masks"]
        if "cylinder" not in m:
            return 0.0, 0.0
        f = np.where(m["cylinder"])[0]
        n = S["face_normals"][f]  # points out of fluid (into cylinder)
        L = S["face_lengths"][f]
        o = S["face_owner"][f]
        d = 0.5 * topo["d_pn"][f]

        # pressure acts on body along −(fluid-outward) = +n_body... here n is
        # owner→face i.e. fluid→solid, so pressure force on body = +p n L
        Fp_x = np.sum(self.p[o] * n[:, 0] * L)
        Fp_y = np.sum(self.p[o] * n[:, 1] * L)

        # wall shear: τ = μ ∂u_t/∂n ; velocity at wall = 0
        tx, ty = -n[:, 1], n[:, 0]
        ut = self.u[o] * tx + self.v[o] * ty
        tau = self.rho * self.nu * ut / d
        # shear force on body opposes fluid motion along tangent
        Fs_x = np.sum(tau * tx * L)
        Fs_y = np.sum(tau * ty * L)

        q = 0.5 * self.rho * self.U_inf**2 * S["D"]
        return float((Fp_x + Fs_x) / q), float((Fp_y + Fs_y) / q)

    def speed(self) -> np.ndarray:
        return np.hypot(self.u, self.v)

    def vorticity(self) -> np.ndarray:
        """ω_z = ∂v/∂x − ∂u/∂y from the Green–Gauss cell gradients."""
        _, duy = self._grad(self.u)
        dvx, _ = self._grad(self.v)
        return dvx - duy


def save_solution(path: Path, solver: GriebelStaggeredNS, hist: list) -> None:
    """Write the unknowns and time history (no figures)."""
    np.savez_compressed(
        path,
        t=solver.t,
        u=solver.u,
        v=solver.v,
        p=solver.p,
        Un=solver.Un,
        cell_centers=solver.S["cell_centers"],
        scheme=np.array(solver.scheme),
        hist_t=np.array([h["t"] for h in hist]),
        hist_div=np.array([h["max_div"] for h in hist]),
        hist_umax=np.array([h["max_u"] for h in hist]),
        hist_Cd=np.array([h.get("Cd", 0.0) for h in hist]),
        hist_Cl=np.array([h.get("Cl", 0.0) for h in hist]),
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="2-D Griebel NS solver (mesh + time stepping, no plots)"
    )
    p.add_argument(
        "--mesh",
        type=str,
        default=str(HERE / "mesh_output" / "staggered_result.npz"),
        help="staggered npz from mesh_from_known_sizes.py",
    )
    p.add_argument("--build-mesh", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--fine", action="store_true")
    p.add_argument("--scheme", choices=("central", "upwind", "donor"), default="donor")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--Re", type=float, default=150.0)
    p.add_argument("--U-inf", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.4)
    p.add_argument("--poisson", choices=("lu", "cg", "sor"), default="lu")
    p.add_argument("--perturb", type=float, default=0.1)
    p.add_argument("--sor-eps", type=float, default=1e-8)
    p.add_argument("--sor-max-it", type=int, default=2000)
    p.add_argument("--out-dir", type=str, default=str(HERE / "output_ns"))
    p.add_argument("--restart", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preset = "fine" if args.fine else None
    mesh_path = ensure_staggered_mesh(
        Path(args.mesh), build=args.build_mesh, quick=args.quick, preset=preset
    )
    S = load_staggered(mesh_path)
    solver = GriebelStaggeredNS(
        S,
        Re=args.Re,
        U_inf=args.U_inf,
        scheme=args.scheme,
        tau=args.tau,
        poisson=args.poisson,
        sor_eps=args.sor_eps,
        sor_max_it=args.sor_max_it,
        perturb=args.perturb,
    )

    t_end = args.t_end
    n_steps = args.steps
    if args.quick:
        if t_end is None:
            t_end = 40.0
        if args.steps == 50:
            n_steps = 10000
    elif args.fine:
        if t_end is None:
            t_end = 120.0
        if args.steps == 50:
            n_steps = 200000

    print("=" * 72)
    print("2-D Griebel NS  (solver only, no post-processing)")
    print("=" * 72)
    print(f"mesh     : {mesh_path}")
    print(
        f"cells    : {solver.topo['Nc']:,}   faces: {solver.topo['Nf']:,}   "
        f"hmin={solver.topo['hmin']:.4g}"
    )
    print(
        f"scheme   : {solver.scheme}   Re={args.Re:g}  ν={solver.nu:.6g}  "
        f"poisson={args.poisson}"
    )
    print(f"steps    : ≤{n_steps}" + (f"  t_end={t_end:g}" if t_end else ""))
    print("-" * 72)

    hist: list[dict] = []
    if args.restart:
        hist = solver.load_state(Path(args.restart))
        print(f"restart  : {args.restart}  t={solver.t:.3f}  ({len(hist)} steps)")
    k0 = len(hist)
    n_steps += k0
    t0 = time.time()
    k = k0
    while True:
        k += 1
        info = solver.step(args.dt)
        hist.append(info)
        if k == k0 + 1 or k % 20 == 0:
            print(
                f"step {k:5d}  t={info['t']:.4f}  dt={info['dt']:.4g}  "
                f"γ={info['gamma']:.3f}  max|div|={info['max_div']:.3e}  "
                f"max|u|={info['max_u']:.3f}  Cd={info['Cd']:.3f}  "
                f"Cl={info['Cl']:+.3f}"
            )
        done = k >= n_steps or (t_end is not None and solver.t >= t_end)
        if done:
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
    tag = f"{solver.scheme}_{mesh_tag}"
    fields = out / f"fields_{tag}.npz"
    save_solution(fields, solver, hist)
    print(f"wrote {fields}")
    return solver, hist


if __name__ == "__main__":
    main()
