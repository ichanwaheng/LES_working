#!/usr/bin/env python3
"""
ns_staggered_euler.py
=====================
2-D incompressible Navier–Stokes on the **structured multiblock staggered
mesh** from ``mesh_from_known_sizes.py``.

Meshing
-------
Cell / face locations come from ``mesh_output/staggered_*.npz``
(P at cell centres, velocities at face centres).  That is the same O–H
channel-cylinder mesh already built for this case — not a coarse Cartesian
box.

Solver method (time / convection / projection)
----------------------------------------------
Algorithm structure follows Griebel, Dornseifer & Neunhoeffer (1998) as in:

    https://github.com/CaptainProton42/NavierStokes

Per step (same as their ``main.c``)::

    1. apply BCs
    2. F, G  = explicit Euler predictors  (no pressure)
         F = u + Δt · ( ν ∇²u − conv_γ(u) )
         G = v + Δt · ( ν ∇²v − conv_γ(v) )
    3. pressure Poisson  ∇²p = (1/Δt) ∇·U*
    4. correct staggered face flux / cell velocity

Convection is the Griebel **donor-cell / γ** face blend
(``integration.c``)::

    φ_f = ½(φ_o+φ_n) − ½ γ sign(U_n) (φ_n−φ_o)

    γ = 0        → central      ``--scheme central``
    γ = 1        → upwind       ``--scheme upwind``
    γ = max(CFL) → adaptive     ``--scheme donor``  (reference default)

On the multiblock mesh the operators are finite-volume (Green–Gauss) at the
extracted staggered faces; the *method* is the Griebel projection above.

Usage
-----
    cd cylinder_re150
    # uses mesh_output/staggered_result.npz from mesh_from_known_sizes.py
    python ns_staggered_euler.py --scheme donor --steps 50 --gif

    # rebuild mesh with the known-sizes INPUT, then solve
    python ns_staggered_euler.py --build-mesh --scheme donor --gif

    # slightly relaxed known-sizes mesh (still O–H multiblock, not Cartesian)
    python ns_staggered_euler.py --quick --scheme donor --steps 80 --gif --demo

Needs numpy, scipy, matplotlib.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


# =====================================================================
# load / build staggered mesh from mesh_from_known_sizes
# =====================================================================
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


def ensure_staggered_mesh(path: Path, build: bool, quick: bool) -> Path:
    """
    Return path to staggered_*.npz from the known-sizes mesher.

    ``--quick`` still uses mesh_from_known_sizes (O–H multiblock) with a
    *larger* d_min for smoke tests — never a Cartesian substitute.
    """
    path = Path(path)
    if path.is_file() and not build and not quick:
        return path

    from mesh_from_known_sizes import INPUT, main as mesh_main

    I = dict(INPUT)
    tag = "result"
    if quick:
        # larger minimum cell → fewer cells, larger Δt, visible wake in smoke test
        # keep enough room in the channel for BL + O-block (domain |y|≤5)
        I.update(
            d_min=0.05,
            d_max=0.30,
            growth_ratio=1.20,
            max_jump=2.5,
            a=2.0,
        )
        tag = "smoke"
        path = HERE / "mesh_output" / f"staggered_{tag}.npz"
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
        poisson: str = "cg",
        sor_max_it: int = 2000,
        sor_eps: float = 1e-8,
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
        self.t = 0.0

        Nc, Nf = self.topo["Nc"], self.topo["Nf"]
        self.u = np.full(Nc, self.U_inf)
        self.v = np.zeros(Nc)
        self.p = np.zeros(Nc)
        self.Un = np.zeros(Nf)
        self._A = self._build_poisson()
        self._M = self._build_preconditioner(self._A)
        self._init_solenoidal()

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
            Un[m["inlet"]] = (
                self.U_inf * n[m["inlet"], 0] + 0.0 * n[m["inlet"], 1]
            )
        for w in ("wall_top", "wall_bottom", "cylinder"):
            if w in m:
                Un[m[w]] = 0.0
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
        Un = uf * n[:, 0] + vf * n[:, 1]
        self.Un = self._project_Un(Un, dt=1.0)

    def _project_Un(self, Un_star: np.ndarray, dt: float) -> np.ndarray:
        S, topo = self.S, self.topo
        own, nei = S["face_owner"], S["face_neigh"]
        d_pn = topo["d_pn"]
        interior = topo["masks"]["interior"]
        Un_star = self._enforce_Un_bc(Un_star)
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
        self.p = self.p + p_corr
        self.Un = Un_new
        self.t += dt

        return dict(
            t=self.t,
            dt=dt,
            gamma=gamma,
            max_div=float(np.max(np.abs(self._div(self.Un)))),
            max_u=float(np.max(np.abs(self.u))),
            max_v=float(np.max(np.abs(self.v))),
            poisson=int(self._poisson_info),
        )

    def speed(self) -> np.ndarray:
        return np.hypot(self.u, self.v)


# =====================================================================
# plots / GIF
# =====================================================================
def save_fields(path: Path, solver: GriebelStaggeredNS, hist: list) -> None:
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
    )


def _wake_field(solver: GriebelStaggeredNS) -> tuple[np.ndarray, str]:
    """
    Streamwise perturbation u − U∞ (shows recirculation / wake clearly).

    Raw |U|≈U∞ everywhere looks static; the deficit makes flow visible.
    """
    return solver.u - solver.U_inf, r"$u - U_\infty$"


def _draw_field(
    ax,
    solver: GriebelStaggeredNS,
    field: np.ndarray,
    vlim: float | None = None,
    cmap: str = "RdBu_r",
):
    S = solver.S
    if vlim is None:
        vlim = float(np.percentile(np.abs(field), 98))
        vlim = max(vlim, 0.15)
    if "quads" in S:
        coll = PolyCollection(
            S["quads"],
            array=field,
            cmap=cmap,
            edgecolors="none",
            linewidths=0,
        )
        coll.set_clim(-vlim, vlim)
        ax.add_collection(coll)
        mappable = coll
    else:
        P = S["cell_centers"]
        mappable = ax.scatter(
            P[:, 0], P[:, 1], c=field, s=3, cmap=cmap, vmin=-vlim, vmax=vlim
        )
    # sparse velocity arrows so motion direction is obvious
    P = S["cell_centers"]
    step = max(len(P) // 400, 1)
    idx = np.arange(0, len(P), step)
    ax.quiver(
        P[idx, 0],
        P[idx, 1],
        solver.u[idx],
        solver.v[idx],
        color="k",
        alpha=0.35,
        scale=25,
        width=0.0025,
        zorder=4,
    )
    ax.add_patch(
        plt.Circle((S["xc"], S["yc"]), S["D"] / 2, fc="white", ec="k", lw=1.2, zorder=5)
    )
    ax.set_aspect("equal")
    ax.set_xlim(S["xc"] - 2, S["xc"] + 12)
    ax.set_ylim(S["yc"] - 4, S["yc"] + 4)
    return mappable


def save_wake_frame(path: Path, solver: GriebelStaggeredNS, vlim: float = 0.8) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    field, label = _wake_field(solver)
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    m = _draw_field(ax, solver, field, vlim=vlim)
    ax.set_title(
        f"Re={solver.Re:g}  scheme={solver.scheme}  tU/D={solver.t:.2f}  "
        f"cells={solver.topo['Nc']:,}  d_min={solver.S['d_min']:g}",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(m, ax=ax, fraction=0.046, pad=0.02, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_solution(path: Path, solver: GriebelStaggeredNS, hist: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    field, label = _wake_field(solver)
    m = _draw_field(axes[0], solver, field)
    axes[0].set_title(
        f"wake  scheme={solver.scheme}  t={solver.t:.3f}  cells={solver.topo['Nc']:,}",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    fig.colorbar(m, ax=axes[0], fraction=0.046, pad=0.04, label=label)
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


def build_gif(frames: list[Path], gif_path: Path, fps: int = 8) -> Path:
    from viz import save_wake_gif

    return save_wake_gif(frames, gif_path, fps=fps, max_width=900)


# =====================================================================
# main
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Griebel MAC projection on mesh_from_known_sizes staggered mesh "
            "(method: github.com/CaptainProton42/NavierStokes)"
        )
    )
    p.add_argument(
        "--mesh",
        type=str,
        default=str(HERE / "mesh_output" / "staggered_result.npz"),
        help="staggered npz from mesh_from_known_sizes.py",
    )
    p.add_argument(
        "--build-mesh",
        action="store_true",
        help="rebuild staggered mesh with known-sizes INPUT before solving",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="smoke mesh with larger d_min (O–H multiblock, visible wake)",
    )
    p.add_argument(
        "--scheme",
        choices=("central", "upwind", "donor"),
        default="donor",
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="stop at this physical time (overrides steps when reached first)",
    )
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--Re", type=float, default=150.0)
    p.add_argument("--U-inf", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.4)
    p.add_argument(
        "--poisson",
        choices=("cg", "sor"),
        default="cg",
        help="pressure solver (SOR = classic Griebel; CG for large meshes)",
    )
    p.add_argument("--sor-eps", type=float, default=1e-8)
    p.add_argument("--sor-max-it", type=int, default=2000)
    p.add_argument("--out-dir", type=str, default=str(HERE / "output_ns"))
    p.add_argument("--gif", action="store_true", default=True)
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--gif-fps", type=int, default=8)
    p.add_argument("--frame-every", type=int, default=2)
    p.add_argument("--demo", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_gif = bool(args.gif) and not bool(args.no_gif)

    mesh_path = ensure_staggered_mesh(
        Path(args.mesh), build=args.build_mesh, quick=args.quick
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
    )

    # smoke defaults: large d_min mesh + enough physical time for a wake
    t_end = args.t_end
    n_steps = args.steps
    every = max(int(args.frame_every), 1)
    if args.quick:
        if t_end is None:
            t_end = 40.0
        if args.steps == 50:
            n_steps = 10000  # safety cap; t_end usually stops first
        if args.frame_every == 2:
            every = 8

    print("=" * 72)
    print("GRIEBEL NS on known-sizes staggered mesh")
    print("method : https://github.com/CaptainProton42/NavierStokes")
    print("=" * 72)
    print(f"mesh     : {mesh_path}")
    print(
        f"cells    : {solver.topo['Nc']:,}   faces: {solver.topo['Nf']:,}   "
        f"hmin={solver.topo['hmin']:.4g}"
    )
    print(
        f"sizing   : d_min={S['d_min']:g}  d_max={S['d_max']:g}  a={S['a']:.4g}"
    )
    print(
        f"scheme   : {solver.scheme}   Re={args.Re:g}  ν={solver.nu:.6g}  "
        f"poisson={args.poisson}"
    )
    print(f"steps    : ≤{n_steps}" + (f"  t_end={t_end:g}" if t_end else ""))
    if write_gif:
        print(f"GIF      : every {every} steps @ {args.gif_fps} fps")
    print("-" * 72)

    for old in out.glob("wake_*.png"):
        old.unlink()

    hist: list[dict] = []
    frames: list[Path] = []
    t0 = time.time()
    k = 0
    while True:
        k += 1
        info = solver.step(args.dt)
        hist.append(info)
        if write_gif and (k == 1 or k % every == 0):
            fp = out / f"wake_{k:05d}_t{solver.t:08.4f}.png"
            frames.append(save_wake_frame(fp, solver))
        if k == 1 or k % max(1, 20) == 0:
            print(
                f"step {k:5d}  t={info['t']:.4f}  dt={info['dt']:.4g}  "
                f"γ={info['gamma']:.3f}  max|div|={info['max_div']:.3e}  "
                f"max|u|={info['max_u']:.3f}"
            )
        done = k >= n_steps or (t_end is not None and solver.t >= t_end)
        if done:
            if write_gif:
                fp = out / f"wake_{k:05d}_t{solver.t:08.4f}.png"
                frames.append(save_wake_frame(fp, solver))
            print(
                f"step {k:5d}  t={info['t']:.4f}  dt={info['dt']:.4g}  "
                f"γ={info['gamma']:.3f}  max|div|={info['max_div']:.3e}  "
                f"max|u|={info['max_u']:.3f}"
            )
            break

    elapsed = time.time() - t0
    print("-" * 72)
    print(f"done in {elapsed:.2f}s  ({elapsed / k * 1000:.1f} ms/step)  t={solver.t:.3f}")

    tag = f"{solver.scheme}_{'quick' if args.quick else 'run'}"
    fields = out / f"fields_{tag}.npz"
    plot = out / f"plot_{tag}.png"
    save_fields(fields, solver, hist)
    plot_solution(plot, solver, hist)
    print(f"wrote {fields}")
    print(f"wrote {plot}")

    gif_path = None
    if write_gif and frames:
        # unique by path
        uniq, seen = [], set()
        for f in frames:
            if f not in seen:
                uniq.append(f)
                seen.add(f)
        gif_path = out / f"cylinder_wake_{solver.scheme}.gif"
        build_gif(uniq, gif_path, fps=int(args.gif_fps))
        print(
            f"wrote {gif_path}  ({len(uniq)} frames, "
            f"{gif_path.stat().st_size / 1e6:.2f} MB)"
        )
        if args.demo:
            demo = HERE / "demo" / "cylinder_wake_griebel.gif"
            demo.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gif_path, demo)
            print(f"wrote {demo}")

    return solver, hist, gif_path


if __name__ == "__main__":
    main()
