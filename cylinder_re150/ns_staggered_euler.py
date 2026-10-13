#!/usr/bin/env python3
"""
ns_staggered_euler.py
=====================
2-D incompressible Navier–Stokes on the staggered multiblock mesh
produced by ``mesh_from_known_sizes.py``.

Unknowns
--------
    p, u, v   at cell centres  (pressure / Cartesian velocity)
    Un        at face centres  (normal velocity; continuity lives here)

Discretisation
--------------
    Time        : forward Euler
    Convection  : ``central``  or  ``upwind``  (face value of momentum)
    Diffusion   : central face gradient (owner ↔ neighbour)
    Continuity  : fractional-step projection (Poisson on cell centres,
                  fluxes corrected on faces)

Governing equations (cell volume V, face length L, unit normal n)
-----------------------------------------------------------------
    ∂u/∂t + ∇·(u ⊗ u) = −(1/ρ) ∇p + ν ∇²u
    ∇·u = 0

Forward Euler + projection
--------------------------
    1.  u* = uⁿ + Δt * [ −conv(uⁿ) + ν lap(uⁿ) ]     (no pressure)
    2.  Build face flux φ* = (u*_f · n) L
    3.  Solve  ∇·(∇p') = (ρ/Δt) ∇·u*   for correction p'
    4.  φⁿ⁺¹ = φ* − (Δt/ρ) (∇p'·n) L
        uⁿ⁺¹ = u* − (Δt/ρ) ∇p'
        pⁿ⁺¹ = pⁿ + p'

Boundary conditions (Re = 150 cylinder)
---------------------------------------
    inlet       : u = U∞, v = 0
    outlet      : zero-gradient velocity
    wall_*      : no-slip
    cylinder    : no-slip

Usage
-----
    cd cylinder_re150
    python ns_staggered_euler.py --quick --scheme upwind
    python ns_staggered_euler.py --scheme central --steps 200
    python ns_staggered_euler.py --mesh mesh_output/staggered_result.npz

Needs numpy, scipy, matplotlib.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


# =====================================================================
# mesh / staggered arrays
# =====================================================================
def load_staggered(path: str | Path) -> dict:
    raw = np.load(path, allow_pickle=False)
    bc_names = [str(x) for x in raw["bc_names"].tolist()]
    return dict(
        cell_centers=raw["cell_centers"].astype(np.float64),
        cell_zone=raw["cell_zone"].astype(np.int32),
        face_centers=raw["face_centers"].astype(np.float64),
        face_normals=raw["face_normals"].astype(np.float64),
        face_lengths=raw["face_lengths"].astype(np.float64),
        face_owner=raw["face_owner"].astype(np.int64),
        face_neigh=raw["face_neigh"].astype(np.int64),
        face_kind=raw["face_kind"].astype(np.int8),
        face_bc=raw["face_bc"].astype(np.int32),
        bc_names=bc_names,
        D=float(raw["D"]),
        xc=float(raw["xc"]),
        yc=float(raw["yc"]),
        d_min=float(raw["d_min"]),
        U_meta={
            "d_max": float(raw["d_max"]),
            "growth_ratio": float(raw["growth_ratio"]),
            "a": float(raw["a"]),
        },
    )


def build_topology(S: dict) -> dict:
    """Cell volumes, face spacing, BC masks."""
    P = S["cell_centers"]
    Fc = S["face_centers"]
    n = S["face_normals"]
    L = S["face_lengths"]
    own = S["face_owner"]
    nei = S["face_neigh"]
    Nc = len(P)
    Nf = len(Fc)

    # V = 1/2 Σ (r_f · n_out) L
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
    masks = {}
    for nm, i in bc_index.items():
        masks[nm] = face_bc == i
    masks["boundary"] = face_bc >= 0
    masks["interior"] = interior

    return dict(
        vol=vol,
        d_pn=d_pn,
        bc_index=bc_index,
        masks=masks,
        Nc=Nc,
        Nf=Nf,
    )


def ensure_mesh(npz_path: Path, quick: bool) -> Path:
    """Build staggered mesh if missing; ``--quick`` uses a coarse mesh."""
    if quick:
        out = HERE / "mesh_output" / "staggered_quick.npz"
    else:
        out = Path(npz_path)
        if out.is_file():
            return out

    if out.is_file() and quick:
        return out

    sys.path.insert(0, str(HERE))
    from mesh_from_known_sizes import INPUT, main as mesh_main

    I = dict(INPUT)
    tag = "result"
    if quick:
        I.update(
            d_min=0.05,
            d_max=0.35,
            growth_ratio=1.25,
            max_jump=2.2,
            a=None,
        )
        tag = "quick"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"building staggered mesh → {out}")
    mesh_main(I, tag=tag)
    return HERE / "mesh_output" / f"staggered_{tag}.npz"


# =====================================================================
# face reconstruction & BCs (vectorised)
# =====================================================================
def face_velocity(
    u: np.ndarray,
    v: np.ndarray,
    S: dict,
    topo: dict,
    scheme: str,
    Un_flux: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cartesian (u,v) at staggered face centres.

    central : 0.5 (owner + neighbour)
    upwind  : donor cell from sign(Un_flux)
    """
    own, nei = S["face_owner"], S["face_neigh"]
    interior = topo["masks"]["interior"]
    uf = u[own].copy()
    vf = v[own].copy()
    if scheme == "upwind":
        donor = np.where(
            interior,
            np.where(Un_flux >= 0.0, own, nei),
            own,
        )
        uf = u[donor]
        vf = v[donor]
    else:
        uf[interior] = 0.5 * (u[own[interior]] + u[nei[interior]])
        vf[interior] = 0.5 * (v[own[interior]] + v[nei[interior]])
    return uf, vf


def accumulate_flux_to_cells(
    flux: np.ndarray, own: np.ndarray, nei: np.ndarray, Nc: int
) -> np.ndarray:
    """Σ flux into owner, −flux into neighbour."""
    out = np.zeros(Nc)
    np.add.at(out, own, flux)
    interior = nei >= 0
    np.add.at(out, nei[interior], -flux[interior])
    return out


def viscous_rhs(
    u: np.ndarray,
    v: np.ndarray,
    S: dict,
    topo: dict,
    nu: float,
    U_inf: float,
) -> tuple[np.ndarray, np.ndarray]:
    """ν ∇²φ via face gradients at staggered locations."""
    L = S["face_lengths"]
    own, nei = S["face_owner"], S["face_neigh"]
    d_pn = topo["d_pn"]
    vol = topo["vol"]
    m = topo["masks"]
    interior = m["interior"]
    Nc, Nf = topo["Nc"], topo["Nf"]

    u_f = u[own].copy()
    v_f = v[own].copy()
    if "inlet" in m:
        u_f[m["inlet"]], v_f[m["inlet"]] = U_inf, 0.0
    for w in ("wall_top", "wall_bottom", "cylinder"):
        if w in m:
            u_f[m[w]] = 0.0
            v_f[m[w]] = 0.0
    # outlet: already owner (zero gradient)

    gdu = np.empty(Nf)
    gdv = np.empty(Nf)
    gdu[interior] = (u[nei[interior]] - u[own[interior]]) / d_pn[interior]
    gdv[interior] = (v[nei[interior]] - v[own[interior]]) / d_pn[interior]
    bnd = ~interior
    gdu[bnd] = (u_f[bnd] - u[own[bnd]]) / (0.5 * d_pn[bnd])
    gdv[bnd] = (v_f[bnd] - v[own[bnd]]) / (0.5 * d_pn[bnd])

    ru = accumulate_flux_to_cells(nu * gdu * L, own, nei, Nc) / vol
    rv = accumulate_flux_to_cells(nu * gdv * L, own, nei, Nc) / vol
    return ru, rv


def green_gauss_grad(
    phi: np.ndarray, S: dict, topo: dict, phi_bnd: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    L = S["face_lengths"]
    n = S["face_normals"]
    own, nei = S["face_owner"], S["face_neigh"]
    vol = topo["vol"]
    interior = topo["masks"]["interior"]
    Nc = topo["Nc"]

    pf = phi[own].copy()
    pf[interior] = 0.5 * (phi[own[interior]] + phi[nei[interior]])
    if phi_bnd is not None:
        pf[~interior] = phi_bnd[~interior]

    gx = accumulate_flux_to_cells(pf * n[:, 0] * L, own, nei, Nc) / vol
    gy = accumulate_flux_to_cells(pf * n[:, 1] * L, own, nei, Nc) / vol
    return gx, gy


def divergence_from_Un(Un: np.ndarray, S: dict, topo: dict) -> np.ndarray:
    L = S["face_lengths"]
    return (
        accumulate_flux_to_cells(Un * L, S["face_owner"], S["face_neigh"], topo["Nc"])
        / topo["vol"]
    )


# =====================================================================
# pressure Poisson
# =====================================================================
def build_poisson(S: dict, topo: dict) -> sparse.csr_matrix:
    """
    Symmetric FV Laplacian on *integrated* residuals:

        Σ_f (L/d) (p_nei − p_own)  =  (ρ/Δt) Σ_f Un L   [= (ρ/Δt) V div]

    Neumann on boundaries; pressure pinned at cell 0.
    """
    L = S["face_lengths"]
    own, nei = S["face_owner"], S["face_neigh"]
    d_pn = topo["d_pn"]
    Nc = topo["Nc"]
    interior = topo["masks"]["interior"]

    o = own[interior]
    c = nei[interior]
    g = L[interior] / d_pn[interior]

    rows = np.concatenate([o, o, c, c])
    cols = np.concatenate([c, o, o, c])
    data = np.concatenate([g, -g, g, -g])

    A = sparse.coo_matrix((data, (rows, cols)), shape=(Nc, Nc)).tocsr()
    A = A.tolil()
    A[0, :] = 0.0
    A[:, 0] = 0.0
    A[0, 0] = 1.0
    return A.tocsr()


# =====================================================================
# solver
# =====================================================================
class StaggeredEulerNS:
    """Fractional-step incompressible NS: Euler + central/upwind."""

    def __init__(
        self,
        S: dict,
        rho: float = 1.0,
        nu: float = 1.0 / 150.0,
        U_inf: float = 1.0,
        scheme: str = "upwind",
    ) -> None:
        self.S = S
        self.topo = build_topology(S)
        self.rho = float(rho)
        self.nu = float(nu)
        self.U_inf = float(U_inf)
        scheme = str(scheme).lower().strip()
        if scheme not in ("central", "upwind"):
            raise ValueError("scheme must be 'central' or 'upwind'")
        self.scheme = scheme
        self.t = 0.0

        Nc, Nf = self.topo["Nc"], self.topo["Nf"]
        self.u = np.full(Nc, self.U_inf)
        self.v = np.zeros(Nc)
        self.p = np.zeros(Nc)
        self.Un = np.zeros(Nf)
        self._A = build_poisson(S, self.topo)
        self._p_corr = np.zeros(Nc)
        self._cg_info = 0
        self._init_divergenceless()

    def _enforce_dirichlet_Un(self, Un: np.ndarray) -> np.ndarray:
        """Keep projected interior fluxes; overwrite only Dirichlet faces."""
        S, topo = self.S, self.topo
        n = S["face_normals"]
        m = topo["masks"]
        Un = Un.copy()
        if "inlet" in m:
            # n points owner→outside at inlet (left): freestream · n
            Un[m["inlet"]] = self.U_inf * n[m["inlet"], 0] + 0.0 * n[m["inlet"], 1]
        for w in ("wall_top", "wall_bottom", "cylinder"):
            if w in m:
                Un[m[w]] = 0.0
        return Un

    def _init_divergenceless(self) -> None:
        """One projection of the freestream IC onto a solenoidal Un."""
        S, topo = self.S, self.topo
        n = S["face_normals"]
        uf, vf = face_velocity(
            self.u, self.v, S, topo, "central", np.zeros(topo["Nf"])
        )
        Un = uf * n[:, 0] + vf * n[:, 1]
        self.Un = self._project_Un(Un, dt=1.0)  # dt cancels in homogeneous IC proj
        self.Un = self._enforce_dirichlet_Un(self.Un)

    def _project_Un(self, Un_star: np.ndarray, dt: float) -> np.ndarray:
        """Return solenoidal Un; also stores last pressure correction in ``_p_corr``."""
        S, topo = self.S, self.topo
        own, nei = S["face_owner"], S["face_neigh"]
        d_pn = topo["d_pn"]
        interior = topo["masks"]["interior"]
        rho = self.rho

        Un_star = self._enforce_dirichlet_Un(Un_star)
        div = divergence_from_Un(Un_star, S, topo)
        # integrated Neumann system needs Σ rhs = 0 (compatibility)
        rhs = (rho / dt) * topo["vol"] * div
        rhs -= float(np.mean(rhs))
        rhs[0] = 0.0
        p_corr, info = cg(self._A, rhs, rtol=1e-8, maxiter=2000)
        p_corr = np.nan_to_num(np.asarray(p_corr), nan=0.0)
        p_corr[0] = 0.0
        self._p_corr = p_corr
        self._cg_info = int(info)

        Un_new = Un_star.copy()
        dp_dn = (p_corr[nei[interior]] - p_corr[own[interior]]) / d_pn[interior]
        Un_new[interior] = Un_star[interior] - (dt / rho) * dp_dn
        return self._enforce_dirichlet_Un(Un_new)

    def _convective(self, u, v, Un) -> tuple[np.ndarray, np.ndarray]:
        S, topo = self.S, self.topo
        L = S["face_lengths"]
        Un_work = self._enforce_dirichlet_Un(Un)
        uf, vf = face_velocity(u, v, S, topo, self.scheme, Un_work)
        # Dirichlet face momentum for flux (inlet / walls)
        m = topo["masks"]
        if "inlet" in m:
            uf[m["inlet"]], vf[m["inlet"]] = self.U_inf, 0.0
        for w in ("wall_top", "wall_bottom", "cylinder"):
            if w in m:
                uf[m[w]] = 0.0
                vf[m[w]] = 0.0
        ru = accumulate_flux_to_cells(
            Un_work * uf * L, S["face_owner"], S["face_neigh"], topo["Nc"]
        )
        rv = accumulate_flux_to_cells(
            Un_work * vf * L, S["face_owner"], S["face_neigh"], topo["Nc"]
        )
        return -ru / topo["vol"], -rv / topo["vol"]

    def step(self, dt: float) -> dict:
        """One forward-Euler + projection step."""
        S, topo = self.S, self.topo
        n = S["face_normals"]
        rho = self.rho

        self.Un = self._enforce_dirichlet_Un(self.Un)
        cu, cv = self._convective(self.u, self.v, self.Un)
        du, dv = viscous_rhs(self.u, self.v, S, topo, self.nu, self.U_inf)

        # 1) predictor (cell velocity)
        u_star = self.u + dt * (cu + du)
        v_star = self.v + dt * (cv + dv)

        # staggered face normal speed from predictor (central), then project
        uf, vf = face_velocity(u_star, v_star, S, topo, "central", self.Un)
        Un_star = uf * n[:, 0] + vf * n[:, 1]
        Un_new = self._project_Un(Un_star, dt)
        p_corr = self._p_corr

        gx, gy = green_gauss_grad(p_corr, S, topo)
        self.u = u_star - (dt / rho) * gx
        self.v = v_star - (dt / rho) * gy
        self.p = self.p + p_corr
        self.Un = Un_new  # keep solenoidal face field (do NOT rebuild from cells)
        self.t += dt

        div_after = divergence_from_Un(self.Un, S, topo)
        return dict(
            t=self.t,
            max_div=float(np.max(np.abs(div_after))),
            max_u=float(np.max(np.abs(self.u))),
            max_v=float(np.max(np.abs(self.v))),
            cg_info=int(self._cg_info),
        )

    def stable_dt(self, cfl: float = 0.4) -> float:
        hmin = float(np.min(np.sqrt(self.topo["vol"])))
        U = max(abs(self.U_inf), float(np.max(np.abs(self.Un))), 1e-6)
        return float(min(cfl * hmin / U, 0.25 * hmin * hmin / max(self.nu, 1e-12)))


# =====================================================================
# I/O
# =====================================================================
def save_fields(path: Path, solver: StaggeredEulerNS, hist: list) -> None:
    np.savez_compressed(
        path,
        t=solver.t,
        u=solver.u,
        v=solver.v,
        p=solver.p,
        Un=solver.Un,
        cell_centers=solver.S["cell_centers"],
        face_centers=solver.S["face_centers"],
        scheme=np.array(solver.scheme),
        hist_t=np.array([h["t"] for h in hist]),
        hist_div=np.array([h["max_div"] for h in hist]),
        hist_umax=np.array([h["max_u"] for h in hist]),
    )


def plot_solution(path: Path, solver: StaggeredEulerNS, hist: list) -> None:
    P = solver.S["cell_centers"]
    xc, yc, R = solver.S["xc"], solver.S["yc"], solver.S["D"] / 2
    speed = np.hypot(solver.u, solver.v)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    sc = ax.scatter(
        P[:, 0],
        P[:, 1],
        c=speed,
        s=4,
        cmap="viridis",
        vmax=max(1.5, float(np.percentile(speed, 99))),
    )
    ax.add_patch(plt.Circle((xc, yc), R, fc="white", ec="k", lw=1.2, zorder=3))
    ax.set_aspect("equal")
    ax.set_xlim(xc - 3, xc + 8)
    ax.set_ylim(yc - 4, yc + 4)
    ax.set_title(
        f"|U|  scheme={solver.scheme}  t={solver.t:.3f}",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    tt = [h["t"] for h in hist]
    ax.semilogy(tt, [max(h["max_div"], 1e-16) for h in hist], "k-", lw=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("max |∇·U|")
    ax.set_title("Continuity residual after projection", loc="left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# main
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Euler NS on staggered cylinder mesh (central / upwind)"
    )
    p.add_argument(
        "--mesh",
        type=str,
        default=str(HERE / "mesh_output" / "staggered_result.npz"),
        help="staggered_*.npz from mesh_from_known_sizes.py",
    )
    p.add_argument(
        "--scheme",
        choices=("central", "upwind"),
        default="upwind",
        help="convective face interpolation",
    )
    p.add_argument("--steps", type=int, default=50, help="number of time steps")
    p.add_argument("--dt", type=float, default=None, help="override Δt")
    p.add_argument("--cfl", type=float, default=0.4)
    p.add_argument("--Re", type=float, default=150.0)
    p.add_argument("--U-inf", type=float, default=1.0)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument(
        "--quick",
        action="store_true",
        help="build a coarse staggered mesh and run a short smoke test",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(HERE / "output_ns"),
        help="directory for fields / plots",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mesh_path = ensure_mesh(Path(args.mesh), quick=args.quick)
    S = load_staggered(mesh_path)
    nu = args.U_inf * S["D"] / args.Re
    solver = StaggeredEulerNS(
        S, rho=args.rho, nu=nu, U_inf=args.U_inf, scheme=args.scheme
    )

    dt = args.dt if args.dt is not None else solver.stable_dt(args.cfl)
    n_steps = 20 if args.quick and args.steps == 50 else args.steps

    print("=" * 72)
    print("STAGGERED NS  |  forward Euler  |  convection:", solver.scheme)
    print("=" * 72)
    print(f"mesh     : {mesh_path}")
    print(f"cells    : {solver.topo['Nc']:,}   faces: {solver.topo['Nf']:,}")
    print(
        f"Re={args.Re:g}  nu={nu:.6g}  U∞={args.U_inf:g}  "
        f"dt={dt:.6g}  steps={n_steps}"
    )
    print("-" * 72)

    hist = []
    t0 = time.time()
    for k in range(1, n_steps + 1):
        info = solver.step(dt)
        hist.append(info)
        if k == 1 or k == n_steps or k % max(1, n_steps // 10) == 0:
            print(
                f"step {k:5d}/{n_steps}  t={info['t']:.4f}  "
                f"max|div|={info['max_div']:.3e}  "
                f"max|u|={info['max_u']:.3f}  cg={info['cg_info']}"
            )
    elapsed = time.time() - t0
    print("-" * 72)
    print(f"done in {elapsed:.2f}s  ({elapsed / n_steps * 1000:.1f} ms/step)")

    tag = f"{solver.scheme}_{'quick' if args.quick else 'run'}"
    fields = out / f"fields_{tag}.npz"
    plot = out / f"plot_{tag}.png"
    save_fields(fields, solver, hist)
    plot_solution(plot, solver, hist)
    print(f"wrote {fields}")
    print(f"wrote {plot}")
    return solver, hist


if __name__ == "__main__":
    main()
