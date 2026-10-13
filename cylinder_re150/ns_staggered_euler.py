#!/usr/bin/env python3
"""
ns_staggered_euler.py
=====================
2-D incompressible Navier–Stokes on a MAC staggered grid, following the
finite-difference projection method of Griebel, Dornseifer & Neunhoeffer
(1998) as implemented in:

    https://github.com/CaptainProton42/NavierStokes

Algorithm (per time step) — identical structure to ``main.c`` there
-------------------------------------------------------------------
1. Set boundary conditions on staggered ``u``, ``v``.
2. Intermediate fields ``F``, ``G`` (explicit Euler, no pressure)::

       F = u + Δt · ( (1/Re) ∇²u − ∂(u²)/∂x − ∂(uv)/∂y )
       G = v + Δt · ( (1/Re) ∇²v − ∂(uv)/∂x − ∂(v²)/∂y )

3. Pressure Poisson by SOR::

       ∇²p = (1/Δt) ( ∂F/∂x + ∂G/∂y )

4. Correct velocities::

       u ← F − Δt ∂p/∂x
       v ← G − Δt ∂p/∂y

Convection: Griebel donor-cell / γ-scheme (``integration.c``)
------------------------------------------------------------
    γ = 0        → pure central difference     (``--scheme central``)
    γ = 1        → full donor-cell upwind      (``--scheme upwind``)
    γ = max(CFL) → adaptive blend              (``--scheme donor``, reference)

Indexing follows the reference: arrays are ``(i_max+2) × (j_max+2)`` with
a one-cell ghost layer; fluid indices run ``i,j = 1 … i_max / j_max``.

Cylinder Re = 150 uses an immersed no-slip disk on a Cartesian MAC grid.

Usage
-----
    cd cylinder_re150
    python ns_staggered_euler.py --quick --scheme donor
    python ns_staggered_euler.py --scheme central --steps 200
    python ns_staggered_euler.py --scheme upwind --steps 200

Needs numpy, matplotlib.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


# =====================================================================
# Donor-cell / central stencils — ported from integration.c
# =====================================================================
def du2_dx(u, i, j, dx, gamma):
    s1 = 0.5 * (u[i, j] + u[i + 1, j])
    s2 = 0.5 * (u[i - 1, j] + u[i, j])
    s3 = abs(s1) * 0.5 * (u[i, j] - u[i + 1, j])
    s4 = abs(s2) * 0.5 * (u[i - 1, j] - u[i, j])
    return (s1 * s1 - s2 * s2) / dx + gamma * (s3 - s4) / dx


def duv_dy(u, v, i, j, dy, gamma):
    s1 = 0.5 * (v[i, j] + v[i + 1, j])
    s2 = 0.5 * (v[i, j - 1] + v[i + 1, j - 1])
    s3 = s1 * 0.5 * (u[i, j] + u[i, j + 1])
    s4 = s2 * 0.5 * (u[i, j - 1] + u[i, j])
    s5 = abs(s1) * 0.5 * (u[i, j] - u[i, j + 1])
    s6 = abs(s2) * 0.5 * (u[i, j - 1] - u[i, j])
    return (s3 - s4) / dy + gamma * (s5 - s6) / dy


def dv2_dy(v, i, j, dy, gamma):
    s1 = 0.5 * (v[i, j] + v[i, j + 1])
    s2 = 0.5 * (v[i, j - 1] + v[i, j])
    s3 = abs(s1) * 0.5 * (v[i, j] - v[i, j + 1])
    s4 = abs(s2) * 0.5 * (v[i, j - 1] - v[i, j])
    return (s1 * s1 - s2 * s2) / dy + gamma * (s3 - s4) / dy


def duv_dx(u, v, i, j, dx, gamma):
    s1 = 0.5 * (u[i, j] + u[i, j + 1])
    s2 = 0.5 * (u[i - 1, j] + u[i - 1, j + 1])
    s3 = s1 * 0.5 * (v[i, j] + v[i + 1, j])
    s4 = s2 * 0.5 * (v[i - 1, j] + v[i, j])
    s5 = abs(s1) * 0.5 * (v[i, j] - v[i + 1, j])
    s6 = abs(s2) * 0.5 * (v[i - 1, j] - v[i, j])
    return (s3 - s4) / dx + gamma * (s5 - s6) / dx


def d2u_dx2(u, i, j, dx):
    return (u[i + 1, j] - 2.0 * u[i, j] + u[i - 1, j]) / (dx * dx)


def d2u_dy2(u, i, j, dy):
    return (u[i, j + 1] - 2.0 * u[i, j] + u[i, j - 1]) / (dy * dy)


def d2v_dx2(v, i, j, dx):
    return (v[i + 1, j] - 2.0 * v[i, j] + v[i - 1, j]) / (dx * dx)


def d2v_dy2(v, i, j, dy):
    return (v[i, j + 1] - 2.0 * v[i, j] + v[i, j - 1]) / (dy * dy)


def dp_dx(p, i, j, dx):
    return (p[i + 1, j] - p[i, j]) / dx


def dp_dy(p, i, j, dy):
    return (p[i, j + 1] - p[i, j]) / dy


def FG(F, G, u, v, i_max, j_max, Re, dt, dx, dy, gamma, solid):
    """CaptainProton42 ``FG()`` — explicit Euler predictors."""
    inv_Re = 1.0 / Re
    for i in range(1, i_max + 1):
        for j in range(1, j_max + 1):
            if solid[i, j]:
                if i <= i_max - 1:
                    F[i, j] = 0.0
                if j <= j_max - 1:
                    G[i, j] = 0.0
                continue
            if i <= i_max - 1:
                F[i, j] = u[i, j] + dt * (
                    inv_Re * (d2u_dx2(u, i, j, dx) + d2u_dy2(u, i, j, dy))
                    - du2_dx(u, i, j, dx, gamma)
                    - duv_dy(u, v, i, j, dy, gamma)
                )
            if j <= j_max - 1:
                G[i, j] = v[i, j] + dt * (
                    inv_Re * (d2v_dx2(v, i, j, dx) + d2v_dy2(v, i, j, dy))
                    - duv_dx(u, v, i, j, dx, gamma)
                    - dv2_dy(v, i, j, dy, gamma)
                )
    return F, G


def L2_fluid(m, i_max, j_max, solid):
    s = 0.0
    n = 0
    for i in range(1, i_max + 1):
        for j in range(1, j_max + 1):
            if solid[i, j]:
                continue
            s += m[i, j] * m[i, j]
            n += 1
    return np.sqrt(s / max(n, 1))


def SOR(p, i_max, j_max, dx, dy, res, RHS, omega, eps, max_it, solid):
    """CaptainProton42 ``SOR()`` with Neumann ghosts + immersed solid."""
    dydy = dy * dy
    dxdx = dx * dx
    denom = 2.0 * (1.0 / dxdx + 1.0 / dydy)
    norm_p = L2_fluid(p, i_max, j_max, solid)

    for it in range(max_it):
        # Neumann ghosts (reference fills border with neighbour values)
        for j in range(1, j_max + 1):
            p[0, j] = p[1, j]
            p[i_max + 1, j] = p[i_max, j]
        for i in range(1, i_max + 1):
            p[i, 0] = p[i, 1]
            p[i, j_max + 1] = p[i, j_max]

        for i in range(1, i_max + 1):
            for j in range(1, j_max + 1):
                if solid[i, j]:
                    p[i, j] = 0.0
                    continue
                # solid neighbours → Neumann (use own value)
                pe = p[i, j] if solid[i + 1, j] else p[i + 1, j]
                pw = p[i, j] if solid[i - 1, j] else p[i - 1, j]
                pn = p[i, j] if solid[i, j + 1] else p[i, j + 1]
                ps = p[i, j] if solid[i, j - 1] else p[i, j - 1]
                p[i, j] = (1.0 - omega) * p[i, j] + (omega / denom) * (
                    (pe + pw) / dxdx + (pn + ps) / dydy - RHS[i, j]
                )

        for i in range(1, i_max + 1):
            for j in range(1, j_max + 1):
                if solid[i, j]:
                    res[i, j] = 0.0
                    continue
                pe = p[i, j] if solid[i + 1, j] else p[i + 1, j]
                pw = p[i, j] if solid[i - 1, j] else p[i - 1, j]
                pn = p[i, j] if solid[i, j + 1] else p[i, j + 1]
                ps = p[i, j] if solid[i, j - 1] else p[i, j - 1]
                res[i, j] = (
                    (pe - 2.0 * p[i, j] + pw) / dxdx
                    + (pn - 2.0 * p[i, j] + ps) / dydy
                    - RHS[i, j]
                )

        res_norm = L2_fluid(res, i_max, j_max, solid)
        if res_norm <= eps * (norm_p + 0.01):
            return it + 1, res_norm
        norm_p = L2_fluid(p, i_max, j_max, solid)

    return max_it, L2_fluid(res, i_max, j_max, solid)


# =====================================================================
# Solver
# =====================================================================
class GriebelMACSolver:
    """
    Griebel / CaptainProton42 MAC projection solver (cylinder Re=150).

    Fields are ghost-padded ``(i_max+2, j_max+2)`` like the C reference.
    """

    def __init__(
        self,
        i_max: int,
        j_max: int,
        a: float,
        b: float,
        Re: float = 150.0,
        U_inf: float = 1.0,
        scheme: str = "donor",
        omega: float = 1.7,
        sor_eps: float = 1e-3,
        sor_max_it: int = 200,
        tau: float = 0.5,
        cx: float = 0.0,
        cy: float = 0.0,
        D: float = 1.0,
    ) -> None:
        self.i_max = int(i_max)
        self.j_max = int(j_max)
        self.a = float(a)
        self.b = float(b)
        self.dx = a / i_max
        self.dy = b / j_max
        self.Re = float(Re)
        self.U_inf = float(U_inf)
        self.scheme = str(scheme).lower().strip()
        if self.scheme not in ("central", "upwind", "donor"):
            raise ValueError("scheme must be central | upwind | donor")
        self.omega = float(omega)
        self.sor_eps = float(sor_eps)
        self.sor_max_it = int(sor_max_it)
        self.tau = float(tau)
        self.cx = float(cx)
        self.cy = float(cy)
        self.D = float(D)
        self.t = 0.0

        sh = (i_max + 2, j_max + 2)
        self.u = np.full(sh, self.U_inf)
        self.v = np.zeros(sh)
        self.p = np.zeros(sh)
        self.F = np.zeros(sh)
        self.G = np.zeros(sh)
        self.RHS = np.zeros(sh)
        self.res = np.zeros(sh)

        # cell-centre coordinates (fluid i,j = 1..imax)
        self.x = (np.arange(i_max + 2) - 0.5) * self.dx
        self.y = (np.arange(j_max + 2) - 0.5) * self.dy
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        R = 0.5 * D
        self.solid = (X - cx) ** 2 + (Y - cy) ** 2 <= R * R
        # ghosts never treated as fluid solid for Poisson fluid loops
        self.solid[0, :] = False
        self.solid[-1, :] = False
        self.solid[:, 0] = False
        self.solid[:, -1] = False

        self._apply_bc()

    def _apply_bc(self) -> None:
        """
        Cylinder channel BCs (analogue of ``boundaries.c`` set_inflow/noslip):
          west  = inlet U∞
          east  = Neumann outlet
          south/north = freestream
          cylinder cells → no-slip (u=v=0 on adjacent staggered faces)
        """
        U = self.U_inf
        im, jm = self.i_max, self.j_max
        u, v = self.u, self.v

        # LEFT inlet
        for j in range(1, jm + 1):
            u[0, j] = U
            v[0, j] = 2.0 * 0.0 - v[1, j]
        # RIGHT outlet (Neumann on u; average for v)
        for j in range(1, jm + 1):
            u[im, j] = u[im - 1, j]
            v[im + 1, j] = v[im, j]
        # BOTTOM
        for i in range(1, im + 1):
            v[i, 0] = 0.0
            u[i, 0] = 2.0 * U - u[i, 1]
        # TOP
        for i in range(1, im + 1):
            v[i, jm] = 0.0
            u[i, jm + 1] = 2.0 * U - u[i, jm]

        # Immersed cylinder: zero velocity in / around solid cells
        for i in range(1, im + 1):
            for j in range(1, jm + 1):
                if self.solid[i, j]:
                    u[i, j] = 0.0
                    v[i, j] = 0.0
                    u[i - 1, j] = 0.0
                    v[i, j - 1] = 0.0

    def _gamma(self, dt: float) -> float:
        if self.scheme == "central":
            return 0.0
        if self.scheme == "upwind":
            return 1.0
        umax = float(np.max(np.abs(self.u)))
        vmax = float(np.max(np.abs(self.v)))
        return float(max(umax * dt / self.dx, vmax * dt / self.dy, 0.0))

    def stable_dt(self) -> float:
        """Adaptive Δt — CaptainProton42 ``main.c`` formula with factor τ."""
        umax = max(float(np.max(np.abs(self.u))), 1e-12)
        vmax = max(float(np.max(np.abs(self.v))), 1e-12)
        dx, dy = self.dx, self.dy
        dt_diff = self.Re / 2.0 / (1.0 / (dx * dx) + 1.0 / (dy * dy))
        return float(self.tau * min(dt_diff, dx / umax, dy / vmax))

    def step(self, dt: float | None = None) -> dict:
        im, jm = self.i_max, self.j_max
        dx, dy = self.dx, self.dy
        self._apply_bc()
        if dt is None:
            dt = self.stable_dt()
        gamma = self._gamma(dt)

        # F, G
        self.F[:, :] = self.u
        self.G[:, :] = self.v
        FG(
            self.F,
            self.G,
            self.u,
            self.v,
            im,
            jm,
            self.Re,
            dt,
            dx,
            dy,
            gamma,
            self.solid,
        )
        # BC faces for F,G (reference copies boundary velocities)
        self.F[0, :] = self.u[0, :]
        self.F[im, :] = self.u[im, :]
        self.G[:, 0] = self.v[:, 0]
        self.G[:, jm] = self.v[:, jm]

        # RHS of Poisson
        for i in range(1, im + 1):
            for j in range(1, jm + 1):
                if self.solid[i, j]:
                    self.RHS[i, j] = 0.0
                else:
                    self.RHS[i, j] = (1.0 / dt) * (
                        (self.F[i, j] - self.F[i - 1, j]) / dx
                        + (self.G[i, j] - self.G[i, j - 1]) / dy
                    )

        sor_it, sor_res = SOR(
            self.p,
            im,
            jm,
            dx,
            dy,
            self.res,
            self.RHS,
            self.omega,
            self.sor_eps,
            self.sor_max_it,
            self.solid,
        )

        # Velocity update
        for i in range(1, im + 1):
            for j in range(1, jm + 1):
                if self.solid[i, j]:
                    self.u[i, j] = 0.0
                    self.v[i, j] = 0.0
                    continue
                if i <= im - 1:
                    self.u[i, j] = self.F[i, j] - dt * dp_dx(self.p, i, j, dx)
                if j <= jm - 1:
                    self.v[i, j] = self.G[i, j] - dt * dp_dy(self.p, i, j, dy)

        self._apply_bc()
        self.t += dt

        # divergence on fluid cells
        max_div = 0.0
        for i in range(1, im + 1):
            for j in range(1, jm + 1):
                if self.solid[i, j]:
                    continue
                div = (self.u[i, j] - self.u[i - 1, j]) / dx + (
                    self.v[i, j] - self.v[i, j - 1]
                ) / dy
                max_div = max(max_div, abs(div))

        return dict(
            t=self.t,
            dt=dt,
            gamma=gamma,
            max_div=float(max_div),
            max_u=float(np.max(np.abs(self.u))),
            max_v=float(np.max(np.abs(self.v))),
            sor_it=int(sor_it),
            sor_res=float(sor_res),
        )

    def cell_speed(self) -> np.ndarray:
        im, jm = self.i_max, self.j_max
        speed = np.full((im, jm), np.nan)
        for i in range(1, im + 1):
            for j in range(1, jm + 1):
                if self.solid[i, j]:
                    continue
                uc = 0.5 * (self.u[i - 1, j] + self.u[i, j])
                vc = 0.5 * (self.v[i, j - 1] + self.v[i, j])
                speed[i - 1, j - 1] = np.hypot(uc, vc)
        return speed


# =====================================================================
# I/O
# =====================================================================
def save_fields(path: Path, solver: GriebelMACSolver, hist: list) -> None:
    np.savez_compressed(
        path,
        t=solver.t,
        u=solver.u,
        v=solver.v,
        p=solver.p,
        speed=solver.cell_speed(),
        x=solver.x,
        y=solver.y,
        scheme=np.array(solver.scheme),
        hist_t=np.array([h["t"] for h in hist]),
        hist_div=np.array([h["max_div"] for h in hist]),
        hist_umax=np.array([h["max_u"] for h in hist]),
        hist_gamma=np.array([h["gamma"] for h in hist]),
    )


def _speed_pcolormesh(ax, solver: GriebelMACSolver, vmax: float | None = None):
    """Draw |U| field + cylinder on ``ax``; return mappable."""
    speed = solver.cell_speed()
    im, jm = solver.i_max, solver.j_max
    xe = np.linspace(0.0, solver.a, im + 1)
    ye = np.linspace(0.0, solver.b, jm + 1)
    if vmax is None:
        finite = speed[np.isfinite(speed)]
        vmax = max(1.5, float(np.percentile(finite, 98))) if finite.size else 1.5
    pcm = ax.pcolormesh(
        xe,
        ye,
        np.ma.masked_invalid(speed).T,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
    )
    ax.add_patch(
        plt.Circle(
            (solver.cx, solver.cy),
            0.5 * solver.D,
            fc="white",
            ec="k",
            lw=1.2,
        )
    )
    ax.set_aspect("equal")
    ax.set_xlim(max(0.0, solver.cx - 2), min(solver.a, solver.cx + 10))
    ax.set_ylim(max(0.0, solver.cy - 3.5), min(solver.b, solver.cy + 3.5))
    return pcm


def save_wake_frame(
    path: Path,
    solver: GriebelMACSolver,
    vmax: float = 1.8,
) -> Path:
    """Single wake frame for GIF assembly (``wake_tXXXX.png``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    pcm = _speed_pcolormesh(ax, solver, vmax=vmax)
    ax.set_title(
        f"Re={solver.Re:g}  scheme={solver.scheme}  tU/D={solver.t:.2f}",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02, label="|U|")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_solution(path: Path, solver: GriebelMACSolver, hist: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    pcm = _speed_pcolormesh(ax, solver)
    ax.set_title(
        f"|U|  Griebel/MAC  scheme={solver.scheme}  t={solver.t:.3f}",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    ax.semilogy(
        [h["t"] for h in hist],
        [max(h["max_div"], 1e-16) for h in hist],
        "k-",
        lw=1.5,
    )
    ax.set_xlabel("t")
    ax.set_ylabel("max |∇·U|")
    ax.set_title("Continuity residual (after SOR)", loc="left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def build_gif(
    frame_paths: list[Path],
    gif_path: Path,
    fps: int = 8,
    max_width: int = 900,
) -> Path:
    """Assemble wake frames into an animated GIF (shared palette)."""
    try:
        from viz import save_wake_gif

        return save_wake_gif(frame_paths, gif_path, fps=fps, max_width=max_width)
    except Exception:
        # Fallback if viz import path differs
        from PIL import Image

        rgb = []
        for p in frame_paths:
            im = Image.open(p).convert("RGB")
            if max_width and im.width > max_width:
                h = int(im.height * max_width / im.width)
                im = im.resize((max_width, h), Image.Resampling.LANCZOS)
            rgb.append(im)
        pal = rgb[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        frames = [pal] + [im.quantize(colors=256, palette=pal) for im in rgb[1:]]
        gif_path = Path(gif_path)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            gif_path,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=max(int(1000 / max(fps, 1)), 40),
            loop=0,
            optimize=False,
            disposal=2,
        )
        return gif_path


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Griebel MAC NS solver "
            "(method of github.com/CaptainProton42/NavierStokes)"
        )
    )
    p.add_argument(
        "--scheme",
        choices=("central", "upwind", "donor"),
        default="donor",
        help="γ=0 / γ=1 / adaptive donor-cell (reference default)",
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--Re", type=float, default=150.0)
    p.add_argument("--U-inf", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--omega", type=float, default=1.7)
    p.add_argument("--sor-eps", type=float, default=1e-3)
    p.add_argument("--sor-max-it", type=int, default=200)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out-dir", type=str, default=str(HERE / "output_ns"))
    p.add_argument("--i-max", type=int, default=None)
    p.add_argument("--j-max", type=int, default=None)
    p.add_argument(
        "--gif",
        action="store_true",
        default=True,
        help="write wake GIF (default: on)",
    )
    p.add_argument("--no-gif", action="store_true", help="skip GIF")
    p.add_argument("--gif-fps", type=int, default=8)
    p.add_argument(
        "--frame-every",
        type=int,
        default=2,
        help="save a wake PNG every N steps for the GIF",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="also copy GIF to demo/cylinder_wake_griebel.gif",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_gif = bool(args.gif) and not bool(args.no_gif)

    # Domain: cylinder in a channel (physical units, D=1)
    D = 1.0
    if args.quick:
        a, b = 16.0, 8.0
        cx, cy = 4.0, 4.0
        i_max = args.i_max or 64
        j_max = args.j_max or 32
        n_steps = 30 if args.steps == 50 else args.steps
        sor_max = min(args.sor_max_it, 80)
    else:
        a, b = 24.0, 12.0
        cx, cy = 6.0, 6.0
        i_max = args.i_max or 160
        j_max = args.j_max or 80
        n_steps = args.steps
        sor_max = args.sor_max_it

    # GIF-oriented quick defaults: enough time to form a wake
    if write_gif and args.quick and args.steps == 50:
        n_steps = 120

    solver = GriebelMACSolver(
        i_max=i_max,
        j_max=j_max,
        a=a,
        b=b,
        Re=args.Re,
        U_inf=args.U_inf,
        scheme=args.scheme,
        omega=args.omega,
        sor_eps=args.sor_eps,
        sor_max_it=sor_max,
        tau=args.tau,
        cx=cx,
        cy=cy,
        D=D,
    )

    print("=" * 72)
    print("GRIEBEL MAC NS  |  method of CaptainProton42/NavierStokes")
    print("=" * 72)
    print(f"grid     : {i_max} x {j_max}   dx={solver.dx:.4g}  dy={solver.dy:.4g}")
    print(f"domain   : {a:g} x {b:g}   cylinder=({cx:g},{cy:g}) D={D:g}")
    print(f"scheme   : {solver.scheme}   Re={args.Re:g}  τ={args.tau:g}  ω={args.omega:g}")
    print(f"steps    : {n_steps}" + (f"  t_end={args.t_end}" if args.t_end else ""))
    if write_gif:
        print(f"GIF      : every {args.frame_every} steps  @ {args.gif_fps} fps")
    print("-" * 72)

    # clear old wake frames in this out dir
    for old in out.glob("wake_t*.png"):
        old.unlink()

    hist = []
    frames: list[Path] = []
    frame_every = max(int(args.frame_every), 1)
    t0 = time.time()
    k = 0
    while True:
        k += 1
        info = solver.step(args.dt)
        hist.append(info)
        if write_gif and (k == 1 or k % frame_every == 0):
            fp = out / f"wake_t{solver.t:07.2f}.png"
            save_wake_frame(fp, solver)
            frames.append(fp)
        if k == 1 or k == n_steps or k % max(1, n_steps // 10) == 0:
            print(
                f"step {k:5d}  t={info['t']:.4f}  dt={info['dt']:.4g}  "
                f"γ={info['gamma']:.3f}  max|div|={info['max_div']:.3e}  "
                f"max|u|={info['max_u']:.3f}  SOR={info['sor_it']}"
            )
        if k >= n_steps or (args.t_end is not None and solver.t >= args.t_end):
            break
    # final frame
    if write_gif and (not frames or frames[-1].name != f"wake_t{solver.t:07.2f}.png"):
        fp = out / f"wake_t{solver.t:07.2f}.png"
        save_wake_frame(fp, solver)
        frames.append(fp)

    elapsed = time.time() - t0
    print("-" * 72)
    print(f"done in {elapsed:.2f}s  ({elapsed / k * 1000:.1f} ms/step)")

    tag = f"{solver.scheme}_{'quick' if args.quick else 'run'}"
    fields = out / f"fields_{tag}.npz"
    plot = out / f"plot_{tag}.png"
    save_fields(fields, solver, hist)
    plot_solution(plot, solver, hist)
    print(f"wrote {fields}")
    print(f"wrote {plot}")

    gif_path = None
    if write_gif and frames:
        gif_path = out / f"cylinder_wake_{solver.scheme}.gif"
        build_gif(frames, gif_path, fps=int(args.gif_fps))
        print(
            f"wrote {gif_path}  ({len(frames)} frames, "
            f"{gif_path.stat().st_size / 1e6:.2f} MB)"
        )
        if args.demo:
            demo = HERE / "demo" / "cylinder_wake_griebel.gif"
            demo.parent.mkdir(parents=True, exist_ok=True)
            demo.write_bytes(gif_path.read_bytes())
            print(f"wrote {demo}")

    return solver, hist, gif_path


if __name__ == "__main__":
    main()
