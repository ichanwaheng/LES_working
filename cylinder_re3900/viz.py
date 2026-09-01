"""Mid-plane plots for the cylinder wake."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_midplane(
    grid,
    state,
    solid_mask: np.ndarray,
    out_path: str | Path,
    title: str = "Cylinder wake",
    cylinder=None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    k = grid.nz // 2
    uc, vc, _wc = state.cell_centered_velocity()
    u = uc[:, :, k]
    v = vc[:, :, k]
    speed = np.sqrt(u * u + v * v)
    # Spanwise vorticity ω_z ≈ dv/dx - du/dy
    dudy = np.gradient(u, grid.dy, axis=1, edge_order=1)
    dvdx = np.gradient(v, grid.dx, axis=0, edge_order=1)
    omega = dvdx - dudy
    solid = solid_mask[:, :, k]
    speed = np.ma.array(speed, mask=solid)
    omega = np.ma.array(omega, mask=solid)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    Xg, Yg = np.meshgrid(grid.x, grid.y, indexing="ij")

    cf0 = axes[0].contourf(Xg, Yg, speed, levels=28, cmap="YlOrBr")
    fig.colorbar(cf0, ax=axes[0], label="|u_xy|")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"{title} — speed")
    axes[0].set_aspect("equal", adjustable="box")

    wmax = float(np.nanpercentile(np.abs(omega.compressed()), 99)) if omega.count() else 1.0
    wmax = max(wmax, 1e-6)
    cf1 = axes[1].contourf(
        Xg, Yg, omega, levels=28, cmap="RdBu_r", vmin=-wmax, vmax=wmax
    )
    fig.colorbar(cf1, ax=axes[1], label="ω_z")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title(f"{title} — spanwise vorticity")
    axes[1].set_aspect("equal", adjustable="box")

    if cylinder is not None:
        theta = np.linspace(0, 2 * np.pi, 120)
        xc = cylinder.cx + cylinder.R * np.cos(theta)
        yc = cylinder.cy + cylinder.R * np.sin(theta)
        for ax in axes:
            ax.plot(xc, yc, color="k", lw=1.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_force_history(
    times: np.ndarray,
    Cd: np.ndarray,
    Cl: np.ndarray,
    out_path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(times, Cd, label="Cd", color="#8b3a2a")
    ax.plot(times, Cl, label="Cl", color="#1f4e79")
    ax.set_xlabel("t U / D")
    ax.set_ylabel("force coefficient")
    ax.legend()
    ax.set_title("Cylinder force history")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
