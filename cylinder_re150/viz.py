"""Wake plots, mesh geometry, and force history for the laminar cylinder case."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_mesh_geometry(
    grid,
    solid_mask: np.ndarray,
    out_path: str | Path,
    cylinder=None,
    title: str = "Mesh geometry (pre-analysis)",
    max_lines: int = 400,
    zoom: bool = True,
) -> Path:
    """Draw the meshed domain with cylinder / solid cells before analysis.

    Writes a full-domain view and (if ``zoom``) a close-up around the cylinder
    so wall clustering is visible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_f = np.asarray(grid.x_u, dtype=float)
    y_f = np.asarray(grid.y_v, dtype=float)
    nx, ny = grid.nx, grid.ny

    # Subsample line drawing on very fine meshes for readable PNGs
    stride_x = max(1, int(np.ceil(nx / max_lines)))
    stride_y = max(1, int(np.ceil(ny / max_lines)))

    def _grid_segments():
        segs = []
        # vertical faces (constant x)
        for i in range(0, nx + 1, stride_x):
            segs.append([(x_f[i], y_f[0]), (x_f[i], y_f[-1])])
        if (nx % stride_x) != 0:
            segs.append([(x_f[-1], y_f[0]), (x_f[-1], y_f[-1])])
        # horizontal faces (constant y)
        for j in range(0, ny + 1, stride_y):
            segs.append([(x_f[0], y_f[j]), (x_f[-1], y_f[j])])
        if (ny % stride_y) != 0:
            segs.append([(x_f[0], y_f[-1]), (x_f[-1], y_f[-1])])
        return segs

    def _draw(ax, xlim=None, ylim=None, show_every_line: bool = False):
        if show_every_line and xlim is not None and ylim is not None:
            # In zoom window, draw all lines that intersect the box
            segs = []
            for i in range(nx + 1):
                if xlim[0] <= x_f[i] <= xlim[1]:
                    segs.append([(x_f[i], max(y_f[0], ylim[0])), (x_f[i], min(y_f[-1], ylim[1]))])
            for j in range(ny + 1):
                if ylim[0] <= y_f[j] <= ylim[1]:
                    segs.append([(max(x_f[0], xlim[0]), y_f[j]), (min(x_f[-1], xlim[1]), y_f[j])])
        else:
            segs = _grid_segments()
        ax.add_collection(
            LineCollection(segs, colors="#6b7c8a", linewidths=0.35, alpha=0.85)
        )
        # Solid IB cells
        if solid_mask is not None and np.any(solid_mask):
            ax.scatter(
                grid.X[solid_mask],
                grid.Y[solid_mask],
                s=6,
                c="#c45c26",
                marker="s",
                linewidths=0,
                label="solid cells",
                zorder=3,
            )
        if cylinder is not None:
            theta = np.linspace(0, 2 * np.pi, 256)
            ax.plot(
                cylinder.cx + cylinder.R * np.cos(theta),
                cylinder.cy + cylinder.R * np.sin(theta),
                color="k",
                lw=1.8,
                label="cylinder",
                zorder=4,
            )
            ax.plot(cylinder.cx, cylinder.cy, "k+", ms=8, zorder=4)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x / D")
        ax.set_ylabel("y / D")
        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(x_f[0], x_f[-1])
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(y_f[0], y_f[-1])
        ax.legend(loc="upper right", fontsize=8)

    n_panels = 2 if zoom and cylinder is not None else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=(7 * n_panels, 6.2), squeeze=False
    )
    ax0 = axes[0, 0]
    _draw(ax0)
    ax0.set_title(f"{title}\n{nx}×{ny} cells  (lines every {stride_x}/{stride_y})")

    if n_panels == 2:
        ax1 = axes[0, 1]
        pad = 1.5 * cylinder.D
        xlim = (cylinder.cx - pad, cylinder.cx + pad)
        ylim = (cylinder.cy - pad, cylinder.cy + pad)
        _draw(ax1, xlim=xlim, ylim=ylim, show_every_line=True)
        ax1.set_title("Zoom near cylinder (all mesh lines)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_mesh_spacing(
    grid,
    out_path: str | Path,
    cx: float | None = None,
    cy: float | None = None,
    R: float | None = None,
) -> Path:
    """Plot Δx(x) and Δy(y) after BL meshing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    axes[0].plot(grid.x, grid.dx_c, color="#1f4e79", lw=1.2)
    axes[0].set_ylabel("Δx")
    axes[0].set_xlabel("x")
    axes[0].set_title("Streamwise cell size after meshing")
    axes[1].plot(grid.y, grid.dy_c, color="#8b3a2a", lw=1.2)
    axes[1].set_ylabel("Δy")
    axes[1].set_xlabel("y")
    axes[1].set_title("Cross-stream cell size after meshing")
    if cx is not None and R is not None:
        axes[0].axvline(cx - R, color="k", ls="--", lw=0.8)
        axes[0].axvline(cx + R, color="k", ls="--", lw=0.8)
    if cy is not None and R is not None:
        axes[1].axvline(cy - R, color="k", ls="--", lw=0.8)
        axes[1].axvline(cy + R, color="k", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_wake(
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

    uc, vc = state.cell_centered_velocity()
    speed = np.sqrt(uc * uc + vc * vc)
    # Variable-spacing gradients along cell-centre coordinates
    dudy = np.gradient(uc, grid.y, axis=1, edge_order=1)
    dvdx = np.gradient(vc, grid.x, axis=0, edge_order=1)
    omega = dvdx - dudy
    solid = solid_mask
    speed = np.ma.array(speed, mask=solid)
    omega = np.ma.array(omega, mask=solid)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    Xg, Yg = np.meshgrid(grid.x, grid.y, indexing="ij")

    cf0 = axes[0].contourf(Xg, Yg, speed, levels=28, cmap="YlOrBr")
    fig.colorbar(cf0, ax=axes[0], label="|u|")
    axes[0].set_ylabel("y / D")
    axes[0].set_title(f"{title} — speed")
    axes[0].set_aspect("equal", adjustable="box")

    wmax = float(np.nanpercentile(np.abs(omega.compressed()), 99)) if omega.count() else 1.0
    wmax = max(wmax, 1e-6)
    cf1 = axes[1].contourf(
        Xg, Yg, omega, levels=28, cmap="RdBu_r", vmin=-wmax, vmax=wmax
    )
    fig.colorbar(cf1, ax=axes[1], label="ω_z")
    axes[1].set_xlabel("x / D")
    axes[1].set_ylabel("y / D")
    axes[1].set_title(f"{title} — vorticity")
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
    Cd_ref: float | None = None,
    St_ref: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    times = np.asarray(times, dtype=float)
    Cd = np.asarray(Cd, dtype=float)
    Cl = np.asarray(Cl, dtype=float)

    # Skip the impulsive start so the shedding signal is visible
    t_cut = 2.0
    mask = times >= t_cut if times.size and times[-1] > t_cut + 1.0 else np.ones_like(times, dtype=bool)
    t_p, Cd_p, Cl_p = times[mask], Cd[mask], Cl[mask]

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    axes[0].plot(t_p, Cd_p, label="Cd", color="#8b3a2a")
    if Cd_ref is not None:
        axes[0].axhline(
            Cd_ref, color="#8b3a2a", ls="--", lw=1.0, alpha=0.7, label=f"Cd ref={Cd_ref}"
        )
    axes[0].set_ylabel("Cd")
    axes[0].legend(loc="best")
    title = "Laminar cylinder force history (Re=150)"
    if St_ref is not None:
        title += f"  |  St_ref≈{St_ref:g}"
    axes[0].set_title(title)

    axes[1].plot(t_p, Cl_p, label="Cl", color="#1f4e79")
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_xlabel("t U / D")
    axes[1].set_ylabel("Cl")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_wake_gif(
    frame_paths: list[str | Path],
    out_path: str | Path,
    fps: int = 8,
    max_width: int = 900,
) -> Path:
    """Assemble wake PNGs into an animated GIF with a shared palette."""
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


def wake_frames_from_dir(out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    return sorted(out_dir.glob("wake_t*.png"))
