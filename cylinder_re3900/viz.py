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


def save_wake_gif(
    frame_paths: list[str | Path],
    out_path: str | Path,
    fps: int = 8,
    max_width: int = 900,
) -> Path:
    """Assemble mid-plane wake PNGs into an animated GIF.

    Uses a single shared palette so frames stay consistent (per-frame
    adaptive palettes otherwise break the animation).
    """
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
    """Sorted ``wake_t*.png`` paths in an output directory."""
    out_dir = Path(out_dir)
    return sorted(out_dir.glob("wake_t*.png"))
