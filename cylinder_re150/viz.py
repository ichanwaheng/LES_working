"""Wake plots and force history for the laminar cylinder case."""

from __future__ import annotations

from pathlib import Path

import numpy as np


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
    dudy = np.gradient(uc, grid.dy, axis=1, edge_order=1)
    dvdx = np.gradient(vc, grid.dx, axis=0, edge_order=1)
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
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(times, Cd, label="Cd", color="#8b3a2a")
    ax.plot(times, Cl, label="Cl", color="#1f4e79")
    if Cd_ref is not None:
        ax.axhline(Cd_ref, color="#8b3a2a", ls="--", lw=1.0, alpha=0.7, label=f"Cd ref={Cd_ref}")
    ax.set_xlabel("t U / D")
    ax.set_ylabel("force coefficient")
    title = "Laminar cylinder force history (Re=150)"
    if St_ref is not None:
        title += f"  |  St_ref≈{St_ref}"
    ax.set_title(title)
    ax.legend(loc="best")
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
