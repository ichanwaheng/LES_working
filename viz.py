


"""Visualization helpers (matplotlib) + GIF animation of membrane flutter."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


def plot_membrane_and_slice(
    nodes: np.ndarray,
    elements: np.ndarray,
    fluid_u: np.ndarray,
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    j_slice: int,
    out_path: str | Path,
    title: str = "Membrane FSI",
) -> None:
    """Save a 2D plot: membrane side-view + mid-plane |U| contour."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    speed = np.sqrt(
        fluid_u[:, j_slice, :] ** 2
        + 0.0  # v not shown
    )
    # full speed if only u passed — recompute properly if 3 components unavailable
    # here fluid_u is u-component; use abs(u) as proxy for streamwise field
    field = np.abs(fluid_u[:, j_slice, :])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    Xg, Zg = np.meshgrid(grid_x, grid_z, indexing="ij")
    cf = ax.contourf(Xg, Zg, field, levels=24, cmap="YlOrBr")
    fig.colorbar(cf, ax=ax, label="|u| [m/s]")

    # membrane projected to xz (average y or all nodes)
    tri = Triangulation(nodes[:, 0], nodes[:, 2], elements)
    ax.triplot(tri, color="#1a1a1a", lw=0.6, alpha=0.85)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_history(history, out_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    t = history.time
    axes[0, 0].plot(t, history.max_disp, color="#0b3d2e")
    axes[0, 0].set_ylabel("max |u| [m]")
    axes[0, 0].set_title("Membrane displacement")
    axes[0, 1].plot(t, history.kinetic, color="#8b3a2a")
    axes[0, 1].set_ylabel("KE [J]")
    axes[0, 1].set_title("Kinetic energy")
    axes[1, 0].plot(t, history.cfl, color="#1f4e79")
    axes[1, 0].set_ylabel("CFL")
    axes[1, 0].set_title("Fluid CFL")
    axes[1, 1].plot(t, history.residual, color="#5c4a1f")
    axes[1, 1].set_ylabel("residual")
    axes[1, 1].set_title("FSI sub-iter residual")
    for ax in axes.ravel():
        ax.set_xlabel("t [s]")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def render_flutter_frame(
    nodes: np.ndarray,
    elements: np.ndarray,
    nodes0: np.ndarray,
    speed_slice: np.ndarray,
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    time: float,
    mesh_nx: int,
    mesh_ny: int,
    speed_max: float,
    disp_max: float,
    z_limits: tuple,
    title: str | None = None,
):
    """Render one animation frame → RGB PIL image.

    Layout: top panel is the 3D membrane surface coloured by vertical
    displacement; bottom panel is the mid-plane fluid speed slice (x–z)
    with the membrane side profile overlaid.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image

    fig = plt.figure(figsize=(7.2, 6.5), dpi=92)
    ax3d = fig.add_subplot(2, 1, 1, projection="3d")
    ax2d = fig.add_subplot(2, 1, 2)

    # --- 3D membrane coloured by vertical displacement -------------------
    dz = nodes[:, 2] - nodes0[:, 2]
    tris = nodes[elements]
    face_dz = dz[elements].mean(axis=1)
    norm = plt.Normalize(vmin=-disp_max, vmax=disp_max)
    colors = cm.coolwarm(norm(face_dz))
    coll = Poly3DCollection(tris, facecolors=colors, edgecolor="#333333", linewidths=0.15)
    ax3d.add_collection3d(coll)
    ax3d.set_xlim(nodes0[:, 0].min() - 0.1, nodes0[:, 0].max() + 0.1)
    ax3d.set_ylim(nodes0[:, 1].min() - 0.1, nodes0[:, 1].max() + 0.1)
    ax3d.set_zlim(*z_limits)
    # Prevent matplotlib from squashing Z relative to the in-plane spans —
    # otherwise amplified deflection looks almost flat in the 3D view.
    xr = float(nodes0[:, 0].max() - nodes0[:, 0].min()) + 0.2
    yr = float(nodes0[:, 1].max() - nodes0[:, 1].min()) + 0.2
    zr = float(z_limits[1] - z_limits[0])
    ax3d.set_box_aspect((xr, yr, max(zr, 0.75 * xr)))
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.view_init(elev=18, azim=-55)
    label = title or "Tensile membrane flutter"
    ax3d.set_title(f"{label}   t = {time:5.2f} s", fontsize=12, pad=8)
    # Large on-axes time stamp so the end time is obvious in the GIF
    ax3d.text2D(
        0.02,
        0.95,
        f"t = {time:.2f} s",
        transform=ax3d.transAxes,
        fontsize=14,
        fontweight="bold",
        color="#0b2e3d",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#1f7a6b"),
    )
    mappable = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    fig.colorbar(mappable, ax=ax3d, shrink=0.55, pad=0.08, label="Δz [m]")

    # --- fluid slice + membrane profile ----------------------------------
    Xg, Zg = np.meshgrid(grid_x, grid_z, indexing="ij")
    cf = ax2d.contourf(
        Xg,
        Zg,
        np.clip(speed_slice, 0.0, speed_max),
        levels=np.linspace(0.0, speed_max, 25),
        cmap="viridis",
    )
    fig.colorbar(cf, ax=ax2d, label="|u| [m/s]")

    # membrane side profiles: one polyline per spanwise node row
    n_row = mesh_ny + 1
    for j in range(n_row):
        row = nodes[j::n_row]  # structured grid: vid(i, j) = i*(ny+1)+j
        lw, alpha = (2.4, 1.0) if j == n_row // 2 else (1.0, 0.4)
        ax2d.plot(row[:, 0], row[:, 2], color="white", lw=lw, alpha=alpha)
    # flat reference line through the supports
    z_ref = float(np.median(nodes0[:, 2]))
    ax2d.axhline(z_ref, color="#ffdd57", lw=1.0, ls="--", alpha=0.9)

    ax2d.set_xlabel("x [m]")
    ax2d.set_ylabel("z [m]")
    ax2d.set_title("Mid-plane fluid speed + membrane profile")
    # Zoom vertically around the (possibly amplified) membrane so the
    # deflection is obvious; keep x over the full channel.
    ax2d.set_ylim(z_limits[0], z_limits[1])
    ax2d.set_aspect("auto")

    fig.tight_layout()
    fig.canvas.draw()
    img = Image.frombuffer(
        "RGBA",
        fig.canvas.get_width_height(),
        fig.canvas.buffer_rgba(),
    ).convert("RGB")
    plt.close(fig)
    return img


def save_gif(
    frames: Sequence,
    out_path: str | Path,
    fps: int = 12,
    duration_ms: int | None = None,
) -> Path:
    """Write a list of PIL images to an animated GIF.

    Parameters
    ----------
    fps :
        Frames per second (used when ``duration_ms`` is None).
    duration_ms :
        Explicit per-frame delay in milliseconds. Overrides ``fps`` when set
        (e.g. physical Δt × 1000 for real-time playback).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("no frames to write")
    frames = list(frames)
    delay = int(duration_ms) if duration_ms is not None else int(1000 / max(fps, 1))
    delay = max(delay, 20)  # many viewers ignore <20 ms
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=delay,
        loop=0,
        optimize=True,
    )
    return out_path


def render_deformation_frame(
    nodes: np.ndarray,
    elements: np.ndarray,
    nodes0: np.ndarray,
    time: float,
    disp_max: float,
    z_limits: tuple[float, float],
    title: str = "Membrane deformation",
):
    """Render one 3D membrane frame coloured by displacement magnitude → PIL RGB."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image

    nodes = np.asarray(nodes, dtype=float)
    nodes0 = np.asarray(nodes0, dtype=float)
    elements = np.asarray(elements, dtype=int)
    disp = nodes - nodes0
    mag = np.linalg.norm(disp, axis=1)
    face_mag = mag[elements].mean(axis=1)

    fig = plt.figure(figsize=(7.0, 5.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    tris = nodes[elements]
    vmax = max(float(disp_max), 1e-12)
    norm = plt.Normalize(vmin=0.0, vmax=vmax)
    colors = cm.cividis(norm(face_mag))
    coll = Poly3DCollection(
        tris, facecolors=colors, edgecolor="#333333", linewidths=0.15, alpha=0.95
    )
    ax.add_collection3d(coll)

    pad = 0.05 * max(
        float(nodes0[:, 0].max() - nodes0[:, 0].min()),
        float(nodes0[:, 1].max() - nodes0[:, 1].min()),
        1e-6,
    )
    ax.set_xlim(nodes0[:, 0].min() - pad, nodes0[:, 0].max() + pad)
    ax.set_ylim(nodes0[:, 1].min() - pad, nodes0[:, 1].max() + pad)
    ax.set_zlim(*z_limits)
    xr = float(nodes0[:, 0].max() - nodes0[:, 0].min()) + 2.0 * pad
    yr = float(nodes0[:, 1].max() - nodes0[:, 1].min()) + 2.0 * pad
    zr = float(z_limits[1] - z_limits[0])
    ax.set_box_aspect((xr, yr, max(zr, 0.45 * xr)))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=22, azim=-60)
    ax.set_title(f"{title}   t = {time:.3g}", fontsize=12, pad=8)
    ax.text2D(
        0.02,
        0.95,
        f"t = {time:.3g}\nmax|u| = {float(mag.max()):.3e} m",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color="#0b2e3d",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            alpha=0.85,
            edgecolor="#1f7a6b",
        ),
    )
    mappable = cm.ScalarMappable(norm=norm, cmap="cividis")
    fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08, label="|u| [m]")

    fig.tight_layout()
    fig.canvas.draw()
    img = Image.frombuffer(
        "RGBA",
        fig.canvas.get_width_height(),
        fig.canvas.buffer_rgba(),
    ).convert("RGB")
    plt.close(fig)
    return img


def save_deformation_gif(
    nodes_over_time: Sequence[np.ndarray],
    elements: np.ndarray,
    nodes0: np.ndarray,
    out_path: str | Path,
    times: Optional[Sequence[float]] = None,
    fps: int = 8,
    amplify: float = 1.0,
    title: str = "Membrane deformation",
) -> Path:
    """Build an animated GIF of membrane deformation from recorded node sets.

    Parameters
    ----------
    nodes_over_time :
        Sequence of ``(n_nodes, 3)`` arrays (one per frame).
    elements :
        Triangle connectivity ``(n_elements, 3)``.
    nodes0 :
        Reference / undeformed node coordinates.
    amplify :
        Visual scale factor applied to displacement only (geometry = nodes0 +
        amplify * (nodes - nodes0)). Use ``1.0`` for physical motion.
    """
    if not nodes_over_time:
        raise ValueError("nodes_over_time is empty — nothing to animate")

    nodes0 = np.asarray(nodes0, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_frames = len(nodes_over_time)
    if times is None:
        times = list(range(n_frames))
    if len(times) != n_frames:
        raise ValueError("times and nodes_over_time must have the same length")

    # Fixed colour / z scales across the whole GIF
    disp_max = 0.0
    z_vals: List[float] = [float(nodes0[:, 2].min()), float(nodes0[:, 2].max())]
    amp = float(amplify)
    shaped: List[np.ndarray] = []
    for nodes in nodes_over_time:
        nodes = np.asarray(nodes, dtype=float)
        disp = nodes - nodes0
        shaped_nodes = nodes0 + amp * disp
        shaped.append(shaped_nodes)
        disp_max = max(disp_max, float(np.linalg.norm(disp, axis=1).max()))
        z_vals.append(float(shaped_nodes[:, 2].min()))
        z_vals.append(float(shaped_nodes[:, 2].max()))

    z_min, z_max = min(z_vals), max(z_vals)
    z_pad = max(0.05 * max(z_max - z_min, disp_max, 1e-6), 1e-4)
    z_limits = (z_min - z_pad, z_max + z_pad)
    if disp_max <= 0.0:
        disp_max = 1e-6

    frames = [
        render_deformation_frame(
            nodes=shaped[i],
            elements=elements,
            nodes0=nodes0,
            time=float(times[i]),
            disp_max=disp_max * max(amp, 1.0),
            z_limits=z_limits,
            title=title,
        )
        for i in range(n_frames)
    ]
    return save_gif(frames, out_path, fps=fps)


def plot_membrane_3d(
    nodes: np.ndarray,
    elements: np.ndarray,
    out_path: str | Path,
    displacement: Optional[np.ndarray] = None,
    title: str = "Tensile membrane",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    tris = nodes[elements]
    coll = Poly3DCollection(tris, alpha=0.85, edgecolor="#222222", linewidths=0.2)
    if displacement is not None:
        mag = np.linalg.norm(displacement, axis=1)
        # color by average nodal magnitude per triangle
        face_mag = mag[elements].mean(axis=1)
        coll.set_array(face_mag)
        coll.set_cmap("cividis")
    else:
        coll.set_facecolor("#c4a574")
    ax.add_collection3d(coll)
    ax.auto_scale_xyz(nodes[:, 0], nodes[:, 1], nodes[:, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

