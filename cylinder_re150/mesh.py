"""MAC staggered grid (2-D) with optional boundary-layer stretching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from mesh_sizing import (
    BoundaryLayerSizing,
    build_cylinder_domain_spacings,
    format_sizing_report,
)


def _centres_and_faces(widths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cell widths → cell-centre coords and face coords (length n+1)."""
    w = np.asarray(widths, dtype=float)
    faces = np.concatenate([[0.0], np.cumsum(w)])
    centres = 0.5 * (faces[:-1] + faces[1:])
    return centres, faces


@dataclass
class FluidGrid:
    """MAC staggered grid on a box ``[0, L] × [0, W]``.

    Pressure / scalars live at cell centres ``(nx, ny)``.
    ``u`` lives on vertical faces ``(nx+1, ny)``.
    ``v`` lives on horizontal faces ``(nx, ny+1)``.

    Cell widths may be **non-uniform** (BL packing near walls). Scalar
    attributes ``dx`` / ``dy`` are the *minimum* cell sizes (useful for CFL);
    full arrays are ``dx_c`` / ``dy_c``.
    """

    L: float
    W: float
    nx: int
    ny: int
    dx_widths: Optional[Sequence[float]] = None
    dy_widths: Optional[Sequence[float]] = None
    mesh_info: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.dx_widths is None:
            w = np.full(self.nx, self.L / self.nx, dtype=float)
        else:
            w = np.asarray(self.dx_widths, dtype=float)
            if w.size != self.nx:
                raise ValueError(f"dx_widths length {w.size} != nx {self.nx}")
        if self.dy_widths is None:
            h = np.full(self.ny, self.W / self.ny, dtype=float)
        else:
            h = np.asarray(self.dy_widths, dtype=float)
            if h.size != self.ny:
                raise ValueError(f"dy_widths length {h.size} != ny {self.ny}")

        # Exact domain fit
        w = w * (self.L / max(float(w.sum()), 1e-30))
        h = h * (self.W / max(float(h.sum()), 1e-30))
        self.dx_c = w
        self.dy_c = h
        self.dx = float(np.min(w))
        self.dy = float(np.min(h))

        self.x, self.x_u = _centres_and_faces(w)
        self.y, self.y_v = _centres_and_faces(h)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # Spacing between neighbouring cell centres (for gradients)
        self.dx_p = np.diff(self.x)  # length nx-1
        self.dy_p = np.diff(self.y)  # length ny-1

        self.n_cells = self.nx * self.ny
        self.volume = float(np.mean(w) * np.mean(h))

    @classmethod
    def from_boundary_layer(
        cls,
        L: float,
        W: float,
        cx: float,
        cy: float,
        D: float,
        Re: float,
        n_bl: int,
        growth: float,
        h_far: float | None = None,
        verbose: bool = True,
    ) -> "FluidGrid":
        """Build a grid from BL inputs (no prescribed nx, ny)."""
        sizing = BoundaryLayerSizing(D=D, Re=Re, n_bl=n_bl, growth=growth)
        R = 0.5 * D
        dx, dy, info = build_cylinder_domain_spacings(
            L, W, cx, cy, R, sizing, h_far=h_far
        )
        if verbose:
            print(format_sizing_report(info))
        return cls(
            L=L,
            W=W,
            nx=int(dx.size),
            ny=int(dy.size),
            dx_widths=dx,
            dy_widths=dy,
            mesh_info=info,
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return self.nx, self.ny

    @property
    def shape_u(self) -> Tuple[int, int]:
        return self.nx + 1, self.ny

    @property
    def shape_v(self) -> Tuple[int, int]:
        return self.nx, self.ny + 1

    def delta(self) -> float:
        return (self.dx * self.dy) ** 0.5

    def cell_area(self) -> np.ndarray:
        """Per-cell area ``dx_c[i] * dy_c[j]``, shape ``(nx, ny)``."""
        return np.outer(self.dx_c, self.dy_c)
