"""Lumped-mass membrane dynamics on a frozen force-density network.

The form-found edge weights define a linear network stiffness. Free nodes
obey::

    m a = f_network(x) + f_ext - c v

with supports held fixed. This yields continuous motion once the membrane
is submerged in a time-varying fluid load.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from UWM import EdgeKey, compute_edge_weights


def nodal_tributary_mass(
    nodes: np.ndarray,
    elements: np.ndarray,
    density: float,
    thickness: float,
    mass_scale: float = 1.0,
) -> np.ndarray:
    """Lumped mass per node from element areas (density × thickness × area/3)."""
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    mass = np.zeros(nodes.shape[0], dtype=float)
    scale = float(density) * float(thickness) * float(mass_scale)
    for conn in elements:
        p0, p1, p2 = nodes[conn[0]], nodes[conn[1]], nodes[conn[2]]
        area = 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
        share = scale * area / 3.0
        mass[conn[0]] += share
        mass[conn[1]] += share
        mass[conn[2]] += share
    # Avoid singular free-node masses on degenerate patches
    mass = np.maximum(mass, 1e-9 * scale)
    return mass


def network_forces(
    nodes: np.ndarray,
    weights: Dict[EdgeKey, float],
) -> np.ndarray:
    """Internal force-density network forces on every node.

    For edge weight ``W`` between ``i`` and ``j``::

        F_i += 2 W (x_j - x_i)
        F_j += 2 W (x_i - x_j)

    Equilibrium of free nodes satisfies ``F_network + f_ext = 0``.
    """
    f = np.zeros_like(nodes, dtype=float)
    x = np.asarray(nodes, dtype=float)
    for (i, j), W in weights.items():
        coef = 2.0 * float(W)
        d = x[j] - x[i]
        f[i] += coef * d
        f[j] -= coef * d
    return f


def freeze_form_weights(
    nodes: np.ndarray,
    elements: np.ndarray,
    N_pre: float,
) -> Dict[EdgeKey, float]:
    """Edge weights from the form-found geometry (held constant afterward)."""
    return compute_edge_weights(
        np.asarray(nodes, dtype=float),
        np.asarray(elements, dtype=int),
        float(N_pre),
    )


def advance_structure(
    nodes: np.ndarray,
    velocities: np.ndarray,
    *,
    fixed: np.ndarray,
    x_bc: np.ndarray,
    weights: Dict[EdgeKey, float],
    f_ext: np.ndarray,
    mass: np.ndarray,
    damping: float,
    dt: float,
    n_substeps: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Semi-implicit Euler: v ← v + dt a, x ← x + dt v (supports clamped)."""
    x = np.asarray(nodes, dtype=float).copy()
    v = np.asarray(velocities, dtype=float).copy()
    fixed = np.asarray(fixed, dtype=bool)
    x_bc = np.asarray(x_bc, dtype=float)
    f_ext = np.asarray(f_ext, dtype=float)
    mass = np.asarray(mass, dtype=float)
    h = float(dt) / max(int(n_substeps), 1)
    c = float(damping)

    for _ in range(max(int(n_substeps), 1)):
        f_net = network_forces(x, weights)
        acc = (f_net + f_ext - c * v) / mass[:, None]
        acc[fixed] = 0.0
        v = v + h * acc
        v[fixed] = 0.0
        x = x + h * v
        x[fixed] = x_bc[fixed]
        v[fixed] = 0.0

    return x, v
