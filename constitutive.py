#!/usr/bin/env python
# coding: utf-8

"""Green–Lagrange membrane kinematics + constitutive stress/strain energy.

Each triangular facet is a constant-strain total-Lagrangian membrane
element. Strain is measured from a reference (typically flat) geometry;
stress follows ``MembraneMaterial`` plane-stress elasticity with prestress.

Potential energy of the membrane::

    Π = Σₑ t A₀ W(ε)  −  Σᵢ f_extᵢ · uᵢ

with ``W = ½ ε·D·ε + σ₀·ε``. Equilibrium is ``∂Π/∂x = 0`` on free nodes
(internal constitutive forces balance external loads).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from materials import MembraneMaterial


@dataclass
class ElementState:
    """Constitutive state on one triangle."""

    strain_voigt: np.ndarray  # [E11, E22, 2E12]
    stress_voigt: np.ndarray  # [S11, S22, S12]
    energy_density: float
    area0: float
    energy: float  # t * A0 * W


def _local_material_frame(
    X0: np.ndarray, X1: np.ndarray, X2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Orthonormal in-plane basis and reference area from undeformed triangle."""
    A1 = X1 - X0
    A2 = X2 - X0
    nrm = np.cross(A1, A2)
    area0 = 0.5 * float(np.linalg.norm(nrm))
    if area0 < 1e-16:
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        return e1, e2, 0.0
    e1 = A1 / (np.linalg.norm(A1) + 1e-16)
    e2 = A2 - np.dot(A2, e1) * e1
    e2 = e2 / (np.linalg.norm(e2) + 1e-16)
    return e1, e2, area0


def reference_2d_coords(
    X0: np.ndarray, X1: np.ndarray, X2: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Map reference triangle to 2D material coordinates (origin at X0)."""
    e1, e2, area0 = _local_material_frame(X0, X1, X2)
    Y = np.zeros((3, 2), dtype=float)
    for a, X in enumerate((X0, X1, X2)):
        d = X - X0
        Y[a, 0] = float(np.dot(d, e1))
        Y[a, 1] = float(np.dot(d, e2))
    return Y, area0


def deformation_gradient(
    x0: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``F`` (3×2) and material shape-function gradients ``∇N`` (3×2)."""
    J = np.column_stack((Y[1] - Y[0], Y[2] - Y[0]))  # 2×2
    det = float(np.linalg.det(J))
    if abs(det) < 1e-18:
        F = np.zeros((3, 2), dtype=float)
        gradN = np.zeros((3, 2), dtype=float)
        return F, gradN
    Jinv = np.linalg.inv(J)
    a1 = x1 - x0
    a2 = x2 - x0
    F = np.column_stack((a1, a2)) @ Jinv  # 3×2

    # N0=1-ξ-η, N1=ξ, N2=η; [ξ,η]^T = Jinv (Y-Y0)
    gradN = np.zeros((3, 2), dtype=float)
    gradN[1] = Jinv.T @ np.array([1.0, 0.0])
    gradN[2] = Jinv.T @ np.array([0.0, 1.0])
    gradN[0] = -gradN[1] - gradN[2]
    return F, gradN


def green_lagrange_voigt(F: np.ndarray) -> np.ndarray:
    """Green–Lagrange strain in Voigt form ``[E₁₁, E₂₂, 2 E₁₂]``."""
    C = F.T @ F
    E11 = 0.5 * (C[0, 0] - 1.0)
    E22 = 0.5 * (C[1, 1] - 1.0)
    E12 = 0.5 * C[0, 1]
    return np.array([E11, E22, 2.0 * E12], dtype=float)


def element_constitutive_state(
    x_nodes: np.ndarray,
    X_nodes: np.ndarray,
    material: MembraneMaterial,
) -> Tuple[ElementState, np.ndarray, np.ndarray]:
    """Strain, stress, energy and ``(F, ∇N)`` for one triangle.

    ``x_nodes``, ``X_nodes`` are ``(3, 3)`` current / reference coordinates.
    """
    Y, area0 = reference_2d_coords(X_nodes[0], X_nodes[1], X_nodes[2])
    F, gradN = deformation_gradient(
        x_nodes[0], x_nodes[1], x_nodes[2], Y
    )
    eps = green_lagrange_voigt(F)
    sig = material.stress(eps)
    W = material.strain_energy_density(eps)
    t = float(material.thickness)
    energy = W * t * area0
    state = ElementState(
        strain_voigt=eps,
        stress_voigt=sig,
        energy_density=W,
        area0=area0,
        energy=energy,
    )
    return state, F, gradN


def element_internal_forces(
    x_nodes: np.ndarray,
    X_nodes: np.ndarray,
    material: MembraneMaterial,
) -> Tuple[np.ndarray, ElementState]:
    """Nodal internal forces ``(3, 3)`` from constitutive stress (total Lagrangian)."""
    state, F, gradN = element_constitutive_state(x_nodes, X_nodes, material)
    S = material.pk2_matrix(state.strain_voigt)  # 2×2
    P = F @ S  # first PK, 3×2
    tA = float(material.thickness) * state.area0
    f = np.zeros((3, 3), dtype=float)
    for a in range(3):
        f[a] = tA * (P @ gradN[a])
    return f, state


def assemble_internal_forces(
    nodes: np.ndarray,
    elements: np.ndarray,
    nodes0: np.ndarray,
    material: MembraneMaterial,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Assemble nodal internal forces and total internal potential.

    Returns
    -------
    f_int, Pi_int, strain_elem, stress_elem
        ``strain_elem`` / ``stress_elem`` are ``(n_elem, 3)`` Voigt arrays.
    """
    nodes = np.asarray(nodes, dtype=float)
    nodes0 = np.asarray(nodes0, dtype=float)
    elements = np.asarray(elements, dtype=int)
    f_int = np.zeros_like(nodes)
    Pi = 0.0
    n_e = elements.shape[0]
    strain_elem = np.zeros((n_e, 3), dtype=float)
    stress_elem = np.zeros((n_e, 3), dtype=float)

    for e, conn in enumerate(elements):
        f_e, state = element_internal_forces(
            nodes[conn], nodes0[conn], material
        )
        f_int[conn] += f_e
        Pi += state.energy
        strain_elem[e] = state.strain_voigt
        stress_elem[e] = state.stress_voigt

    return f_int, float(Pi), strain_elem, stress_elem


def total_potential(
    nodes: np.ndarray,
    elements: np.ndarray,
    nodes0: np.ndarray,
    material: MembraneMaterial,
    f_ext: Optional[np.ndarray] = None,
    x_ref_work: Optional[np.ndarray] = None,
) -> float:
    """Total potential ``Π_int − f_ext · (x − x_ref)``."""
    _, Pi_int, _, _ = assemble_internal_forces(
        nodes, elements, nodes0, material
    )
    if f_ext is None:
        return Pi_int
    x_ref = nodes0 if x_ref_work is None else np.asarray(x_ref_work, dtype=float)
    u = np.asarray(nodes, dtype=float) - x_ref
    return float(Pi_int - np.sum(np.asarray(f_ext, dtype=float) * u))
