#!/usr/bin/env python
# coding: utf-8

"""Green–Lagrange membrane kinematics with ``F = I + ∇u``.

Continuum (plane stress)::

    u  = x - X                         # displacement field (ux, uy, uz)
    F  = I + ∇u                        # deformation gradient (membrane: 3×2)
    ε  = ½ (Fᵀ F - I)                  # Green–Lagrange (I is 2×2 in-plane)
    σ  = D ε + σ₀                      # plane-stress constitutive relation

Discrete CST triangle: ``∇u = Σ_a u_a ⊗ ∇N_a`` in the material chart ``Y``,
and the reference embedding satisfies ``∂X/∂Y ≡ I_surf``, so
``F = I_surf + ∇u = ∂x/∂Y``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from materials import MembraneMaterial, green_lagrange_from_F


@dataclass
class ElementState:
    """Constitutive state on one triangle."""

    strain_voigt: np.ndarray  # [E11, E22, 2E12]
    stress_voigt: np.ndarray  # [S11, S22, S12]
    energy_density: float
    area0: float
    energy: float  # t * A0 * W
    F: np.ndarray  # 3×2
    grad_u: np.ndarray  # 3×2


def _local_material_frame(
    X0: np.ndarray, X1: np.ndarray, X2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Orthonormal in-plane basis and reference area."""
    A1 = X1 - X0
    A2 = X2 - X0
    nrm = np.cross(A1, A2)
    area0 = 0.5 * float(np.linalg.norm(nrm))
    if area0 < 1e-16:
        return (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            0.0,
        )
    e1 = A1 / (np.linalg.norm(A1) + 1e-16)
    e2 = A2 - np.dot(A2, e1) * e1
    e2 = e2 / (np.linalg.norm(e2) + 1e-16)
    return e1, e2, area0


def reference_2d_coords(
    X0: np.ndarray, X1: np.ndarray, X2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Material coords ``Y`` (3×2), surface identity ``I_surf`` (3×2), area."""
    e1, e2, area0 = _local_material_frame(X0, X1, X2)
    Y = np.zeros((3, 2), dtype=float)
    for a, X in enumerate((X0, X1, X2)):
        d = X - X0
        Y[a, 0] = float(np.dot(d, e1))
        Y[a, 1] = float(np.dot(d, e2))
    # ∂X/∂Y in the local frame where X = X0 + Y1 e1 + Y2 e2: the surface "I"
    I_surf = np.column_stack((e1, e2))  # 3×2
    return Y, I_surf, area0


def shape_gradients(Y: np.ndarray) -> np.ndarray:
    """CST material gradients ``∇N`` with shape ``(3_nodes, 2)``."""
    J = np.column_stack((Y[1] - Y[0], Y[2] - Y[0]))
    det = float(np.linalg.det(J))
    gradN = np.zeros((3, 2), dtype=float)
    if abs(det) < 1e-18:
        return gradN
    Jinv = np.linalg.inv(J)
    gradN[1] = Jinv.T @ np.array([1.0, 0.0])
    gradN[2] = Jinv.T @ np.array([0.0, 1.0])
    gradN[0] = -gradN[1] - gradN[2]
    return gradN


def grad_u_triangle(
    u_nodes: np.ndarray,
    gradN: np.ndarray,
) -> np.ndarray:
    """``∇u = Σ_a u_a ⊗ ∇N_a`` → ``(3, 2)`` for one triangle.

    ``u_nodes`` is ``(3, 3)`` with rows ``[ux, uy, uz]`` at the three vertices.
    """
    g = np.zeros((3, 2), dtype=float)
    for a in range(3):
        g += np.outer(u_nodes[a], gradN[a])
    return g


def deformation_gradient_from_u(
    u_nodes: np.ndarray,
    X_nodes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build ``F = I + ∇u`` on one triangle.

    Returns
    -------
    F, grad_u, gradN, area0
        ``F`` and ``grad_u`` are ``(3, 2)``; ``I`` is the surface embedding
        ``I_surf = ∂X/∂Y`` so that ``F = I_surf + ∇u``.
    """
    Y, I_surf, area0 = reference_2d_coords(
        X_nodes[0], X_nodes[1], X_nodes[2]
    )
    gradN = shape_gradients(Y)
    gu = grad_u_triangle(np.asarray(u_nodes, dtype=float), gradN)
    F = I_surf + gu  # F = I + grad(u)
    return F, gu, gradN, area0


def deformation_gradient(
    x0: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy helper: ``F = ∂x/∂Y`` (equivalent to ``I + ∇u``)."""
    J = np.column_stack((Y[1] - Y[0], Y[2] - Y[0]))
    det = float(np.linalg.det(J))
    if abs(det) < 1e-18:
        return np.zeros((3, 2)), np.zeros((3, 2))
    Jinv = np.linalg.inv(J)
    F = np.column_stack((x1 - x0, x2 - x0)) @ Jinv
    gradN = shape_gradients(Y)
    return F, gradN


def green_lagrange_voigt(F: np.ndarray) -> np.ndarray:
    """``ε = ½(Fᵀ F - I)`` in Voigt form ``[E₁₁, E₂₂, 2 E₁₂]``."""
    _E, eps = green_lagrange_from_F(F)
    return eps


def element_state_from_u(
    u_nodes: np.ndarray,
    X_nodes: np.ndarray,
    material: MembraneMaterial,
) -> Tuple[ElementState, np.ndarray]:
    """Strain/stress on one triangle from nodal displacement ``u = x - X``."""
    F, gu, gradN, area0 = deformation_gradient_from_u(u_nodes, X_nodes)
    eps = green_lagrange_voigt(F)
    sig = material.constitutive_relation(eps)  # σ = D ε + σ₀
    W = material.strain_energy_density(eps)
    energy = W * float(material.thickness) * area0
    state = ElementState(
        strain_voigt=eps,
        stress_voigt=sig,
        energy_density=W,
        area0=area0,
        energy=energy,
        F=F,
        grad_u=gu,
    )
    return state, gradN


def element_constitutive_state(
    x_nodes: np.ndarray,
    X_nodes: np.ndarray,
    material: MembraneMaterial,
) -> Tuple[ElementState, np.ndarray, np.ndarray]:
    """Same as :func:`element_state_from_u` with ``u = x - X``."""
    u_nodes = np.asarray(x_nodes, dtype=float) - np.asarray(X_nodes, dtype=float)
    state, gradN = element_state_from_u(u_nodes, X_nodes, material)
    return state, state.F, gradN


def element_internal_forces(
    x_nodes: np.ndarray,
    X_nodes: np.ndarray,
    material: MembraneMaterial,
) -> Tuple[np.ndarray, ElementState]:
    """Nodal internal forces from constitutive stress (total Lagrangian)."""
    state, F, gradN = element_constitutive_state(x_nodes, X_nodes, material)
    S = material.pk2_matrix(state.strain_voigt)
    P = F @ S
    tA = float(material.thickness) * state.area0
    f = np.zeros((3, 3), dtype=float)
    for a in range(3):
        f[a] = tA * (P @ gradN[a])
    return f, state


def stress_from_displacement_field(
    u: np.ndarray,
    nodes_ref: np.ndarray,
    elements: np.ndarray,
    material: MembraneMaterial,
) -> Tuple[np.ndarray, np.ndarray]:
    """Green–Lagrange strain and plane-stress ``σ`` on every element.

    Parameters
    ----------
    u :
        Nodal displacement ``(n_nodes, 3)`` with columns ``ux, uy, uz``
        (i.e. ``sim.nodes - sim.nodes_ref``).
    """
    u = np.asarray(u, dtype=float)
    nodes_ref = np.asarray(nodes_ref, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_e = elements.shape[0]
    strain_elem = np.zeros((n_e, 3), dtype=float)
    stress_elem = np.zeros((n_e, 3), dtype=float)
    for e, conn in enumerate(elements):
        state, _ = element_state_from_u(u[conn], nodes_ref[conn], material)
        strain_elem[e] = state.strain_voigt
        stress_elem[e] = state.stress_voigt
    return strain_elem, stress_elem


def assemble_internal_forces(
    nodes: np.ndarray,
    elements: np.ndarray,
    nodes0: np.ndarray,
    material: MembraneMaterial,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Assemble ``f_int``, ``Π_int``, and element strain/stress."""
    nodes = np.asarray(nodes, dtype=float)
    nodes0 = np.asarray(nodes0, dtype=float)
    elements = np.asarray(elements, dtype=int)
    f_int = np.zeros_like(nodes)
    Pi = 0.0
    n_e = elements.shape[0]
    strain_elem = np.zeros((n_e, 3), dtype=float)
    stress_elem = np.zeros((n_e, 3), dtype=float)

    for e, conn in enumerate(elements):
        f_e, state = element_internal_forces(nodes[conn], nodes0[conn], material)
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
    """Total potential ``Π_int − f_ext · u``."""
    _, Pi_int, _, _ = assemble_internal_forces(
        nodes, elements, nodes0, material
    )
    if f_ext is None:
        return Pi_int
    x_ref = nodes0 if x_ref_work is None else np.asarray(x_ref_work, dtype=float)
    u = np.asarray(nodes, dtype=float) - x_ref
    return float(Pi_int - np.sum(np.asarray(f_ext, dtype=float) * u))
