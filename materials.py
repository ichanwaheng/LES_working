#!/usr/bin/env python
# coding: utf-8

"""Membrane material + plane-stress constitutive relation.

Continuum statement (plane stress, Green–Lagrange)::

    u = x - X
    F = I + ∇u
    ε = ½ (Fᵀ F - I)
    σ = D ε + σ₀

On the membrane, ``I`` / ``∇u`` / ``F`` live in the *in-plane material*
chart (``F`` is 3×2, ``I`` in ``FᵀF - I`` is 2×2). Nodal ``grad(u)`` is
the CST shape-function gradient — not UFL ``grad`` on a NumPy vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass
class MembraneMaterial:
    E: float = 5.0e8
    nu: float = 0.3
    thickness: float = 0.001
    prestress: float = 5.0e4

    def plane_stress_matrix(self) -> np.ndarray:
        """Constitutive matrix ``D`` (plane stress, Voigt)."""
        E, nu = float(self.E), float(self.nu)
        factor = E / (1.0 - nu**2)
        return factor * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ],
            dtype=float,
        )

    # Confirm vs Allan's thesis: prestress [Pa] × thickness → resultant [N/m].
    @property
    def N_pre(self) -> float:
        return float(self.prestress) * float(self.thickness)

    def prestress_voigt(self) -> np.ndarray:
        """Isotropic prestress ``σ₀ = [S₁₁, S₂₂, S₁₂]``."""
        s0 = float(self.prestress)
        return np.array([s0, s0, 0.0], dtype=float)

    def constitutive_relation(
        self,
        epsilon: np.ndarray,
    ) -> np.ndarray:
        """Plane-stress map ``σ = D ε + σ₀`` (Voigt).

        Parameters
        ----------
        epsilon :
            Green–Lagrange strain ``[E₁₁, E₂₂, 2 E₁₂]``.
        """
        eps = np.asarray(epsilon, dtype=float).reshape(3)
        return self.plane_stress_matrix() @ eps + self.prestress_voigt()

    # Aliases used by constitutive.py / energy_minimize.py
    def stress(self, strain_voigt: np.ndarray) -> np.ndarray:
        return self.constitutive_relation(strain_voigt)

    def strain_energy_density(self, strain_voigt: np.ndarray) -> float:
        eps = np.asarray(strain_voigt, dtype=float).reshape(3)
        D = self.plane_stress_matrix()
        s0 = self.prestress_voigt()
        return float(0.5 * eps @ D @ eps + s0 @ eps)

    def pk2_matrix(self, strain_voigt: np.ndarray) -> np.ndarray:
        s = self.constitutive_relation(strain_voigt)
        return np.array([[s[0], s[2]], [s[2], s[1]]], dtype=float)

    def constitutive_relation_from_sim(self, sim: Any) -> np.ndarray:
        """Element stresses from deformed ``sim`` using ``ε = ½(FᵀF - I)``.

        Displacement field::

            u = sim.nodes - sim.nodes_ref   # (n_nodes, 3); columns ux, uy, uz

        Per triangle, ``F = I + ∇u`` in the material plane, then
        ``ε = ½(FᵀF - I)`` and ``σ = D ε + σ₀``.

        Returns
        -------
        sigma :
            ``(n_elements, 3)`` with rows ``[S₁₁, S₂₂, S₁₂]``.
        """
        from constitutive import stress_from_displacement_field

        u = np.asarray(sim.nodes, dtype=float) - np.asarray(
            sim.nodes_ref, dtype=float
        )
        _eps, sigma = stress_from_displacement_field(
            u,
            sim.nodes_ref,
            sim.mesh.elements,
            self,
        )
        return sigma


def green_lagrange_from_F(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``ε = ½(Fᵀ F - I)`` → (2×2 tensor, Voigt ``[E11, E22, 2E12]``)."""
    I2 = np.eye(2)
    E = 0.5 * (F.T @ F - I2)
    eps_voigt = np.array([E[0, 0], E[1, 1], 2.0 * E[0, 1]], dtype=float)
    return E, eps_voigt
