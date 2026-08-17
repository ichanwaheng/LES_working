#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Plane stress: in-plane Green–Lagrange strain → in-plane stress via D.
# Prestress is NOT added inside σ = D ε; it is kept separate as N_pre.


@dataclass
class MembraneMaterial:
    E: float = 5.0e8
    nu: float = 0.3
    thickness: float = 0.001
    prestress: float = 5.0e4

    def plane_stress_matrix(self) -> np.ndarray:
        """Constitutive matrix D (plane stress, Voigt 3×3)."""
        E, nu = self.E, self.nu
        factor = E / (1.0 - nu**2)
        return factor * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ],
            dtype=float,
        )

    # Separate from constitutive map σ = D ε (check vs Allan's thesis).
    @property
    def N_pre(self) -> float:
        return self.prestress * self.thickness

    def constitutive_relation(self, sim: Any) -> np.ndarray:
        """
        Continuum:
            u = x - X
            F = I + ∇u
            ε = ½ (Fᵀ F - I)          # Green–Lagrange
            σ = D ε                   # plane stress (NO prestress term)

        Discrete ∇u is the CST shape-function gradient per triangle
        (see constitutive.py), not ufl.grad / np.gradient.

        Returns
        -------
        sigma : (n_elements, 3) with rows [S11, S22, S12]
        """
        from constitutive import stress_from_displacement_field

        u = sim.nodes - sim.nodes_ref  # (n_nodes, 3): ux, uy, uz
        _eps, sigma = stress_from_displacement_field(
            u,
            sim.nodes_ref,
            sim.mesh.elements,
            self,
        )
        return sigma

    def stress(self, strain_voigt: np.ndarray) -> np.ndarray:
        """σ = D ε with ε = [E11, E22, 2 E12]."""
        eps = np.asarray(strain_voigt, dtype=float).reshape(3)
        return self.plane_stress_matrix() @ eps

    def strain_energy_density(self, strain_voigt: np.ndarray) -> float:
        """W = ½ ε · D · ε."""
        eps = np.asarray(strain_voigt, dtype=float).reshape(3)
        D = self.plane_stress_matrix()
        return float(0.5 * eps @ D @ eps)

    def pk2_matrix(self, strain_voigt: np.ndarray) -> np.ndarray:
        s = self.stress(strain_voigt)
        return np.array([[s[0], s[2]], [s[2], s[1]]], dtype=float)
