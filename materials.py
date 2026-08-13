#!/usr/bin/env python
# coding: utf-8

"""Membrane material model and plane-stress constitutive relation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MembraneMaterial:
    """Isotropic prestressed membrane with linear plane-stress elasticity.

    Constitutive law (Voigt, engineering shear ``γ = 2 E₁₂``)::

        σ = D ε + σ₀

    where ``ε = [E₁₁, E₂₂, 2 E₁₂]``, ``σ = [S₁₁, S₂₂, S₁₂]``,
    ``D = plane_stress_matrix()``, and ``σ₀`` is isotropic prestress.
    """

    E: float = 5.0e8
    nu: float = 0.3
    thickness: float = 0.001
    prestress: float = 5.0e4

    def plane_stress_matrix(self) -> np.ndarray:
        """Return the 3×3 plane-stress constitutive matrix ``D``."""
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

    @property
    def N_pre(self) -> float:
        """Prestress resultant [N/m] = prestress × thickness."""
        return float(self.prestress) * float(self.thickness)

    def prestress_voigt(self) -> np.ndarray:
        """Isotropic PK2 prestress in Voigt form ``[S₁₁, S₂₂, S₁₂]``."""
        s0 = float(self.prestress)
        return np.array([s0, s0, 0.0], dtype=float)

    def stress(self, strain_voigt: np.ndarray) -> np.ndarray:
        """Cauchy/PK2 stress (Voigt) from Green–Lagrange strain (Voigt).

        Parameters
        ----------
        strain_voigt :
            ``[E₁₁, E₂₂, 2 E₁₂]``
        """
        eps = np.asarray(strain_voigt, dtype=float).reshape(3)
        return self.plane_stress_matrix() @ eps + self.prestress_voigt()

    def strain_energy_density(self, strain_voigt: np.ndarray) -> float:
        """Strain-energy density ``½ ε·D·ε + σ₀·ε`` [J/m³]."""
        eps = np.asarray(strain_voigt, dtype=float).reshape(3)
        D = self.plane_stress_matrix()
        s0 = self.prestress_voigt()
        return float(0.5 * eps @ D @ eps + s0 @ eps)

    def pk2_matrix(self, strain_voigt: np.ndarray) -> np.ndarray:
        """Return 2×2 second Piola–Kirchhoff stress from Voigt strain."""
        s = self.stress(strain_voigt)
        return np.array([[s[0], s[2]], [s[2], s[1]]], dtype=float)
