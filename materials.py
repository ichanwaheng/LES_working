#!/usr/bin/env python
# coding: utf-8

"""Membrane material model and plane-stress constitutive relation.

Plane stress is written in the usual reduced form of isotropic elasticity
(``D`` below). Out-of-plane strain is *not* forced to zero in 3D space:
the membrane lives on a surface, and we only constitutively relate the
*in-plane* Green–Lagrange strain to the in-plane PK2 stress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MembraneMaterial:
    """Isotropic prestressed membrane with linear plane-stress elasticity.

    Constitutive law (Voigt, engineering shear ``γ = 2 E₁₂``)::

        σ = D ε + σ₀

    where ``ε = [E₁₁, E₂₂, 2 E₁₂]``, ``σ = [S₁₁, S₂₂, S₁₂]``,
    ``D = plane_stress_matrix()``, and ``σ₀`` is isotropic prestress.

    Do **not** form ``F = I + grad(u)`` with UFL/`grad` on nodal arrays.
    Membrane kinematics build a 3×2 ``F`` per triangle from deformed
    positions ``x`` vs reference ``X`` (see :mod:`constitutive`);
    displacement is only ``u = x - X`` for bookkeeping.
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

    # N_pre = prestress × thickness is the membrane stress *resultant* [N/m]
    # used by the legacy UWM path. Confirm against Allan's thesis whether the
    # tabulated "prestress" is already a resultant or a Cauchy stress [Pa].
    @property
    def N_pre(self) -> float:
        """Prestress resultant [N/m] = prestress × thickness."""
        return float(self.prestress) * float(self.thickness)

    def prestress_voigt(self) -> np.ndarray:
        """Isotropic PK2 prestress in Voigt form ``[S₁₁, S₂₂, S₁₂]``."""
        s0 = float(self.prestress)
        return np.array([s0, s0, 0.0], dtype=float)

    def constitutive_relation(self, strain_voigt: np.ndarray) -> np.ndarray:
        """Constitutive map used by potential-energy minimisation.

        Parameters
        ----------
        strain_voigt :
            Green–Lagrange strain ``[E₁₁, E₂₂, 2 E₁₂]``.

        Returns
        -------
        sigma :
            PK2 stress ``[S₁₁, S₂₂, S₁₂] = D ε + σ₀``.
        """
        return self.stress(strain_voigt)

    def stress(self, strain_voigt: np.ndarray) -> np.ndarray:
        """PK2 stress (Voigt) from Green–Lagrange strain (Voigt)."""
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

    def constitutive_relation_from_sim(self, sim: Any) -> np.ndarray:
        """Element PK2 stresses from a deformed ``QuasiStaticFSI`` state.

        Uses ``sim.nodes`` (deformed ``x``) and ``sim.nodes_ref`` (reference
        ``X``). Displacement ``u = x - X`` is implied by those positions;
        it is not differentiated with ``grad(u)``.

        Returns
        -------
        stress_elem :
            Array ``(n_elements, 3)`` with rows ``[S₁₁, S₂₂, S₁₂]``.
        """
        from constitutive import assemble_internal_forces

        _f, _Pi, _strain, stress_elem = assemble_internal_forces(
            sim.nodes,
            sim.mesh.elements,
            sim.nodes_ref,
            self,
        )
        return stress_elem
