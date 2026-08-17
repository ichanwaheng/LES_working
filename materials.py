#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Do NOT use: from ufl import grad
# ufl.grad only works on UFL/FEniCS Function fields in a form compiler,
# not on NumPy nodal arrays.


# Plane stress (in-plane Green–Lagrange → in-plane PK2).
# Out-of-plane *membrane* motion (uz) is allowed; the constitutive law
# only relates the in-plane strain components on the surface.


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

    # Check vs Allan's thesis: prestress [Pa] × thickness → resultant [N/m].
    @property
    def N_pre(self) -> float:
        return self.prestress * self.thickness

    def prestress_voigt(self) -> np.ndarray:
        s0 = float(self.prestress)
        return np.array([s0, s0, 0.0], dtype=float)

    def constitutive_relation(self, sim: Any) -> np.ndarray:
        """Plane-stress constitutive relation for the deformed membrane.

        Continuum statement
        -------------------
        ::

            u = x - X                         # displacement (ux, uy, uz)
            F = I + ∇u                        # deformation gradient
            ε = ½ (Fᵀ F - I)                  # Green–Lagrange strain
            σ = D ε + σ₀                      # plane stress (D = plane_stress_matrix)

        Why drafts with ``np.gradient`` / ``ufl.grad`` fail
        -------------------------------------------------
        1. ``from ufl import grad`` — FEniCS only; not for NumPy nodes.
        2. ``grad(u) = ...`` is invalid Python (cannot assign to a call).
        3. After ``u = sim.nodes - sim.nodes_ref``, there are no separate
           ``ux, uy, uz`` names unless you unpack columns::

               ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]

        4. ``np.gradient(ux, dx, dy, dz)`` needs a *regular 3D grid* and
           spacings ``dx,dy,dz``. Membrane nodes are an unstructured /
           surface mesh — those spacings are undefined, and nodal vectors
           are not ``(nx,ny,nz)`` fields.
        5. Membrane ``F`` is ``3×2`` (surface chart). ``I`` in
           ``ε = ½(FᵀF − I)`` is ``2×2``, not ``eye(3)``.
        6. ``D @ epsilon`` needs Voigt ``[E11, E22, 2 E12]``, not a
           ``3×3`` matrix.
        7. Must be a *method* on this class; include prestress ``σ₀``.

        Correct discrete ``∇u`` is the CST shape-function gradient on each
        triangle (see ``constitutive.deformation_gradient_from_u``)::

            F = I_surf + Σ_a u_a ⊗ ∇N_a

        Returns
        -------
        sigma : ndarray, shape (n_elements, 3)
            PK2 stress per triangle ``[S11, S22, S12]``.
        """
        from constitutive import stress_from_displacement_field

        # u = x - X  (n_nodes, 3) with columns ux, uy, uz
        u = np.asarray(sim.nodes, dtype=float) - np.asarray(
            sim.nodes_ref, dtype=float
        )

        # Per triangle (constitutive.py):
        #   F = I_surf + grad(u)
        #   epsilon = 0.5 * (F.T @ F - I2)
        #   sigma = D @ voigt(epsilon) + prestress
        _eps, sigma = stress_from_displacement_field(
            u,
            sim.nodes_ref,
            sim.mesh.elements,
            self,
        )
        return sigma

    # --- helpers used by energy minimisation (Voigt form) -------------------

    def stress(self, strain_voigt: np.ndarray) -> np.ndarray:
        """σ = D ε + σ₀ with ε = [E11, E22, 2 E12]."""
        eps = np.asarray(strain_voigt, dtype=float).reshape(3)
        return self.plane_stress_matrix() @ eps + self.prestress_voigt()

    def strain_energy_density(self, strain_voigt: np.ndarray) -> float:
        eps = np.asarray(strain_voigt, dtype=float).reshape(3)
        D = self.plane_stress_matrix()
        s0 = self.prestress_voigt()
        return float(0.5 * eps @ D @ eps + s0 @ eps)

    def pk2_matrix(self, strain_voigt: np.ndarray) -> np.ndarray:
        s = self.stress(strain_voigt)
        return np.array([[s[0], s[2]], [s[2], s[1]]], dtype=float)
