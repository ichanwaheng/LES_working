#!/usr/bin/env python
# coding: utf-8

"""Potential-energy minimisation using constitutive stress/strain.

Finds free-node positions that minimise::

    Π(x) = Σₑ t A₀ (½ ε·D·ε)  −  f_ext · u

with analytic gradient from constitutive internal forces (σ = D ε).
The gradient ``∇Π = f_int − f_ext`` uses the total-Lagrangian constitutive
internal forces, so stress/strain enter the minimisation directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from constitutive import assemble_internal_forces, total_potential
from materials import MembraneMaterial


@dataclass
class EnergyMinResult:
    nodes: np.ndarray
    n_iters: int
    residual: float
    objective: float
    strain_elem: np.ndarray
    stress_elem: np.ndarray
    max_strain: float
    max_stress: float


def minimize_potential_energy(
    nodes: np.ndarray,
    elements: np.ndarray,
    fixed: np.ndarray,
    material: MembraneMaterial,
    nodes0: Optional[np.ndarray] = None,
    f_ext: Optional[np.ndarray] = None,
    support_nodes: Optional[np.ndarray] = None,
    max_iters: int = 25,
    tol: float = 1e-7,
    under_relaxation: float = 1.0,
    step_scale: float = 1.0,
) -> EnergyMinResult:
    """Minimise constitutive potential energy for free membrane nodes.

    Parameters
    ----------
    nodes0 :
        Undeformed reference geometry for Green–Lagrange strain (defaults to
        ``support_nodes`` / flat supports).
    support_nodes :
        Dirichlet values for fixed nodes.
    under_relaxation :
        Blend factor between the previous guess and the minimiser result
        (``1`` = take the PE minimum fully).
    """
    del step_scale  # kept for call-site compatibility; L-BFGS uses its own steps

    x_init = np.asarray(nodes, dtype=float).copy()
    elements = np.asarray(elements, dtype=int)
    fixed = np.asarray(fixed, dtype=bool)
    x_support = (
        np.asarray(nodes, dtype=float).copy()
        if support_nodes is None
        else np.asarray(support_nodes, dtype=float).copy()
    )
    x0 = (
        np.asarray(x_support, dtype=float).copy()
        if nodes0 is None
        else np.asarray(nodes0, dtype=float).copy()
    )
    f_ext_arr = (
        np.zeros_like(x_init)
        if f_ext is None
        else np.asarray(f_ext, dtype=float).copy()
    )
    alpha = float(np.clip(under_relaxation, 1e-3, 1.0))
    free = np.where(~fixed)[0]

    if free.size == 0:
        f_int, Pi_int, strain_elem, stress_elem = assemble_internal_forces(
            x_support, elements, x0, material
        )
        return EnergyMinResult(
            nodes=x_support,
            n_iters=0,
            residual=0.0,
            objective=float(Pi_int),
            strain_elem=strain_elem,
            stress_elem=stress_elem,
            max_strain=float(np.max(np.abs(strain_elem))) if strain_elem.size else 0.0,
            max_stress=float(np.max(np.abs(stress_elem))) if stress_elem.size else 0.0,
        )

    def unpack(v: np.ndarray) -> np.ndarray:
        out = x_support.copy()
        out[free] = np.asarray(v, dtype=float).reshape(-1, 3)
        out[fixed] = x_support[fixed]
        return out

    def fun(v: np.ndarray) -> float:
        return total_potential(
            unpack(v),
            elements,
            x0,
            material,
            f_ext_arr,
            x_ref_work=x_support,
        )

    def jac(v: np.ndarray) -> np.ndarray:
        xv = unpack(v)
        f_int, _, _, _ = assemble_internal_forces(xv, elements, x0, material)
        R = f_int - f_ext_arr
        return R[free].ravel()

    v0 = x_init[free].ravel().copy()
    opt = minimize(
        fun,
        v0,
        jac=jac,
        method="L-BFGS-B",
        options={
            "maxiter": int(max_iters),
            "ftol": float(tol),
            "gtol": float(tol),
            "maxls": 40,
        },
    )
    x_star = unpack(opt.x)
    x = (1.0 - alpha) * x_init + alpha * x_star
    x[fixed] = x_support[fixed]

    f_int, Pi_int, strain_elem, stress_elem = assemble_internal_forces(
        x, elements, x0, material
    )
    R = f_int - f_ext_arr
    R[fixed] = 0.0
    # Normalize by load/internal scale; fall back to prestress×length so a
    # self-equilibrated prestressed state (‖f_int‖≈0) is not reported as 1.
    span = float(np.linalg.norm(x0.max(axis=0) - x0.min(axis=0))) + 1e-12
    f_scale = max(
        float(np.linalg.norm(f_int[free])),
        float(np.linalg.norm(f_ext_arr[free])),
        float(material.prestress) * float(material.thickness) * span,
        1e-12,
    )
    residual = float(np.linalg.norm(R[free]) / f_scale)
    Pi = float(
        total_potential(
            x, elements, x0, material, f_ext_arr, x_ref_work=x_support
        )
    )
    max_strain = float(np.max(np.abs(strain_elem))) if strain_elem.size else 0.0
    max_stress = float(np.max(np.abs(stress_elem))) if stress_elem.size else 0.0

    return EnergyMinResult(
        nodes=x,
        n_iters=int(opt.nit),
        residual=residual,
        objective=Pi,
        strain_elem=strain_elem,
        stress_elem=stress_elem,
        max_strain=max_strain,
        max_stress=max_stress,
    )
