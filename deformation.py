"""Post form-finding membrane deformation capture.

After ``QuasiStaticFSI.run()`` has found the equilibrium form, use this module
to advance the fluid in time and update membrane node positions by
**minimizing potential energy** under the current fluid loads.

Form-finding code in ``Coupling.py`` is left unchanged.

Typical usage
-------------
::

    from Coupling import QuasiStaticFSI
    from deformation import MembraneDeformationCapture

    sim = QuasiStaticFSI(cfg)
    sim.run()                          # stage 1: form-find (unchanged)

    cap = MembraneDeformationCapture(sim)
    hist = cap.run(t_end=2.0, callback=on_frame)   # stage 2: deformation
    cap.write_gif("output/membrane_deformation.gif")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mesh_update import under_relax, update_immsersed_boundary as update_immersed_boundary
from UWM import (
    compute_edge_weights,
    solve_force_density,
    updated_weight_form_find,
    uwm_objective,
)


Callback = Optional[Callable[[Any, Dict[str, float], int], None]]


@dataclass
class DeformationHistory:
    """Time history of post-form membrane deformation."""

    iteration: List[int] = field(default_factory=list)
    time: List[float] = field(default_factory=list)
    max_disp: List[float] = field(default_factory=list)
    max_disp_from_form: List[float] = field(default_factory=list)
    energy: List[float] = field(default_factory=list)
    residual: List[float] = field(default_factory=list)
    pressure_max: List[float] = field(default_factory=list)
    cfl: List[float] = field(default_factory=list)
    nodes: List[np.ndarray] = field(default_factory=list)


class MembraneDeformationCapture:
    """Capture membrane deformation after the form has been found.

    Each step:
      1. update immersed boundary
      2. advance fluid ``fluid_substeps``
      3. map pressure → nodal loads
      4. minimize potential energy for free nodes
      5. record displacement relative to the form-found shape

    By default edge weights are **frozen** from the form-found geometry so
    this stage tracks deformation of that form under changing fluid load,
    rather than re-doing full form-finding.
    """

    def __init__(
        self,
        sim: Any,
        *,
        freeze_weights: bool = True,
        under_relaxation: Optional[float] = None,
        max_energy_iters: int = 15,
        energy_tol: Optional[float] = None,
        record_nodes: bool = True,
    ) -> None:
        self.sim = sim
        self.freeze_weights = bool(freeze_weights)
        self.alpha = float(
            sim.alpha_shape if under_relaxation is None else under_relaxation
        )
        self.max_energy_iters = int(max_energy_iters)
        self.energy_tol = float(sim.uwm_tol if energy_tol is None else energy_tol)
        self.record_nodes = bool(record_nodes)

        # Reference = form-found shape at capture start
        self.nodes_form = np.asarray(sim.nodes, dtype=float).copy()
        self.x_bc = np.asarray(sim.x_bc, dtype=float).copy()
        self.elements = np.asarray(sim.mesh.elements, dtype=int)
        self.fixed = np.asarray(sim.mesh.fixed, dtype=bool)
        self.N_pre = float(sim.N_pre)

        self.weights = compute_edge_weights(
            self.nodes_form, self.elements, self.N_pre
        )
        self._f_old: Optional[np.ndarray] = None
        self.history = DeformationHistory()
        self.frame = 0

        # Seed history with the form-found configuration
        self._record(
            time=float(sim.time),
            nodes=self.nodes_form,
            energy=float(uwm_objective(self.nodes_form, self.weights)),
            residual=0.0,
            pressure_max=0.0,
            cfl=0.0,
        )

    # ------------------------------------------------------------------ API

    def step(self) -> Dict[str, float]:
        """One deformation step: fluid advance + energy minimization."""
        sim = self.sim
        x_prev = np.asarray(sim.nodes, dtype=float).copy()

        update_immersed_boundary(
            sim.fluid,
            sim.grid,
            sim.mesh,
            sim.nodes,
            np.zeros_like(sim.nodes),
        )
        for _ in range(sim.fluid_substeps):
            sim.fluid.step(sim.dt)

        pressure, f_nodal = sim._compute_loads()
        if self._f_old is None:
            f_use = f_nodal
        else:
            alpha_load = max(float(sim.alpha_load), 0.65)
            f_use = under_relax(f_nodal, self._f_old, alpha_load)
        self._f_old = f_use.copy()

        x_new, energy, residual, n_iters = self._minimize_energy(x_prev, f_use)
        x_new = under_relax(x_new, x_prev, self.alpha)
        x_new[self.fixed] = self.x_bc[self.fixed]

        sim.nodes = x_new
        sim.mesh.nodes = x_new.copy()
        sim.iteration += 1
        sim.time += sim.fluid_substeps * sim.dt

        disp_from_bc = float(
            np.max(np.linalg.norm(x_new - self.x_bc, axis=1))
        )
        disp_from_form = float(
            np.max(np.linalg.norm(x_new - self.nodes_form, axis=1))
        )
        info: Dict[str, float] = {
            "iteration": float(sim.iteration),
            "time": float(sim.time),
            "max_disp": disp_from_bc,
            "max_disp_from_form": disp_from_form,
            "energy": float(energy),
            "energy_residual": float(residual),
            "energy_iters": float(n_iters),
            "pressure_max": float(np.max(np.abs(pressure))),
            "net_fz": float(np.sum(f_use[:, 2])),
            "cfl": float(sim.fluid.max_cfl(sim.dt)),
        }
        self._record(
            time=info["time"],
            nodes=x_new,
            energy=info["energy"],
            residual=info["energy_residual"],
            pressure_max=info["pressure_max"],
            cfl=info["cfl"],
            max_disp=info["max_disp"],
            max_disp_from_form=info["max_disp_from_form"],
        )
        return info

    def run(
        self,
        t_end: Optional[float] = None,
        n_steps: Optional[int] = None,
        callback: Callback = None,
    ) -> DeformationHistory:
        """Advance deformation capture until ``t_end`` or ``n_steps``."""
        sim = self.sim
        target = float(sim.t_end if t_end is None else t_end)
        k = 0
        max_steps = (
            int(n_steps)
            if n_steps is not None
            else max(int(sim.max_iters), 1) * 50
        )

        while sim.time < target - 1e-15 and k < max_steps:
            info = self.step()
            if callback is not None:
                callback(sim, info, k)
            k += 1
        return self.history

    def write_gif(
        self,
        out_path: str | Path,
        fps: int = 8,
        amplify: float = 1.0,
        title: str = "Membrane deformation (post form-find)",
    ) -> Path:
        """Write a GIF from recorded node sets (relative to form-found shape)."""
        from viz import save_deformation_gif

        if not self.history.nodes:
            raise ValueError("no recorded frames — run() first")
        return save_deformation_gif(
            nodes_over_time=self.history.nodes,
            elements=self.elements,
            nodes0=self.nodes_form,
            out_path=out_path,
            times=self.history.time,
            fps=fps,
            amplify=amplify,
            title=title,
        )

    # -------------------------------------------------------------- private

    def _minimize_energy(
        self,
        nodes: np.ndarray,
        f_ext: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, int]:
        """Minimize membrane potential energy under ``f_ext``.

        Frozen-weight mode solves the force-density system with the form-found
        edge weights (deformation of the found form).

        If ``freeze_weights`` is False, falls back to a short UWM update.
        """
        if self.freeze_weights:
            x = np.asarray(nodes, dtype=float).copy()
            residual = np.inf
            energy = 0.0
            it = 0
            for it in range(1, self.max_energy_iters + 1):
                x_new = solve_force_density(x, self.fixed, self.weights, f_ext)
                x_new[self.fixed] = self.x_bc[self.fixed]
                residual = float(
                    np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-12)
                )
                x = (1.0 - self.alpha) * x + self.alpha * x_new
                x[self.fixed] = self.x_bc[self.fixed]
                energy = float(uwm_objective(x, self.weights))
                if residual < self.energy_tol:
                    break
            return x, energy, residual, it

        uwm = updated_weight_form_find(
            nodes,
            self.elements,
            self.fixed,
            N_pre=self.N_pre,
            f_ext=f_ext,
            support_nodes=self.x_bc,
            max_weight_updates=self.max_energy_iters,
            tol=self.energy_tol,
            under_relaxation=self.alpha,
        )
        return uwm.nodes, float(uwm.objective), float(uwm.residual), int(
            uwm.n_weight_updates
        )

    def _record(
        self,
        *,
        time: float,
        nodes: np.ndarray,
        energy: float,
        residual: float,
        pressure_max: float,
        cfl: float,
        max_disp: Optional[float] = None,
        max_disp_from_form: Optional[float] = None,
    ) -> None:
        nodes = np.asarray(nodes, dtype=float)
        if max_disp is None:
            max_disp = float(np.max(np.linalg.norm(nodes - self.x_bc, axis=1)))
        if max_disp_from_form is None:
            max_disp_from_form = float(
                np.max(np.linalg.norm(nodes - self.nodes_form, axis=1))
            )
        self.frame += 1
        self.history.iteration.append(self.frame)
        self.history.time.append(float(time))
        self.history.max_disp.append(float(max_disp))
        self.history.max_disp_from_form.append(float(max_disp_from_form))
        self.history.energy.append(float(energy))
        self.history.residual.append(float(residual))
        self.history.pressure_max.append(float(pressure_max))
        self.history.cfl.append(float(cfl))
        if self.record_nodes:
            self.history.nodes.append(nodes.copy())


def capture_deformation_after_form_find(
    sim: Any,
    *,
    t_end: Optional[float] = None,
    n_steps: Optional[int] = None,
    callback: Callback = None,
    freeze_weights: bool = True,
    gif_path: Optional[str | Path] = None,
    gif_amplify: float = 5.0,
    fps: int = 8,
) -> MembraneDeformationCapture:
    """Convenience: run post-form deformation capture on a finished QS-FSI sim.

    Parameters
    ----------
    sim :
        ``QuasiStaticFSI`` instance after ``sim.run()`` (form already found).
    t_end, n_steps :
        Stopping criteria for the deformation stage.
    gif_path :
        If set, write a deformation GIF when finished.
    """
    cap = MembraneDeformationCapture(sim, freeze_weights=freeze_weights)
    cap.run(t_end=t_end, n_steps=n_steps, callback=callback)
    if gif_path is not None:
        cap.write_gif(gif_path, fps=fps, amplify=1.0)
        if float(gif_amplify) != 1.0:
            amp_path = Path(gif_path)
            amp_path = amp_path.with_name(
                amp_path.stem + "_amplified" + amp_path.suffix
            )
            cap.write_gif(
                amp_path,
                fps=fps,
                amplify=float(gif_amplify),
                title=f"Membrane deformation (×{float(gif_amplify):g})",
            )
    return cap
