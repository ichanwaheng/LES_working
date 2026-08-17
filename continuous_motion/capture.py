"""Capture continuous motion of a form-found membrane in fluid flow.

Typical usage
-------------
::

    from Coupling import QuasiStaticFSI
    from continuous_motion import ContinuousMotionCapture

    sim = QuasiStaticFSI(cfg)
    sim.run()   # form-find — membrane has found its formed shape

    cap = ContinuousMotionCapture(sim)
    cap.run(t_end=2.0)
    cap.write_video("continuous_motion/output/membrane_continuous_motion.mp4")
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

# Package lives in continuous_motion/; repo root must be importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from FSI_load_transfer import (
    dynamic_pressure_loads,
    interpolated_field_loads,
    pressure_jump_loads,
)
from mesh_update import (
    under_relax,
    update_immsersed_boundary as update_immersed_boundary,
)

from .dynamics import (
    advance_structure,
    freeze_form_weights,
    nodal_tributary_mass,
)
from .video import write_simulation_video

Callback = Optional[Callable[[Any, Dict[str, float], int], None]]


@dataclass
class ContinuousMotionHistory:
    """Time history for post-form continuous submerged motion."""

    iteration: List[int] = field(default_factory=list)
    time: List[float] = field(default_factory=list)
    max_disp: List[float] = field(default_factory=list)
    max_disp_from_form: List[float] = field(default_factory=list)
    max_speed: List[float] = field(default_factory=list)
    kinetic: List[float] = field(default_factory=list)
    pressure_max: List[float] = field(default_factory=list)
    cfl: List[float] = field(default_factory=list)
    nodes: List[np.ndarray] = field(default_factory=list)
    fluid_slices: List[np.ndarray] = field(default_factory=list)


class ContinuousMotionCapture:
    """Advance a formed membrane submerged in fluid flow; record video frames.

    Unlike quasi-static form-finding / energy-minimization deformation, this
    stage integrates membrane momentum so the surface keeps moving under the
    unsteady flow.
    """

    def __init__(
        self,
        sim: Any,
        *,
        mass_scale: Optional[float] = None,
        damping: Optional[float] = None,
        fluid_substeps: Optional[int] = None,
        structure_substeps: int = 4,
        load_scale: Optional[float] = None,
        load_mode: Optional[str] = None,
        record_fluid_slice: bool = True,
        record_nodes: bool = True,
    ) -> None:
        self.sim = sim
        mcfg = sim.cfg.get("membrane", {})
        cmcfg = sim.cfg.get("continuous_motion", {})

        self.mass_scale = float(
            mass_scale
            if mass_scale is not None
            else cmcfg.get("mass_scale", mcfg.get("mass_scale", 80.0))
        )
        self.damping = float(
            damping
            if damping is not None
            else cmcfg.get("damping", mcfg.get("damping", 80.0))
        )
        self.fluid_substeps = int(
            fluid_substeps
            if fluid_substeps is not None
            else cmcfg.get("fluid_substeps", max(sim.fluid_substeps // 5, 1))
        )
        self.structure_substeps = int(
            cmcfg.get("structure_substeps", structure_substeps)
        )
        self.load_scale = float(
            load_scale
            if load_scale is not None
            else cmcfg.get("load_scale", sim.load_scale)
        )
        self.load_mode = str(
            load_mode
            if load_mode is not None
            else cmcfg.get("load_mode", sim.load_mode)
        )
        self.record_fluid_slice = bool(record_fluid_slice)
        self.record_nodes = bool(record_nodes)

        self.nodes_form = np.asarray(sim.nodes, dtype=float).copy()
        self.x_bc = np.asarray(sim.x_bc, dtype=float).copy()
        self.elements = np.asarray(sim.mesh.elements, dtype=int)
        self.fixed = np.asarray(sim.mesh.fixed, dtype=bool)
        self.N_pre = float(sim.N_pre)
        self.weights = freeze_form_weights(
            self.nodes_form, self.elements, self.N_pre
        )
        self.mass = nodal_tributary_mass(
            self.nodes_form,
            self.elements,
            density=float(sim.mesh.density),
            thickness=float(sim.mesh.thickness),
            mass_scale=self.mass_scale,
        )
        self.velocities = np.zeros_like(self.nodes_form)
        self._f_old: Optional[np.ndarray] = None
        self.history = ContinuousMotionHistory()
        self.frame = 0
        self.mesh_nx = int(sim.mesh.nx)
        self.mesh_ny = int(sim.mesh.ny)

        # Seed with the formed (static) shape at capture start
        self._record(
            time=float(sim.time),
            nodes=self.nodes_form,
            max_disp=0.0,
            max_disp_from_form=0.0,
            max_speed=0.0,
            kinetic=0.0,
            pressure_max=0.0,
            cfl=0.0,
        )

    # ------------------------------------------------------------------ API

    def _compute_loads(self):
        sim = self.sim
        if self.load_mode == "interpolated_field":
            pressure, f_nodal = interpolated_field_loads(
                sim.fluid, sim.mesh, sim.nodes
            )
        elif self.load_mode == "pressure_jump":
            offset = 3.0 * sim.grid.dz
            pressure, f_nodal = pressure_jump_loads(
                sim.fluid,
                sim.mesh,
                sim.nodes,
                rho=float(sim.cfg["fluid"]["rho"]),
                U_ref=float(sim.cfg["fluid"]["U_inlet"]),
                offset=offset,
            )
        else:
            pressure, f_nodal = dynamic_pressure_loads(
                sim.fluid,
                sim.mesh,
                sim.nodes,
                rho=float(sim.cfg["fluid"]["rho"]),
                U_ref=float(sim.cfg["fluid"]["U_inlet"]),
            )
        return pressure, f_nodal * self.load_scale

    def step(self) -> Dict[str, float]:
        """One outer frame: fluid advance + dynamic membrane update."""
        sim = self.sim

        update_immersed_boundary(
            sim.fluid,
            sim.grid,
            sim.mesh,
            sim.nodes,
            self.velocities,
        )
        for _ in range(self.fluid_substeps):
            sim.fluid.step(sim.dt)

        pressure, f_nodal = self._compute_loads()
        if self._f_old is None:
            f_use = f_nodal
        else:
            alpha_load = max(float(sim.alpha_load), 0.55)
            f_use = under_relax(f_nodal, self._f_old, alpha_load)
        self._f_old = f_use.copy()

        dt_struct = self.fluid_substeps * sim.dt
        x_new, v_new = advance_structure(
            sim.nodes,
            self.velocities,
            fixed=self.fixed,
            x_bc=self.x_bc,
            weights=self.weights,
            f_ext=f_use,
            mass=self.mass,
            damping=self.damping,
            dt=dt_struct,
            n_substeps=self.structure_substeps,
        )

        sim.nodes = x_new
        sim.mesh.nodes = x_new.copy()
        self.velocities = v_new
        sim.iteration += 1
        sim.time += dt_struct

        speed = np.linalg.norm(v_new, axis=1)
        free = ~self.fixed
        ke = 0.5 * float(np.sum(self.mass[free] * speed[free] ** 2))
        info: Dict[str, float] = {
            "iteration": float(sim.iteration),
            "time": float(sim.time),
            "max_disp": float(
                np.max(np.linalg.norm(x_new - self.x_bc, axis=1))
            ),
            "max_disp_from_form": float(
                np.max(np.linalg.norm(x_new - self.nodes_form, axis=1))
            ),
            "max_speed": float(np.max(speed)),
            "kinetic": ke,
            "pressure_max": float(np.max(np.abs(pressure))),
            "net_fz": float(np.sum(f_use[:, 2])),
            "cfl": float(sim.fluid.max_cfl(sim.dt)),
        }
        self._record(
            time=info["time"],
            nodes=x_new,
            max_disp=info["max_disp"],
            max_disp_from_form=info["max_disp_from_form"],
            max_speed=info["max_speed"],
            kinetic=info["kinetic"],
            pressure_max=info["pressure_max"],
            cfl=info["cfl"],
        )
        return info

    def run(
        self,
        t_end: Optional[float] = None,
        n_steps: Optional[int] = None,
        callback: Callback = None,
    ) -> ContinuousMotionHistory:
        """Integrate until ``t_end`` (absolute sim time) or ``n_steps`` frames."""
        sim = self.sim
        cmcfg = sim.cfg.get("continuous_motion", {})
        if t_end is None:
            t_end = float(cmcfg.get("t_end", sim.t_end))
        # Absolute target: if form-find already advanced time, keep moving forward
        target = float(t_end)
        if target <= sim.time + 1e-15:
            target = sim.time + float(t_end)

        k = 0
        max_steps = (
            int(n_steps)
            if n_steps is not None
            else max(int(sim.max_iters), 1) * 100
        )
        while sim.time < target - 1e-15 and k < max_steps:
            info = self.step()
            if callback is not None:
                callback(sim, info, k)
            k += 1
        return self.history

    def write_video(
        self,
        out_path: str | Path,
        fps: int = 12,
        amplify: float = 1.0,
        title: str = "Formed membrane in fluid flow",
    ):
        """Write the simulation video (MP4, GIF fallback) from recorded frames."""
        if not self.history.nodes:
            raise ValueError("no recorded frames — run() first")

        fluid_slices: Optional[Sequence[np.ndarray]] = None
        grid_x = grid_z = None
        if self.history.fluid_slices:
            fluid_slices = self.history.fluid_slices
            grid_x = self.sim.grid.x
            grid_z = self.sim.grid.z

        return write_simulation_video(
            nodes_over_time=self.history.nodes,
            elements=self.elements,
            nodes0=self.nodes_form,
            out_path=out_path,
            times=self.history.time,
            fps=fps,
            amplify=amplify,
            title=title,
            fluid_slices=fluid_slices,
            grid_x=grid_x,
            grid_z=grid_z,
            mesh_nx=self.mesh_nx,
            mesh_ny=self.mesh_ny,
        )

    # -------------------------------------------------------------- private

    def _midplane_speed(self) -> np.ndarray:
        st = self.sim.fluid.state
        j = self.sim.grid.ny // 2
        u = np.asarray(st.u[:, j, :], dtype=float)
        v = np.asarray(st.v[:, j, :], dtype=float)
        w = np.asarray(st.w[:, j, :], dtype=float)
        return np.sqrt(u * u + v * v + w * w)

    def _record(
        self,
        *,
        time: float,
        nodes: np.ndarray,
        max_disp: float,
        max_disp_from_form: float,
        max_speed: float,
        kinetic: float,
        pressure_max: float,
        cfl: float,
    ) -> None:
        self.frame += 1
        self.history.iteration.append(self.frame)
        self.history.time.append(float(time))
        self.history.max_disp.append(float(max_disp))
        self.history.max_disp_from_form.append(float(max_disp_from_form))
        self.history.max_speed.append(float(max_speed))
        self.history.kinetic.append(float(kinetic))
        self.history.pressure_max.append(float(pressure_max))
        self.history.cfl.append(float(cfl))
        if self.record_nodes:
            self.history.nodes.append(np.asarray(nodes, dtype=float).copy())
        if self.record_fluid_slice:
            self.history.fluid_slices.append(self._midplane_speed())


def capture_continuous_motion(
    sim: Any,
    *,
    t_end: Optional[float] = None,
    n_steps: Optional[int] = None,
    callback: Callback = None,
    video_path: Optional[str | Path] = None,
    video_amplify: float = 5.0,
    fps: int = 12,
) -> ContinuousMotionCapture:
    """Convenience: run continuous motion on a finished form-find sim."""
    cap = ContinuousMotionCapture(sim)
    cap.run(t_end=t_end, n_steps=n_steps, callback=callback)
    if video_path is not None:
        path, kind = cap.write_video(
            video_path, fps=fps, amplify=1.0
        )
        print(f"[continuous_motion] video ({kind}) → {path}")
        if float(video_amplify) != 1.0:
            amp_path = Path(video_path)
            amp_path = amp_path.with_name(
                amp_path.stem + "_amplified" + amp_path.suffix
            )
            path_a, kind_a = cap.write_video(
                amp_path,
                fps=fps,
                amplify=float(video_amplify),
                title=f"Formed membrane in fluid flow (×{float(video_amplify):g})",
            )
            print(f"[continuous_motion] amplified video ({kind_a}) → {path_a}")
    return cap
