


from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from mesh_update import update_immsersed_boundary as update_immersed_boundary



# In[13]:


from mesh import FluidGrid
from PISO import FluidSolver



# In[15]:


import sys
from pathlib import Path

# Fallback safely to current working directory if running interactively
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import FSI_load_transfer
from FSI_load_transfer import(
    dynamic_pressure_loads,
    interpolated_field_loads,
    pressure_jump_loads,
)
from mesh_update import under_relax, update_immsersed_boundary
from geometry import build_rectangular_membrane
from materials import MembraneMaterial
from prestress import initial_sag_shape

try:
    from .uwm import updated_weight_form_find
except ImportError:
    from UWM import updated_weight_form_find
    


# In[17]:


@dataclass
class QuasiStaticHistory:
    iteration: List[int] = field(default_factory=list)
    max_disp: List[float] = field(default_factory=list)
    shape_residual: List[float] = field(default_factory=list)
    uwm_residual: List[float] = field(default_factory =list)
    pressure_max: List[float] = field(default_factory=list)
    cfl: List[float] = field(default_factory=list)

class QuasiStaticFSI:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        mcfg = cfg["membrane"]
        fcfg = cfg["fluid"]
        qcfg = cfg.get("quasi_static", cfg.get("fsi", {}))
        les = cfg.get ("les", {})
        tcfg = cfg.get("time",{})

        origin=(
            fcfg["membrane_x0"],
            fcfg["membrane_y0"],
            fcfg["membrane_z0"],
        )
        # Membrane length/width/nx/ny/thickness/density/fixed_edges come from
        # geometry.build_rectangular_membrane defaults, not from YAML.
        self.mesh = build_rectangular_membrane(origin=origin)
        sag = float(qcfg.get("initial_sag", 0.03))
        self.mesh.nodes = initial_sag_shape(self.mesh, sag=sag)

        self.x_bc = self.mesh.nodes.copy()

        material = MembraneMaterial(
            E= float(mcfg["youngs_modulus"]),
            nu= float(mcfg["poisson"]),
            thickness = float(self.mesh.thickness),
            prestress = float(mcfg["prestress"]),
        )
        self.N_pre = float(material.N_pre)

        ff = updated_weight_form_find(
            self.mesh.nodes,
            self.mesh.elements,
            self. mesh.fixed,
            N_pre = self.N_pre,
            f_ext = None,
            support_nodes = self.x_bc,
            max_weight_updates = int(qcfg.get("uwm_weight_updates", 20)),
            tol = float(qcfg.get("uwm_tol", 1e-7)),
        )
        self.nodes = ff.nodes
        self.mesh.nodes = self.nodes.copy()

        self.grid = FluidGrid(
            L= float(fcfg["domain"]["L"]),
            W= float(fcfg["domain"]["W"]),
            H = float( fcfg["domain"]["H"]),
            nx = int (fcfg["nx"]),
            ny= int(fcfg["ny"]),
            nz = int(fcfg["nz"]),
        )
        self.fluid = FluidSolver(
            self.grid,
            rho=float(fcfg["rho"]),
            nu= float(fcfg["nu"]),
            U_inlet = float(fcfg["U_inlet"]),
            use_les = bool(les.get("enabled", True)),
            Cs = float(les.get("Cs", 0.17)),
            gust_amp = float(fcfg.get("gust_amp", 0.0)),
            gust_freq = float(fcfg.get("gust_freq", 1.0)),
            u_clip = float(fcfg["u_clip"]) if "u_clip" in fcfg else None,
            n_correctors = int(les.get("piso_correctors", fcfg.get("piso_correctors",2))),
        )
        update_immersed_boundary(
            self.fluid,
            self.grid,
            self.mesh,
            self.nodes,
            np.zeros_like(self.nodes),
        )

        self.dt = float(tcfg.get("dt", 0.002))
        self.fluid_substeps = int(qcfg.get("fluid_substeps",40))
        self.max_iters = int(qcfg.get("max_iters", 15))
        self.shape_tol = float(qcfg.get("shape_tol", 1e-3))
        self.alpha_shape = float(qcfg.get("under_relaxation", 0.5))
        self.alpha_load = float(qcfg.get("load_relaxation", 0.5))
        self.load_scale = float( qcfg.get("load_scale", 0.5))
        self.load_mode = qcfg.get("load_mode", "pressure_jump")
        self.uwm_weight_updates = int(qcfg.get("uwm_weight_updates", 20))
        self.uwm_tol = float(qcfg.get("uwm_tol", 1e-7))
        self.uwm_relax = float(qcfg.get("uwm_relaxation", 1.0))

        self.history = QuasiStaticHistory()
        self._f_old: Optional[np.ndarray] = None
        self.iteration = 0
        self.time =0.0
        self.t_end = float(tcfg.get("t_end", self.max_iters * self.fluid_substeps * self.dt))

    def _compute_loads(self):
        if self.load_mode == "interpolated_field":
            pressure, f_nodal = interpolated_field_loads(
                self.fluid, self.mesh, self.nodes
            )
        elif self.load_mode == "pressure_jump":
            offset = 3.0 * self.grid.dz
            pressure, f_nodal = pressure_jump_loads(
                self.fluid, 
                self.mesh,
                self.nodes,
                rho = float(self.cfg["fluid"]["rho"]),
                U_ref = float(self.cfg["fluid"]["U_inlet"]),
                offset = offset,
            )

        else:
            pressure, f_nodal = dynamic_pressure_loads(
                self.fluid,
                self.mesh,
                self.nodes,
                rho = float(self.cfg["fluid"]["rho"]),
                U_ref = float(self.cfg["fluid"]["U_inlet"]),
            )
        return pressure, f_nodal * self.load_scale

    def step(self) -> Dict[str, float]:
        x_prev = self.nodes.copy()

        update_immersed_boundary(
            self.fluid,
            self.grid,
            self.mesh,
            self.nodes,
            np.zeros_like(self.nodes),
        )
        for _ in range(self.fluid_substeps):
            self.fluid.step(self.dt)

        pressure, f_nodal = self._compute_loads()

        if self._f_old is None:
            f_use = f_nodal
        else:
            alpha = max(self.alpha_load, 0.65)
            f_use = under_relax(f_nodal, self._f_old, alpha)
        self._f_old = f_use.copy()

        net_fz = float(np.sum(f_use[:,2]))
        uwm = updated_weight_form_find(
            self.nodes,
            self.mesh.elements,
            self.mesh.fixed,
            N_pre = self.N_pre,
            f_ext = f_use,
            support_nodes = self.x_bc,
            max_weight_updates = self.uwm_weight_updates,
            tol = self.uwm_tol,
            under_relaxation = self. uwm_relax,
        )
        x_new = under_relax(uwm.nodes, x_prev,self.alpha_shape)
        x_new[self.mesh.fixed] = self.x_bc[self.mesh.fixed]

        shape_res = float(
            np.linalg.norm(x_new - x_prev) / (np.linalg.norm(x_prev) + 1e-12)
        )

        self.nodes = x_new
        self.mesh.nodes = self.nodes.copy()
        mean_uz = float(
            np. mean(self.nodes[~self.mesh.fixed,2] - self.x_bc[~self.mesh.fixed,2])
        )
        self.iteration +=1
        self.time += self.fluid_substeps * self.dt
        info={
            "iteration": float(self.iteration),
            "time": float(self.time),
            "max_disp":float(
                np.max(np.linalg.norm(self.nodes - self.x_bc, axis =1))
            ),
            "shape_residual": shape_res,
            "uwm_residual" : float(uwm.residual),
            "uwm_updates" : float(uwm.n_weight_updates),
            "pressure_max": float(np.max(np.abs(pressure))),
            "net_fz": net_fz,
            "mean_uz": mean_uz,
            "cfl": float(self.fluid.max_cfl(self.dt)),
            "objective": float(uwm.objective),
        }
        self.history.iteration.append(self.iteration)
        self.history.max_disp.append(info["max_disp"])
        self.history.shape_residual.append(info["shape_residual"])
        self.history.uwm_residual.append(info["uwm_residual"])
        self.history.pressure_max.append(info["pressure_max"])
        self.history.cfl.append(info["cfl"])
        return info

    def run(self, callback=None) -> QuasiStaticHistory:
        for k in range(self.max_iters):
            info = self.step()
            if callback is not None:
                callback(self, info, k)
            if info["shape_residual"] < self.shape_tol:
                break
        return self.history

    def run_timed(self, t_end: Optional[float] = None, callback = None) -> QuasiStaticHistory:

        target = float(self.t_end if t_end is None else t_end)
        k=0
        while self. time < target - 1e-15:
            info = self.step()
            if callback is not None:
                callback(self, info,k)
            k+=1
            if k>= max(self.max_iters,1) *50:
                break
        return self.history
        
                
        
        
        


# In[ ]:




