


from __future__ import annotations
from typing import Tuple
import numpy as np

import PISO
import mesh # chcek this module again if it can be generalised
from geometry import element_areas_normals


# In[ ]:


def element_centroids(nodes: np.ndarray, elements:np.ndarray)-> np.ndarray:
    return nodes[elements].mean(axis=1)
def dynamic_pressure_loads(
    fluid:FluidSolver,
    mesh:MembraneMesh,
    nodes: np.ndarray,
    rho:float,
    U_ref: float,
) -> Tuple[np.ndarray, np.ndarray]:
    
    centroids = element_centroids(nodes, mesh.elements)
    areas, normals= element_areas_normals(nodes, mesh.elements)
    vel, p_field = fluid.sample_at(centroids)
    
    Un = np.sum(vel * normals, axis=1)
    q_dyn = 0.5 * rho * Un * np.abs(Un)
    p_inf = 0.0
    pressure = q_dyn + 0.3 *( p_field - p_inf)
    
    q_ref = 0.5 *rho * max(U_ref, 1e-6)**2
    pressure = np.clip(pressure, -5* q_ref, 5*q_ref)
    
    f_nodal = np. zeros_like(nodes)
    for e, (a,b,c) in enumerate(mesh.elements):
        Fe = (pressure[e] * areas[e]/3.0)* normals[e]
        f_nodal[a] +=Fe
        f_nodal[b] += Fe
        f_nodal[c]+= Fe
    return pressure, f_nodal

def pressure_jump_loads(
    fluid:FluidSolver,
    mesh: MembraneMesh,
    nodes: np.ndarray,
    rho: float,
    U_ref: float,
    offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    centroids = element_centroids(nodes, mesh.elements)
    areas, normals = element_areas_normals(nodes, mesh.elements)
    _,p_plus = fluid.sample_at(centroids + offset * normals)
    _, p_minus = fluid. sample_at(centroids - offset * normals)
    pressure = p_minus -p_plus
    
    q_ref = 0.5*rho * max(U_ref, 1e-6)**2
    pressure = np.clip(pressure, -5 * q_ref, 5*q_ref)
    
    f_nodal = np.zeros_like(nodes)
    for e, (a,b,c) in enumerate(mesh.elements):
        Fe = (pressure[e] * areas[e]/ 3.0) * normals[e]
        f_nodal[a] += Fe
        f_nodal[b] += Fe
        f_nodal[c] += Fe
    return pressure, f_nodal

def interpolated_field_loads(
    fluid:FluidSolver,
    mesh: MembraneMesh,
    nodes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    
    centroids = element_centroids(nodes, mesh.elements)
    areas, normals = element_areas_normals(nodes, mesh.elements)
    _, p_field = fluid.sample_at(centroids)
    pressure = p_field - float(np.mean(p_field))
    f_nodal = np.zeros_like (nodes)
    
    for e, (a,b,c) in enumerate( mesh.elements):
        Fe = (pressure[e] * areas[e] / 3.0) * normals[e]
        f_nodal[a] +=Fe
        f_nodal[b]+= Fe
        f_nodal[c] += Fe
        
    return pressure, f_nodal

        

