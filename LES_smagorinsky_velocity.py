
#no need for this module in the present notebook, just checking if the modules created can be imported in the present notebook

import Inputs
import mesh

import __future__
print(__future__.all_feature_names) #checking __future__ modulefrom __future__ import annotations

import numpy as np
from mesh import FluidGrid

def smagorinsky_viscosity(
     u: np.ndarray,
     v:np.ndarray,
     w:np.ndarray,
     grid:FluidGrid,
     Cs: float =0.17,
     nu: float= 1.5e-5,
)-> np.ndarray:
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        # function of all these parameters to give strain rate magnitude |S| : |S|= srt(2 S_ij S_ij); nu_t = (Cs delta)^2 |S|
        dudx = np.gradient(u,dx,axis=0,edge_order=1) #edge order is the first order accuracy here, come back to see the significance of this
        dudy = np.gradient(u,dy,axis=1,edge_order=1)
        dudz = np.gradient(u,dz, axis=2,edge_order=1)
        dvdx = np.gradient(v,dx,axis=0,edge_order=1)
        dvdy = np.gradient(v,dy,axis=1, edge_order=1)
        dvdz = np.gradient(v,dz, axis=2,edge_order=1)
        dwdx = np.gradient(w,dx,axis=0,edge_order=1)
        dwdy = np.gradient(w,dy, axis=1, edge_order=1)
        dwdz = np.gradient(w,dz,axis=2,edge_order=1)
        
        S2= (2.0 * dudx**2 + 2.0* dvdy**2 + 2.0* dwdz**2 + (dudy+ dvdx)**2 + (dudz +dwdx)**2 + (dvdz + dwdy)**2)
        
        S_mag = np.sqrt(np.maximum(S2, 0.0))
        delta = grid.delta()
        nu_t = (Cs *delta)**2* S_mag
        
        return nu + nu_t
