


from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


# In[2]:


@dataclass
class MembraneMesh:
    nodes: np.ndarray
    elements: np.ndarray  # <-- ADD THIS MISSING FIELD HERE
    fixed: np.ndarray
    length: float = 0.0
    width: float = 0.0
    nx: int = 0
    ny: int = 0
    thickness: float = 0.001
    density: float = 1200.0

    areas0: np.ndarray = field(default_factory = lambda: np.zeros(0))
    normals0: np. ndarray = field(default_factory= lambda: np.zeros((0,3)))

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.elements = np.asarray(self.elements, dtype = int)
        self.fixed = np. asarray(self.fixed, dtype= bool)
        self.areas0, self.normals0 = element_areas_normals(self.nodes, self.elements)

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    def free_dofs(self) -> np.ndarray:
        return np.where(~self.fixed)[0]

    def update_geometry(self, nodes: np. ndarray) -> Tuple [ np. ndarray, np.ndarray]:
        return element_areas_normals(nodes, self.elements)
        


# In[3]:


def element_areas_normals(
    nodes: np.ndarray, elements: np.ndarray
) -> Tuple [np.ndarray, np.ndarray]:

    p0 = nodes[elements[:,0]]
    p1 = nodes[elements[:,1]]
    p2 = nodes[elements[:,2]]
    cross = np. cross(p1-p0, p2-p0)
    norms = np. linalg.norm(cross, axis=1)
    areas = 0.5* norms
    normals = np.zeros_like(cross)
    mask = norms > 1e-14
    normals[mask] = cross[mask] / norms[mask, None]
    
    return areas, normals 

    


# In[4]:


def build_rectangular_membrane(
    length:float = 2.0,
    width: float = 1.5,
    nx:int = 24,
    ny: int = 24,
    z0: float =0.0,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    fixed_edges: Sequence[str] = ("left", "right"),
    thickness:float = 0.001,
    density: float = 1200.0,
) -> MembraneMesh:
    ox, oy, oz = origin
    xs = np. linspace(0.0, length, nx+1)
    ys = np. linspace( 0.0, width, ny+1)
    X, Y = np.meshgrid(xs, ys, indexing = "ij")
    Z = np.full_like( X, oz+ z0)
    nodes = np. column_stack([X.ravel() + ox, Y.ravel() + oy, Z. ravel()])

    def vid(i: int, j:int) -> int:
        return i * (ny+1) +j

    tris: List[List[int]] = []
    for i in range (nx):
        for j in range (ny):
            n00, n10 = vid(i,j), vid ( i+1,j)
            n01, n11 = vid(i, j+1), vid(i+1, j+1)

            tris.append([n00, n10, n11])
            tris.append([n00, n11,n01])

    elements = np.asarray(tris, dtype=int)

    fixed = np. zeros(nodes.shape[0], dtype = bool)
    edge_set = { e.lower() for e in fixed_edges}
    for i in range(nx+1):
        for j in range(ny+1):
            idx = vid(i,j)
            if "left" in edge_set and i==0:
                fixed[idx] = True
            if "right" in edge_set and i==nx:
                fixed[idx] = True
            if "bottom" in edge_set and j ==0:
                fixed[idx] = True
            if "top" in edge_set and j ==ny:
                fixed[idx] = True

    return MembraneMesh(
        nodes=nodes,
        elements=elements,
        fixed=fixed,
        length=length,
        width=width,
        nx=nx,
        ny=ny,
        thickness=thickness,
        density=density,
    )
            


# In[5]:


def nodal_mass_lumped(mesh:MembraneMesh) -> np.ndarray:
    area1 = mesh.density * mesh.thickness
    mass = np.zeros(mesh.n_nodes)
    for e, (a,b,c) in enumerate( mesh.elements):
        m = area1 * mesh.areas0[e] /3.0
        mass[a] += m
        mass[b] +=m
        mass[c] +=m
    mass = np. maximum(mass, 1e-12)
    return mass


# In[ ]:




