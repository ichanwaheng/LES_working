


from __future__ import annotations
import numpy as np

import geometry


# In[ ]:


def initial_sag_shape(
    mesh:MembraneMesh,
    sag: float = 0.05,
) -> np.ndarray:
    x= mesh.nodes[:,0]
    y= mesh.nodes[:,1]
    x0, y0 = x.min(), y.min()
    L,W = mesh.length, mesh.width
    xi = (x-x0)/ max(L, 1e-12)
    eta = (y-y0)/ max(W, 1e-12)
    dz = -sag * np.sin(np.pi *xi) * np.sin(np.pi *eta)
    nodes = mesh.nodes.copy()
    nodes[:,2] += dz
    return nodes
    

