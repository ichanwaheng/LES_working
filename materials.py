


from __future__ import annotations
from dataclasses import dataclass

import numpy as np

@dataclass
class MembraneMaterial:
    E: float = 5.0e8
    nu:float =0.3
    thickness:float = 0.001
    prestress: float = 5.0e4

    def plane_stress_matrix(self) -> np.ndarray:
        E, nu = self.E , self.nu
        factor = E/ (1.0 -nu**2)
        return factor * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5*( 1.0 -nu)],
            ]
        )

    @property
    def N_pre(self) -> float:
        return self.prestress * self.thickness


# In[ ]:




