#!/usr/bin/env python
# coding: utf-8

# In[1]:


simulation = {
    "name": "membrane_fsi",
    "output_dir": "output",
    "save_interval": 10,
    "plot": True
}

time = {
    "dt": 0.002,
    "t_end": 2.0,
    "cfl_max": 0.5
}

# Membrane geometric parameters
membrane = {
    "length": 2.0,
    "width": 1.5,
    "nx": 24,
    "ny": 18,
    "thickness": 0.001,
    "density": 1200.0,
    "youngs_modulus": 5.0e+8,
    "poisson": 0.3,
    "prestress": 5.0e+4,
    "damping": 80.0,
    "mass_scale": 80.0,
    "fixed_edges": ["left", "right", "bottom", "top"],
    
    "fluid": {
        "domain": {
            "L": 8.0,
            "W": 3.0,
            "H": 2.5
        },
        "nx": 48,
        "ny": 24,
        "nz": 20,
        "rho": 1.225,
        "nu": 1.0e-3,
        "U_inlet": 5.0,
        "membrane_x0": 2.0,
        "membrane_y0": 0.75,
        "membrane_z0": 1.0
    },
    
    "fsi": {
        "scheme": "serial_staggered",
        "under_relaxation": 0.5,
        "max_subiters": 2,
        "residual_tol": 1.0e-3,
        "load_scale": 0.25,
        "load_mode": "dynamic_pressure"
    },
    
    "les": {
        "enabled": True,
        "Cs": 0.17
    }
}




