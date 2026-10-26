# 3-D staggered solver (new files only)

Two extra files sit beside the 2-D solver. Nothing in `mesh_from_known_sizes.py`
or `ns_staggered_euler.py` is changed.

| File | Role |
|------|------|
| `mesh_from_known_sizes_3d.py` | extrudes the 2-D O–H staggered mesh into `nz` spanwise layers |
| `ns_staggered_euler_3d.py` | 3-D Griebel projection on that mesh, Smagorinsky `ν_eff` by default |

The xy section is the same known-sizes cylinder mesh. It is copied into `nz`
layers of thickness `dz = Lz / nz`. Spanwise faces are periodic (cylinder LES
convention). Cell index `cid = k · Nc_2d + c`.

## How to run

```bash
cd cylinder_re150

# smoke xy mesh × 4 spanwise cells, Smagorinsky, to t U/D = 40
python3 ns_staggered_euler_3d.py --quick --les --scheme central --nz 4 --demo

# mesh only
python3 mesh_from_known_sizes_3d.py --quick --nz 4 --Lz 3.1416

# laminar 3-D (molecular ν, no eddy viscosity)
python3 ns_staggered_euler_3d.py --quick --no-les --scheme central --nz 4

# high-Re flag (still uses the Re=150 xy sizes unless you rebuild with smaller d_min)
python3 ns_staggered_euler_3d.py --fine --les --Re 3900 --nz 8 --poisson cg
```

Outputs go to `output_ns3d/` (wake frames of the **mid-span** slice, GIF, forces,
`fields_*.npz`). Mesh: `mesh_output/staggered_3d_<preset>_nz<nz>.npz`.
`--demo` also copies the GIF to `demo/cylinder_wake_3d_smagorinsky.gif`.

Tracked demo copy: `demo/cylinder_wake_3d_smagorinsky.gif` (smoke mesh, mid-span
vorticity / velocity, Smagorinsky, last 20 convective units of a run to
`t U/D = 40`).

## Smagorinsky

`ν_t = (C_s Δ)² |S|`, `Δ = V^{1/3}`, `|S|` from Green–Gauss gradients of
`(u,v,w)`. `ν_eff` is interpolated to faces in the viscous term
`∇·(ν_eff ∇φ)`. Van Driest damping near the cylinder is on by default
(`--no-van-driest` to turn it off). Wall shear in Cd/Cl still uses molecular `ν`.

`--scheme central` is the LES-sensible convection blend (`γ = 0`).

## Limits

This is a 3-D method on the Re = 150 known-sizes xy mesh. A resolved LES at
Re = 3900 still needs a finer `d_min` (and typically more `nz`). For the
Cartesian 3-D LES already in the repo see `cylinder_re3900/`.
