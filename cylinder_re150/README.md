# Laminar cylinder at Re = 150 (NASA Study #1)

Self-contained 2-D incompressible Navier–Stokes solver for laminar flow past a
circular cylinder at

\[
\mathrm{Re}_D = U_\infty D / \nu = 150,
\]

on the O–H multiblock **staggered** mesh produced by `mesh_from_known_sizes.py`.
The time-stepping / projection method follows Griebel, Dornseifer & Neunhoeffer
(1998) as implemented in <https://github.com/CaptainProton42/NavierStokes>;
the operators are finite-volume on the unstructured quad faces of that mesh.

Reference case: NASA GRC Wind-US validation page
<https://www.grc.nasa.gov/www/wind/valid/lamcyl/Study1_files/Study1.html>
(experimental \(C_D \approx 1.78\), \(S_t = 0.179\)–\(0.182\)). The comparison
with our results is in `VALIDATION_NASA.md`.

## Setup

Domain 24 D × 12 D, cylinder of diameter D = 1 at (6 D, 6 D); uniform inlet at
x = 0, mass-balanced outlet at x = 24 D, no-slip top and bottom walls.

Mesh (fine preset): boundary-layer stack from `d_min = 0.015` growing by 1.15
up to `d_max = 0.12`, O-block ring + transition to a square around the cylinder,
Cartesian blocks elsewhere, ≈ 26 k cells. Cell size jumps at block interfaces
are kept ≤ 2.

Time step: \(\Delta t = \tau \min(h_{min}/|u|_{max},\; \mathrm{Re}\, h_{min}^2/4)\)
with \(\tau = 0.4\) (≈ 0.0045 on the fine mesh).

## How to run

```bash
cd cylinder_re150

# fine mesh, central convection, run to tU/D = 120, write frames + GIF + plots
python3 ns_staggered_euler.py --fine --scheme central --build-mesh --demo

# coarse smoke test (visible wake, no shedding)
python3 ns_staggered_euler.py --quick --scheme donor --steps 80 --demo

# continue a finished run (same mesh)
python3 ns_staggered_euler.py --fine --scheme central --t-end 240 \
        --restart output_ns/fields_central_fine.npz

# mesh only (prints the sizing report and all PASS/FAIL checks)
python3 mesh_from_known_sizes.py

# compare Cd, Cl, St of the stored runs against NASA
python3 tools/compare_nasa.py
```

Important flags of `ns_staggered_euler.py`:

| Flag | Meaning |
|------|---------|
| `--quick` / `--fine` | mesh preset (smoke / shedding-resolving) |
| `--build-mesh` | rebuild the mesh instead of reusing `mesh_output/staggered_*.npz` |
| `--scheme central|upwind|donor` | γ = 0 / γ = 1 / γ = max CFL face blend for convection |
| `--poisson lu|cg|sor` | pressure solver (SuperLU direct is the default) |
| `--t-end`, `--steps`, `--tau`, `--dt` | run length and time-step control |
| `--perturb` | amplitude of the near-wake seed that triggers shedding (0 = off) |
| `--frame-every`, `--gif-fps`, `--gif-from`, `--no-gif` | frame and GIF output |
| `--restart` | continue from a `fields_*.npz` |
| `--demo` | copy GIF / force plot / final frame into `demo/` |

## Outputs

| Path | Content |
|------|---------|
| `mesh_output/staggered_<tag>.npz` | cell / face locations and topology of the staggered mesh |
| `mesh_output/mesh_<tag>.{png,vtk,msh}` | mesh picture and exports |
| `output_ns/wake_<step>_t<time>.png` | velocity and vorticity plots per frame (`--out-dir` to choose another folder) |
| `output_ns/cylinder_wake_<scheme>.gif` | animated wake |
| `output_ns/forces_<scheme>_<preset>.png` | \(C_D\), \(C_L\) vs \(t U_\infty / D\) |
| `output_ns/fields_<scheme>_<preset>.npz` | final fields and force history (input for `--restart` and `tools/compare_nasa.py`) |
| `demo/` | tracked copies of GIF, plots, NASA comparison figures and the two PowerPoint decks |

## Files

| File | Role |
|------|------|
| `mesh_from_known_sizes.py` | mesher: sizes → O–H multiblock quads → checks → staggered extraction |
| `ns_staggered_euler.py` | solver: topology, Euler predictor, γ convection, projection, forces, plots, GIF, CLI |
| `tools/compare_nasa.py` | \(C_D\), \(C_L\), \(S_t\) of stored runs vs NASA Study #1, blockage extrapolation |
| `tools/make_tutorial_deck.py`, `tools/make_12_steps_deck.py`, `tools/pptkit.py` | PowerPoint decks in `demo/` |
| `THEORY_ns_staggered.md` | term-by-term discretisation of the equations as solved here |
| `THEORY_12_steps.md` | the method as twelve short steps |
| `VALIDATION_NASA.md` | comparison with NASA (two blockage ratios) |

## Method (short)

- Incompressible 2-D Navier–Stokes, constant \(\nu = U_\infty D / \mathrm{Re}\).
- Pressure and cell velocity at cell centres, normal velocity at face centres (MAC on quads).
- Explicit Euler predictor; convection by the Griebel face blend
  \(\phi_f = \tfrac12(\phi_o+\phi_n) - \tfrac12\gamma\,\mathrm{sign}(U_n)(\phi_n-\phi_o)\); central diffusion.
- Chorin / Griebel projection: Neumann Poisson for p, then face-flux correction (max |div| ≈ 1e-11).
- Forces by integrating pressure and wall shear on the cylinder faces; \(S_t\) from the lift signal.

Central convection matters at Re = 150: first-order upwind adds enough numerical
diffusion to damp the shedding.

## References

1. NASA GRC, *Laminar Flow over a Circular Cylinder: Study #1* (Wind-US validation).
2. Roshko, A., *On the Development of Turbulent Wakes from Vortex Streets*, NACA Report 1191, 1954.
3. Schlichting, H., *Boundary-Layer Theory*.
4. Griebel, M., Dornseifer, T., Neunhoeffer, T., *Numerical Simulation in Fluid Dynamics: A Practical Introduction*, SIAM, 1998.
