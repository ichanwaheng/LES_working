# Laminar cylinder at Re = 150 (NASA Study #1)

Self-contained **2-D incompressible PISO** solver for laminar flow past a circular cylinder at

\[
\mathrm{Re}_D = U_\infty D / \nu = 150.
\]

This case mirrors the NASA GRC Wind-US validation page:

- https://www.grc.nasa.gov/www/wind/valid/lamcyl/Study1_files/Study1.html

Independent of `cylinder_re3900` (3-D LES) and of the membrane FSI code at the repo root.

## Physics

At Re = 150 the wake is essentially **two-dimensional** with periodic vortex shedding. Validation quantities:

| Quantity | Symbol | Experimental target | Source |
|----------|--------|---------------------|--------|
| Mean drag | \(C_D\) | ≈ 1.78 | Schlichting (cited by NASA) |
| Strouhal | \(S_t = f D / U\) | 0.179 – 0.182 | Roshko NACA 1191 |

NASA Wind-US runs A–D reported \(C_D \approx 1.68\)–\(1.80\) and \(S_t \approx 0.15\)–\(0.18\) depending on time-marching options.

## How to run

```bash
cd cylinder_re150
python3 run.py --quick              # coarse grid, short time + GIF
python3 run.py                      # config.yaml defaults (longer)
python3 run.py --t-end 120 --gif-fps 8
python3 run.py --no-gif
```

Config: `config.yaml` (`physical`, `domain`, `grid`, `time`, `solver`, `validation`, `quick`).

## Outputs

| File | Content |
|------|---------|
| `output/force_history.csv` / `.png` | \(C_D\), \(C_L\) vs \(t U/D\) |
| `output/validation_report.json` | mean \(C_D\), Strouhal vs NASA targets |
| `output/wake_tXXXX.XX.png` | speed + vorticity |
| `output/cylinder_wake.gif` | animated wake |
| `output/final_state.npz` | final staggered fields |
| `demo/cylinder_wake.gif` | tracked demo copy |

## Modules

| File | Role |
|------|------|
| `mesh.py` | 2-D MAC staggered box |
| `cylinder.py` | immersed cylinder mask + surface band |
| `piso.py` | laminar NS + PISO (no LES) |
| `validate.py` | mean \(C_D\) + Strouhal from \(C_L\) peaks / FFT |
| `viz.py` | wake plots + GIF |
| `run.py` | driver / I/O |

## Method (short)

- Incompressible 2-D Navier–Stokes, constant \(\nu = U D / \mathrm{Re}\)
- MAC staggered grid; **central** convection by default (upwind optional — more dissipative)
- Explicit momentum + Issa PISO pressure–velocity coupling (implicit Poisson)
- Immersed-boundary cylinder (stair-stepped zero-velocity faces)
- Forces from pressure on a thin fluid band; \(S_t\) from lift peak period

Central convection is important at Re = 150: first-order upwind on coarse grids adds enough numerical diffusion to suppress vortex shedding.

## References

1. NASA GRC, *Laminar Flow over a Circular Cylinder: Study #1* (Wind-US validation).
2. Roshko, A., *On the Development of Turbulent Wakes from Vortex Streets*, NACA Report 1191, 1954.
3. Schlichting, H., *Boundary-Layer Theory*.
4. Issa, R. I., *Solution of the implicitly discretised fluid flow equations by operator-splitting*, J. Comput. Phys., 1986.
