# Cylinder Re=3900 — code pipeline

Self-contained folder: staggered-grid **PISO + Smagorinsky LES** for flow past a circular cylinder at \(\mathrm{Re}=U D/\nu=3900\).

Independent of the membrane FSI code at the repo root.

---

## 1. How to run

```bash
cd cylinder_re3900
python3 run.py --quick              # coarse grid, short time + GIF
python3 run.py                      # config.yaml defaults
python3 run.py --t-end 50 --gif-fps 8
python3 run.py --no-gif             # skip GIF
```

Config: `config.yaml` (`physical`, `domain`, `grid`, `time`, `les`, `quick`).

Outputs (under `output/`):

| File | Content |
|------|---------|
| `force_history.csv` / `.png` | \(C_D\), \(C_L\) vs \(tU/D\) |
| `wake_tXXXX.XX.png` | Mid-span speed + \(\omega_z\) |
| `cylinder_wake.gif` | Animated wake |
| `final_state.npz` | Final staggered fields |

---

## 2. Module pipeline (call graph)

```text
config.yaml
    │
    ▼
run.py ─────────────────────────────┐
    │                               │
    ├─ mesh.FluidGrid               │  MAC staggered box
    ├─ cylinder.Cylinder            │  mask + surface band
    ├─ piso.FluidSolver             │  NS + PISO + LES
    │       └─ les.smagorinsky_…    │  ν_t from staggered |S|
    └─ time loop                    │
            ├─ fluid.step(dt)       │
            ├─ force_coefficients   │
            ├─ viz.plot_midplane    │
            └─ viz.save_wake_gif    ▼
```

| Module | Role |
|--------|------|
| `mesh.py` | Grid; `p` at centres `(nx,ny,nz)`; `u,v,w` on faces |
| `cylinder.py` | Solid cells \((x-c_x)^2+(y-c_y)^2\le R^2\) (extruded in \(z\)) |
| `les.py` | Staggered \(S_{ij}\) → \(\nu_t\); `cell_centered_velocity` for plots/CFL only |
| `piso.py` | BCs, momentum, PISO correctors, IB faces, forces |
| `viz.py` | Mid-plane plots + GIF |
| `run.py` | Driver / I/O |

---

## 3. Setup pipeline (`run.build_case`)

1. Read YAML → \(D\), \(U_\infty\), \(\mathrm{Re}\) → \(\nu=U D/\mathrm{Re}\).
2. Build `FluidGrid(L,W,H,nx,ny,nz)`.
3. Place cylinder at `(cx,cy)`; build solid `mask` and near-surface `band`.
4. Construct `FluidSolver` (ρ, ν, \(U_\infty\), LES on/off, \(C_s\), PISO correctors).
5. `set_immersed_boundary(mask)` — zero velocity on solid-adjacent faces.
6. Tiny random seed on `v` so the wake can break symmetry.

---

## 4. Per-step fluid pipeline (`FluidSolver.step`)

MAC unknowns: face \(u,v,w\), centre \(p\).

```text
uⁿ, vⁿ, wⁿ, pⁿ
        │
        ▼
[1] LES  (les.py) — face velocities stay on faces
        S_xx,S_yy,S_zz  ← face jumps at centres
        shears          ← face derivatives → average scalars to centres
        |S| = √(2 S_ij S_ij)
        ν_eff = ν + (C_s Δ)² |S|     (at centres)
        ν_u,ν_v,ν_w ← average ν_eff onto faces
        │
        ▼
[2] Momentum predictor
        H(u) = −(u·∇)u + ν_face ∇²u   (upwind + Laplacian on faces)
        u* = uⁿ + Δt (H(u) − ∇pⁿ/ρ)   (∇p on faces)
        apply BC + IB
        │
        ▼
[3] PISO correctors  (n_correctors times, default 2)
        if corrector > 1: rebuild H from latest u*, re-predict
        ∇·u* at centres  (face flux divergence)
        solve  ∇² p' = (ρ/Δt) ∇·u*
        u* ← u* − (Δt/ρ) ∇p'         (face gradient of p')
        p  ← p + p'
        apply BC + IB
        │
        ▼
uⁿ⁺¹, vⁿ⁺¹, wⁿ⁺¹, pⁿ⁺¹, ν_eff
```

### Boundary conditions (external flow)

| Boundary | Condition |
|----------|-----------|
| Inlet (−x) | \(u=U_\infty\), \(v=w=0\) |
| Outlet (+x) | Neumann |
| Far y | Freestream / slip |
| Span z | Slip (short-span stand-in for periodic) |
| Cylinder | IB: solid faces → 0 |

---

## 5. Outer driver pipeline (`run.main` time loop)

```text
for step = 1 … N:
    state ← fluid.step(dt)
    Cd, Cl ← pressure on surface band
    log CSV; print on save_interval
    every plot_interval: write wake_t*.png
write force_history.png
write cylinder_wake.gif from wake frames
write final_state.npz
```

---

## 6. Field locations (staggered)

```text
        v (nx, ny+1, nz)          w (nx, ny, nz+1)
              │                         │
              ▼                         ▼
     ┌──── u ────┬──── u ────┐   p, ν_eff at cell centre
     │           │           │   (nx, ny, nz)
     │     p     │     p     │
     │           │           │
     └───────────┴───────────┘
         u: (nx+1, ny, nz)
```

- **Do not** average \(u,v,w\) for the LES strain (uses face operators).
- **Do** use `cell_centered_velocity()` only for mid-plane plots, CFL, and speed stats.

---

## 7. Physical defaults (`config.yaml`)

| Quantity | Default |
|----------|---------|
| \(\mathrm{Re}\) | 3900 |
| \(D\), \(U_\infty\) | 1, 1 → \(\nu=1/3900\) |
| Domain | \(\sim 20D \times 12D \times \pi D\) |
| Grid | \(128\times 80\times 24\) (quick: \(48\times 32\times 8\)) |
| LES | Smagorinsky \(C_s=0.1\), 2 PISO correctors |
