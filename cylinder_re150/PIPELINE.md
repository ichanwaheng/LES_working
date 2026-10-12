# Cylinder Re=150 — code pipeline

Self-contained folder: staggered-grid **2-D laminar PISO** for flow past a circular cylinder at \(\mathrm{Re}=U D/\nu=150\).

Independent of `cylinder_re3900` and the membrane FSI code.

---

## 1. How to run

```bash
cd cylinder_re150
python3 run.py --quick
python3 run.py
python3 run.py --t-end 120 --gif-fps 8
python3 run.py --no-gif

# GIF only (from existing wake_t*.png)
python3 make_gif.py --demo
python3 make_gif.py --run-quick --demo
```

---

## 2. Module pipeline

```text
config.yaml  (mesh.n_bl, mesh.growth [, mesh.h_far])
    │
    ▼
mesh_sizing.BoundaryLayerSizing
    δ=D/√Re → Δ₁…Δ_max → line spacings at cylinder walls
    │
    ▼
run.py
    ├─ mesh.FluidGrid.from_boundary_layer   stretched MAC box
    ├─ cylinder.Cylinder                    mask + surface band
    ├─ piso.FluidSolver                     NS + PISO (variable Δ)
    └─ time loop
            ├─ fluid.step(dt)
            ├─ force_coefficients
            ├─ viz.plot_wake
            └─ validate.validate_history  → Cd, St vs NASA targets
```

---

## 3. Per-step fluid pipeline

```text
uⁿ, vⁿ, pⁿ
   │
   ▼
[1] Momentum predictor (explicit)
       H(u) = −(u·∇)u + ν ∇²u   (central or upwind convection)
       u* = uⁿ + Δt (H(u) − ∇pⁿ/ρ)
       apply BC + IB
   │
   ▼
[2] PISO correctors
       rebuild H if corrector > 1
       ∇·u* → solve ∇² p' = (ρ/Δt) ∇·u*
       u* ← u* − (Δt/ρ) ∇p'
       p  ← p + p'
   │
   ▼
uⁿ⁺¹, vⁿ⁺¹, pⁿ⁺¹
```

---

## 4. Validation

After the run, `validate.py` uses the last `settle_fraction` of the history:

- mean \(C_D\) vs 1.78
- Strouhal from \(C_L\) peak period (FFT fallback) vs 0.179–0.182

Writes `output/validation_report.json`.
