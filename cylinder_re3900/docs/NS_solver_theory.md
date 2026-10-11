# From Navier–Stokes to the Discrete Solver

## Explicit momentum and PISO pressure projection on a staggered grid

This note explains, from continuum theory to the fully discrete algorithm, how the cylinder flow solver advances the incompressible Navier–Stokes equations. No source code is included. The goal is that a reader can understand *what* is solved, *why* each step exists, and *how* time and space are discretised.

---

## 1. Continuum starting point: incompressible Navier–Stokes

For a constant-density Newtonian fluid the unknowns are the velocity field \(\mathbf{u}=(u,v,w)\) and the pressure \(p\). They satisfy

**Mass (continuity)**

\[
\nabla\cdot\mathbf{u}=0.
\]

**Momentum**

\[
\frac{\partial\mathbf{u}}{\partial t}
+(\mathbf{u}\cdot\nabla)\mathbf{u}
=-\frac{1}{\rho}\nabla p
+\nu\nabla^{2}\mathbf{u}.
\]

Here \(\rho\) is density and \(\nu\) is kinematic viscosity. Pressure is not an independent thermodynamic equation of state: it is determined so that the velocity remains divergence-free.

In components,

\[
\begin{aligned}
\partial_{t}u+u\partial_{x}u+v\partial_{y}u+w\partial_{z}u
&=-\frac{1}{\rho}\partial_{x}p+\nu\nabla^{2}u,\\
\partial_{t}v+u\partial_{x}v+v\partial_{y}v+w\partial_{z}v
&=-\frac{1}{\rho}\partial_{y}p+\nu\nabla^{2}v,\\
\partial_{t}w+u\partial_{x}w+v\partial_{y}w+w\partial_{z}w
&=-\frac{1}{\rho}\partial_{z}p+\nu\nabla^{2}w.
\end{aligned}
\]

These are the equations before any turbulence modelling.

---

## 2. Large-eddy simulation (LES)

At Reynolds number \(\mathrm{Re}_{D}=U_{\infty}D/\nu=3900\), not all scales are resolved on an affordable grid. LES evolves only a spatially filtered (resolved) velocity \(\overline{\mathbf{u}}\). Filtering the nonlinear term produces an unclosed residual stress. The Smagorinsky model closes it with an eddy viscosity \(\nu_{t}\):

\[
\overline{S}_{ij}
=\frac12\Bigl(
\partial_{j}\overline{u}_{i}+\partial_{i}\overline{u}_{j}
\Bigr),
\qquad
|\overline{S}|=\sqrt{2\,\overline{S}_{ij}\overline{S}_{ij}},
\]

\[
\nu_{t}=(C_{s}\Delta)^{2}\,|\overline{S}|,
\qquad
\Delta=(\Delta x\,\Delta y\,\Delta z)^{1/3}.
\]

The resolved momentum equation then has the same structure as Navier–Stokes with an effective viscosity

\[
\nu_{\mathrm{eff}}=\nu+\nu_{t}.
\]

Below, overbars are dropped: \(\mathbf{u}\) and \(p\) mean the **resolved** LES fields, and \(\nu\) in the viscous term is replaced by \(\nu_{\mathrm{eff}}\) wherever LES is active.

**Why LES here?** The large wake eddies are intended to be resolved; \(\nu_{t}\) only represents the drain of energy to unresolved scales.

---

## 3. Spatial discretisation: MAC staggered grid

### 3.1 Arrangement of unknowns

The domain is a uniform Cartesian box divided into \(n_{x}\times n_{y}\times n_{z}\) cells of size \(\Delta x,\Delta y,\Delta z\).

- Pressure \(p\) (and \(\nu_{\mathrm{eff}}\)) live at **cell centres**.
- \(u\) lives on faces normal to \(x\) (one more plane in \(x\) than the number of cells).
- \(v\) lives on faces normal to \(y\).
- \(w\) lives on faces normal to \(z\).

This Marker-and-Cell (MAC) staggering is chosen so that the discrete divergence of velocity and the discrete gradient of pressure act between neighbouring locations and largely avoid checkerboard pressure modes that appear on collocated grids.

### 3.2 Discrete divergence (at cell centres)

\[
(\nabla\cdot\mathbf{u})_{i,j,k}
=\frac{u_{i+1/2}-u_{i-1/2}}{\Delta x}
+\frac{v_{j+1/2}-v_{j-1/2}}{\Delta y}
+\frac{w_{k+1/2}-w_{k-1/2}}{\Delta z}.
\]

### 3.3 Discrete pressure gradient (on faces)

On an interior \(u\)-face,

\[
(\partial_{x}p)_{i+1/2}
=\frac{p_{i+1}-p_{i}}{\Delta x},
\]

and likewise for \(y\) and \(z\) faces.

### 3.4 Convection: first-order upwind

The convective operator \((\mathbf{u}\cdot\nabla)\phi\) is approximated by first-order upwind differences: when the advecting speed is positive, the backward difference is used; when negative, the forward difference. On staggered faces, the three advecting components at a \(u\)-face are \(u\) itself and \(v,w\) interpolated from neighbouring faces (cyclic for \(v\)- and \(w\)-momentum).

**Why upwind?** It stabilises explicit advection. **Cost:** added numerical diffusion compared with central schemes.

### 3.5 Diffusion: seven-point Laplacian

Viscous terms use the standard second-order central Laplacian of each velocity component on its own staggered array, multiplied by the face value of \(\nu_{\mathrm{eff}}\) (obtained by averaging neighbouring cell-centre viscosities onto that face).

### 3.6 Explicit momentum residual without pressure

Collect convection and diffusion into

\[
\mathbf{H}(\mathbf{u})
=-(\mathbf{u}\cdot\nabla)\mathbf{u}
+\nu_{\mathrm{eff}}\nabla^{2}\mathbf{u}.
\]

Pressure is handled separately by the projection / PISO step below.

### 3.7 Staggered evaluation of \(|S|\) for LES

Normal strains \(S_{xx},S_{yy},S_{zz}\) are face jumps of velocity at cell centres. Shear strains are formed from face-tangential derivatives of the face velocities, then those derivative fields are averaged to centres. One does **not** average all three velocity components to the centre first and then differentiate for the LES closure used here.

---

## 4. Time integration method

### 4.1 What is used?

| Part of the equation | Time treatment |
|----------------------|----------------|
| Convection | **Explicit** |
| Viscous diffusion | **Explicit** |
| Old pressure in the predictor | **Explicit** |
| Pressure correction | **Implicit** (Poisson solve) |

Overall: **forward Euler (explicit) momentum** combined with an **implicit pressure Poisson** inside a **PISO** (Pressure Implicit with Splitting of Operators) corrector loop.

This is **not** Crank–Nicolson, Runge–Kutta, or a fully implicit coupled NS solve.

### 4.2 Explicit forward Euler for momentum

Given fields at time \(t^{n}\), the provisional velocity is

\[
\mathbf{u}^{*}
=\mathbf{u}^{n}
+\Delta t\Biggl(
\mathbf{H}(\mathbf{u}^{n})
-\frac{1}{\rho}\nabla p^{n}
\Biggr).
\]

All terms on the right are known from time level \(n\). No linear system is solved for \(\mathbf{u}^{*}\). That is classical **explicit** integration, first-order accurate in time.

**Stability implication.** The step \(\Delta t\) must respect a CFL restriction for advection,

\[
\mathrm{CFL}=\frac{U_{\max}\Delta t}{\min(\Delta x,\Delta y,\Delta z)},
\]

and a viscous restriction when \(\nu_{\mathrm{eff}}\) is large (the implementation also caps \(\nu_{\mathrm{eff}}\) for that reason).

### 4.3 Why pressure cannot stay fully explicit

If one advanced velocity with an arbitrary pressure and never corrected it, \(\nabla\cdot\mathbf{u}\) would drift. Incompressibility requires that the end-of-step velocity satisfy

\[
\nabla\cdot\mathbf{u}^{n+1}=0.
\]

That constraint determines a pressure correction through an elliptic equation, solved **implicitly**.

---

## 5. From continuous projection idea to discrete PISO

### 5.1 Projection idea

Write the update as a provisional velocity plus a pressure correction:

\[
\mathbf{u}^{n+1}
=\mathbf{u}^{*}
-\frac{\Delta t}{\rho}\nabla p'.
\]

Taking the divergence and requiring \(\nabla\cdot\mathbf{u}^{n+1}=0\) yields the Poisson problem

\[
\nabla^{2}p'
=\frac{\rho}{\Delta t}\,\nabla\cdot\mathbf{u}^{*}.
\]

Solve for \(p'\), then correct \(\mathbf{u}^{*}\). This is the heart of fractional-step / projection methods.

### 5.2 PISO organisation (Issa)

PISO repeats the pressure–velocity correction \(N_{c}\) times (typically \(N_{c}=2\)). After the first corrector, the explicit residual \(\mathbf{H}\) is rebuilt from the latest velocity (neighbour / flux update) before forming a new provisional field and solving Poisson again. That reduces splitting error relative to a single projection with \(\mathbf{H}(\mathbf{u}^{n})\) frozen.

### 5.3 Discrete algorithm for one time step

**A. LES viscosities**

From \(\mathbf{u}^{n}\), compute \(|S|\), \(\nu_{t}\), \(\nu_{\mathrm{eff}}\) at centres; interpolate \(\nu_{\mathrm{eff}}\) to faces.

**B. Momentum predictor (explicit)**

\[
\mathbf{u}^{*}
=\mathbf{u}^{n}
+\Delta t\Bigl(
\mathbf{H}(\mathbf{u}^{n})-\tfrac{1}{\rho}\nabla p^{n}
\Bigr),
\]

then impose boundary conditions and the immersed-cylinder condition (solid-adjacent faces set to zero velocity).

**C. Corrector loop** \(m=1,\ldots,N_{c}\)

1. If \(m>1\): recompute \(\mathbf{H}\) from the current \(\mathbf{u}^{*}\), form a new provisional velocity from \(\mathbf{u}^{n}\) using the pressure accumulated so far, and re-apply boundary/immersed conditions.
2. Evaluate \(d=\nabla\cdot\mathbf{u}^{*}\) at cell centres (set to zero in solid cells).
3. Solve the discrete Poisson equation
   \[
   \nabla^{2}p'=\frac{\rho}{\Delta t}\,d
   \]
   with a seven-point Laplacian, Neumann-type treatment at the outer boundary, and a pinned reference cell to remove the additive null space. This solve is **implicit**.
4. Correct face velocities:
   \[
   \mathbf{u}^{*}
   \leftarrow
   \mathbf{u}^{*}
   -\frac{\Delta t}{\rho}\nabla p'.
   \]
5. Update pressure: \(p\leftarrow p+p'\).
6. Re-apply boundary and immersed conditions.

**D. Accept the new time level**

\[
\mathbf{u}^{n+1}=\mathbf{u}^{*},\qquad
p^{n+1}=p-\langle p\rangle
\]

(mean pressure removed; only \(\nabla p\) matters).

---

## 6. Boundary and obstacle treatment (conceptual)

- **Inlet:** prescribed freestream velocity.
- **Outlet:** homogeneous Neumann (zero streamwise derivative) on velocity.
- **Far lateral boundaries:** freestream / slip-type conditions appropriate to external flow (not a viscous channel).
- **Cylinder:** Cartesian immersed mask; faces touching solid cells carry zero velocity (stationary body).

These conditions are enforced after the predictor and after each velocity correction so that the discrete fields remain consistent with the geometry.

---

## 7. What “solving the discretised NS equations” means in practice

Putting the pieces together:

1. Start from continuum incompressible NS.
2. Replace \(\nu\) by \(\nu_{\mathrm{eff}}(\mathbf{u})\) from Smagorinsky LES.
3. Replace derivatives by staggered finite differences (upwind convection, central Laplacian, face–centre divergence and gradient).
4. Advance time by **explicit forward Euler** on the momentum residual \(\mathbf{H}-\nabla p/\rho\).
5. Enforce incompressibility by **implicitly** solving a pressure Poisson equation and correcting the face velocities, repeated in a **PISO** loop.

So the discrete system is not solved as one giant fully implicit nonlinear system for \((\mathbf{u}^{n+1},p^{n+1})\). It is solved as a **split** scheme: cheap explicit momentum update, then elliptic pressure solves that restore discrete mass conservation.

---

## 8. One-line summary

**Discretisation:** MAC staggered finite differences with first-order upwind convection and central viscous fluxes, plus Smagorinsky \(\nu_{t}\).  
**Time integration:** explicit forward Euler for momentum; implicit Poisson pressure corrections in a PISO loop.
