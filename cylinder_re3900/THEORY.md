# Theory and Numerical Method: LES–PISO Flow Past a Circular Cylinder at Re = 3900

This document describes the mathematical model and numerical algorithm used in the `cylinder_re3900` simulation. It is written so that an independent implementation can be reproduced from the theory alone. No source code is included.

---

## 1. Problem statement

### 1.1 Physical problem

Simulate three-dimensional, unsteady, incompressible turbulent flow of a Newtonian fluid past a circular cylinder of diameter \(D\) at Reynolds number

\[
\mathrm{Re}_D = \frac{U_\infty D}{\nu} = 3900,
\]

where \(U_\infty\) is the freestream speed and \(\nu\) is the kinematic viscosity. This Reynolds number lies in the shear-layer transition regime: the wake is turbulent and three-dimensional, while the attached boundary layers on the cylinder remain laminar. Large-eddy simulation (LES) is used to resolve the large energetic eddies and to model the effect of unresolved scales through an eddy viscosity.

### 1.2 Computational domain

The fluid occupies a rectangular box

\[
\Omega = [0,L]\times[0,W]\times[0,H].
\]

Default geometric choices (in units of \(D\)):

| Quantity | Symbol | Typical value | Role |
|----------|--------|---------------|------|
| Streamwise length | \(L\) | \(20\,D\) | inlet → outlet |
| Cross-stream width | \(W\) | \(12\,D\) | far-field \(y\) |
| Spanwise height | \(H\) | \(\pi D\) | cylinder axis / span |
| Cylinder centre | \((c_x,c_y)\) | \((5D,\,6D)\) | axis parallel to \(z\) |
| Cylinder radius | \(R=D/2\) | \(0.5\,D\) | solid obstacle |

The cylinder is an infinite (extruded) circular cylinder: every \((x,y)\) inside the disk of radius \(R\) is solid for all \(z\).

### 1.3 Inputs that fully define a run

A complete case is specified by:

1. **Physical parameters:** density \(\rho\), freestream speed \(U_\infty\), diameter \(D\), Reynolds number \(\mathrm{Re}_D\) (hence \(\nu = U_\infty D/\mathrm{Re}_D\)).
2. **Domain:** \(L,W,H\) and cylinder centre \((c_x,c_y)\).
3. **Grid:** integers \(n_x,n_y,n_z\) (number of pressure cells in each direction).
4. **Time:** step \(\Delta t\), end time \(t_{\mathrm{end}}\) (often reported as convective time \(t U_\infty/D\)).
5. **LES:** Smagorinsky constant \(C_s\) (default \(0.1\)), on/off flag.
6. **PISO:** number of pressure–velocity correctors \(N_c\) (default \(2\)).
7. **Initial fields:** usually uniform freestream \(u=U_\infty\), \(v=w=0\), \(p=0\), plus a tiny random perturbation on \(v\) to break symmetry so vortex shedding can start.
8. **Outputs:** force coefficients, mid-span velocity/vorticity visualisations, optional animation.

With the nondimensional choice \(D=1\), \(U_\infty=1\), one has \(\nu=1/3900\).

---

## 2. Governing equations: incompressible Navier–Stokes

Before any LES modelling, the continuum equations for a constant-density Newtonian fluid are the incompressible Navier–Stokes (NS) equations.

### 2.1 Continuity (mass)

\[
\nabla\cdot\mathbf{u} = 0,
\qquad
\mathbf{u}=(u,v,w).
\]

### 2.2 Momentum

\[
\frac{\partial\mathbf{u}}{\partial t}
+ (\mathbf{u}\cdot\nabla)\mathbf{u}
= -\frac{1}{\rho}\nabla p
+ \nu\nabla^2\mathbf{u}.
\]

Component form:

\[
\begin{aligned}
\frac{\partial u}{\partial t}
+ u\frac{\partial u}{\partial x}
+ v\frac{\partial u}{\partial y}
+ w\frac{\partial u}{\partial z}
&= -\frac{1}{\rho}\frac{\partial p}{\partial x}
+ \nu\nabla^2 u,\\[0.5em]
\frac{\partial v}{\partial t}
+ u\frac{\partial v}{\partial x}
+ v\frac{\partial v}{\partial y}
+ w\frac{\partial v}{\partial z}
&= -\frac{1}{\rho}\frac{\partial p}{\partial y}
+ \nu\nabla^2 v,\\[0.5em]
\frac{\partial w}{\partial t}
+ u\frac{\partial w}{\partial x}
+ v\frac{\partial w}{\partial y}
+ w\frac{\partial w}{\partial z}
&= -\frac{1}{\rho}\frac{\partial p}{\partial z}
+ \nu\nabla^2 w.
\end{aligned}
\]

**Why these equations?** Constant density and low Mach number justify incompressibility. Pressure is not thermodynamic; it is a Lagrange multiplier that enforces \(\nabla\cdot\mathbf{u}=0\).

**What is unknown?** The velocity \(\mathbf{u}(\mathbf{x},t)\) and pressure \(p(\mathbf{x},t)\).

---

## 3. Large-eddy simulation (LES) and the Smagorinsky model

### 3.1 Why LES?

Direct numerical simulation (DNS) of all scales at \(\mathrm{Re}=3900\) is expensive. LES resolves only scales larger than a filter width \(\Delta\) comparable to the grid, and models the effect of smaller eddies on the resolved motion.

Formally, a spatial filter \(\overline{(\cdot)}\) applied to the NS equations yields equations for the resolved velocity \(\overline{\mathbf{u}}\). The nonlinear term produces an unclosed residual stress

\[
\tau_{ij}
= \overline{u_i u_j}-\overline{u}_i\overline{u}_j,
\]

which must be modelled.

### 3.2 Eddy-viscosity closure

The Smagorinsky model treats the residual stress like a Newtonian viscous stress with an eddy (turbulent) viscosity \(\nu_t\):

\[
\tau_{ij}-\tfrac13\tau_{kk}\delta_{ij}
\approx -2\nu_t\,\overline{S}_{ij},
\]

where the resolved strain-rate tensor is

\[
\overline{S}_{ij}
= \frac12\left(
\frac{\partial\overline{u}_i}{\partial x_j}
+\frac{\partial\overline{u}_j}{\partial x_i}
\right).
\]

Its magnitude is

\[
|\overline{S}|
= \sqrt{2\,\overline{S}_{ij}\overline{S}_{ij}}.
\]

### 3.3 Smagorinsky eddy viscosity

\[
\nu_t = (C_s\Delta)^2\,|\overline{S}|.
\]

The filter width on a Cartesian cell is taken as the geometric mean of the spacings,

\[
\Delta = (\Delta x\,\Delta y\,\Delta z)^{1/3}.
\]

The constant \(C_s\) is an input (here \(C_s=0.1\) for the cylinder case).

### 3.4 Effective viscosity in the momentum equation

Substituting the model into the filtered momentum equation gives the same form as NS, but with molecular viscosity replaced by

\[
\nu_{\mathrm{eff}} = \nu + \nu_t.
\]

Thus the resolved equations solved numerically are

\[
\begin{aligned}
\nabla\cdot\overline{\mathbf{u}} &= 0,\\
\frac{\partial\overline{\mathbf{u}}}{\partial t}
+ (\overline{\mathbf{u}}\cdot\nabla)\overline{\mathbf{u}}
&= -\frac{1}{\rho}\nabla\overline{p}
+ \nabla\cdot\bigl(\nu_{\mathrm{eff}}\,\nabla\overline{\mathbf{u}}\bigr)
\approx -\frac{1}{\rho}\nabla\overline{p}
+ \nu_{\mathrm{eff}}\nabla^2\overline{\mathbf{u}},
\end{aligned}
\]

where the last step uses the implementation’s constant-within-a-cell treatment of \(\nu_{\mathrm{eff}}\) in the viscous term (Laplacian with face-interpolated \(\nu_{\mathrm{eff}}\)).

Overlines are dropped below: \(\mathbf{u}\) and \(p\) mean **resolved** LES fields.

### 3.5 Discrete evaluation of \(|S|\) on the staggered grid

Velocities live on faces (Section 4). Strain components are **not** formed by first averaging all velocities to cell centres.

**Normal strains at pressure (cell) centres** from face jumps:

\[
S_{xx}=\frac{u_{i+1/2}-u_{i-1/2}}{\Delta x},\quad
S_{yy}=\frac{v_{j+1/2}-v_{j-1/2}}{\Delta y},\quad
S_{zz}=\frac{w_{k+1/2}-w_{k-1/2}}{\Delta z}.
\]

**Shear strains:** evaluate \(\partial u/\partial y\), \(\partial u/\partial z\) on \(u\)-faces (central differences in the face-tangential directions; one-sided at domain edges), and likewise \(\partial v/\partial x\), \(\partial v/\partial z\) on \(v\)-faces and \(\partial w/\partial x\), \(\partial w/\partial y\) on \(w\)-faces. Average those derivative fields from faces to the adjacent cell centre, then

\[
S_{xy}=\tfrac12\Bigl(\tfrac{\partial u}{\partial y}+\tfrac{\partial v}{\partial x}\Bigr)_c,
\quad
S_{xz}=\tfrac12\Bigl(\tfrac{\partial u}{\partial z}+\tfrac{\partial w}{\partial x}\Bigr)_c,
\quad
S_{yz}=\tfrac12\Bigl(\tfrac{\partial v}{\partial z}+\tfrac{\partial w}{\partial y}\Bigr)_c.
\]

**Magnitude:**

\[
|S|
= \sqrt{
2(S_{xx}^2+S_{yy}^2+S_{zz}^2)
+4(S_{xy}^2+S_{xz}^2+S_{yz}^2)
}.
\]

Then \(\nu_t=(C_s\Delta)^2|S|\) and \(\nu_{\mathrm{eff}}=\nu+\nu_t\) are stored at cell centres. For the viscous term on faces, \(\nu_{\mathrm{eff}}\) is linearly interpolated from the two adjacent cells onto each velocity face.

**Stability clipping (implementation detail that must be reproduced):** \(\nu_{\mathrm{eff}}\) is limited from below by \(\nu\), from above by a multiple of \(\nu\), and also by an explicit viscous stability bound of the form

\[
\nu_{\mathrm{eff}}
\le
\frac{C_{\nu}}{\Delta t\bigl(\Delta x^{-2}+\Delta y^{-2}+\Delta z^{-2}\bigr)},
\]

with \(C_{\nu}\approx 0.15\) in this solver, so that the explicit viscous update remains stable.

---

## 4. Spatial discretisation: MAC staggered Cartesian grid

### 4.1 Why staggered?

On a collocated grid, central pressure gradients and velocity divergences can admit odd–even decoupling (checkerboard pressure). The Marker-and-Cell (MAC) staggered arrangement places each velocity component on the face normal to that component and pressure at the cell centre. Then:

- \(\nabla\cdot\mathbf{u}\) at centres uses face fluxes directly;
- \(\nabla p\) on faces uses neighbouring cell pressures;

so the discrete gradient and divergence are adjoint (up to boundary terms), which strongly reduces checkerboarding.

### 4.2 Mesh

Divide \([0,L]\times[0,W]\times[0,H]\) into \(n_x\times n_y\times n_z\) equal cells:

\[
\Delta x=\frac{L}{n_x},\quad
\Delta y=\frac{W}{n_y},\quad
\Delta z=\frac{H}{n_z}.
\]

Cell-centre coordinates:

\[
x_i=(i+\tfrac12)\Delta x,\quad
y_j=(j+\tfrac12)\Delta y,\quad
z_k=(k+\tfrac12)\Delta z,
\]

with \(i=0,\ldots,n_x-1\), etc.

Face coordinates:

- \(u\)-faces at \(x=i\Delta x\), \(i=0,\ldots,n_x\) → array size \((n_x+1)\times n_y\times n_z\);
- \(v\)-faces at \(y=j\Delta y\), \(j=0,\ldots,n_y\) → size \(n_x\times(n_y+1)\times n_z\);
- \(w\)-faces at \(z=k\Delta z\), \(k=0,\ldots,n_z\) → size \(n_x\times n_y\times(n_z+1)\);
- pressure / \(\nu_{\mathrm{eff}}\): size \(n_x\times n_y\times n_z\).

### 4.3 Discrete divergence (cell centres)

\[
(\nabla\cdot\mathbf{u})_{i,j,k}
=
\frac{u_{i+1,j,k}-u_{i,j,k}}{\Delta x}
+
\frac{v_{i,j+1,k}-v_{i,j,k}}{\Delta y}
+
\frac{w_{i,j,k+1}-w_{i,j,k}}{\Delta z}.
\]

Inside solid cells this is set to zero (immersed boundary).

### 4.4 Discrete pressure gradient (faces)

For interior \(u\)-faces \(i=1,\ldots,n_x-1\):

\[
\Bigl(\frac{\partial p}{\partial x}\Bigr)_{i,j,k}
=
\frac{p_{i,j,k}-p_{i-1,j,k}}{\Delta x},
\]

and analogously for \(v\)- and \(w\)-faces. Boundary faces that carry Dirichlet velocity are not corrected by pressure in the same way (inlet \(u\) is prescribed).

### 4.5 Convection: first-order upwind

For a scalar \(\phi\) advected by a velocity component \(c\) along a grid direction with spacing \(h\),

\[
c\frac{\partial\phi}{\partial\xi}
\approx
\begin{cases}
c\,(\phi_m-\phi_{m-1})/h & c\ge 0,\\
c\,(\phi_{m+1}-\phi_m)/h & c< 0.
\end{cases}
\]

**Why upwind?** It is dissipative and stabilises explicit advection at finite CFL. **Trade-off:** more numerical diffusion than central or higher-order schemes.

On staggered faces, the three advecting velocities at a \(u\)-face are: \(u\) itself, and \(v,w\) interpolated from neighbouring \(v\)- and \(w\)-faces to that \(u\)-face (and cyclic permutations for \(v\)- and \(w\)-momentum).

The convective residual for \(u\) is

\[
\bigl((\mathbf{u}\cdot\nabla)u\bigr)
=
u\partial_x u + v_u\partial_y u + w_u\partial_z u,
\]

with \(v_u,w_u\) the interpolants onto the \(u\)-face.

### 4.6 Viscous term: seven-point Laplacian

On each velocity component’s staggered array,

\[
\nabla^2\phi
\approx
\frac{\phi_{i+1}-2\phi_i+\phi_{i-1}}{\Delta x^2}
+
\frac{\phi_{j+1}-2\phi_j+\phi_{j-1}}{\Delta y^2}
+
\frac{\phi_{k+1}-2\phi_k+\phi_{k-1}}{\Delta z^2},
\]

with one-sided (copy / Neumann-like) formulae at the array edges. Multiplied by the face value of \(\nu_{\mathrm{eff}}\).

### 4.7 Explicit momentum residual without pressure

Define, on each face family,

\[
\mathbf{H}(\mathbf{u})
=
-(\mathbf{u}\cdot\nabla)\mathbf{u}
+ \nu_{\mathrm{eff}}\nabla^2\mathbf{u}.
\]

Pressure is treated separately in the PISO projection (Section 6).

---

## 5. Time integration: explicit or implicit?

### 5.1 Answer in one sentence

**Momentum advection and diffusion are advanced explicitly** (forward Euler using residuals at the known time level). **Pressure is obtained implicitly** by solving a Poisson equation each corrector so that the updated velocity is (approximately) divergence-free.

### 5.2 Explicit momentum step

Given fields at time \(t^n\), the provisional velocity (predictor) is

\[
\mathbf{u}^{*}
=
\mathbf{u}^{n}
+ \Delta t\left(
\mathbf{H}(\mathbf{u}^{n})
-\frac{1}{\rho}\nabla p^{n}
\right).
\]

No linear system is solved for \(\mathbf{u}^{*}\). That is classical **explicit** (forward Euler) treatment of \(\mathbf{H}\).

**Consequences:**

- Time step is limited by CFL for advection,

\[
\mathrm{CFL}
=
\frac{U_{\max}\Delta t}{\min(\Delta x,\Delta y,\Delta z)}
\lesssim O(1)
\]

  (a soft monitor value \(\approx 0.5\) is used as guidance), and by viscous stability when \(\nu_{\mathrm{eff}}\) is large (hence the \(\nu_{\mathrm{eff}}\) cap above).

- Implementation is simple; accuracy in time is first order.

### 5.3 Implicit pressure Poisson

Enforcing \(\nabla\cdot\mathbf{u}^{n+1}=0\) leads to

\[
\nabla^2 p'
=
\frac{\rho}{\Delta t}\,\nabla\cdot\mathbf{u}^{*},
\]

solved as a sparse linear system (conjugate gradient, with a direct sparse fallback). This solve is **implicit** in \(p'\). Velocity is then corrected explicitly using \(\nabla p'\).

So the overall scheme is: **explicit momentum + implicit pressure projection** (a fractional-step / projection family method), organised as PISO.

---

## 6. PISO algorithm (Issa)

PISO (Pressure Implicit with Splitting of Operators) improves a single projection by repeating pressure–velocity corrections and, after the first corrector, rebuilding the explicit residual \(\mathbf{H}\) from the latest velocity (neighbour / flux update).

Let \(N_c\) be the number of correctors (default \(2\)).

### 6.1 Step A — LES viscosities

From \(\mathbf{u}^{n}\), compute \(\nu_{\mathrm{eff}}\) at cell centres (Section 3) and interpolate to faces.

### 6.2 Step B — Momentum predictor

\[
\mathbf{u}^{*}
=
\mathbf{u}^{n}
+ \Delta t\left(
\mathbf{H}(\mathbf{u}^{n})
-\frac{1}{\rho}\nabla p^{n}
\right),
\]

then apply boundary conditions and the immersed-boundary condition (Section 7).

### 6.3 Step C — Corrector loop (\(m=1,\ldots,N_c\))

1. **If \(m>1\)** (PISO neighbour update): recompute

\[
\mathbf{u}^{*}
=
\mathbf{u}^{n}
+ \Delta t\left(
\mathbf{H}(\mathbf{u}^{*}_{\mathrm{prev}})
-\frac{1}{\rho}\nabla p
\right)
\]

   using the latest corrected velocity inside \(\mathbf{H}\), and re-apply BCs/IB.  
   (Here \(p\) is the pressure accumulated so far in the step.)

2. **Divergence:** \(d=\nabla\cdot\mathbf{u}^{*}\) at cell centres (zero in solids).

3. **Pressure-correction Poisson:**

\[
\nabla^2 p' = \frac{\rho}{\Delta t}\,d.
\]

   Discrete Laplacian: standard 7-point stencil with homogeneous Neumann treatment at domain faces (missing neighbour omitted). The null space of Neumann Poisson is removed by pinning one cell (\(p'=0\) at a reference cell) and removing the mean of the right-hand side. After the solve, subtract the mean of \(p'\).

4. **Velocity correction** (on faces):

\[
\mathbf{u}^{*}
\leftarrow
\mathbf{u}^{*}
- \frac{\Delta t}{\rho}\nabla p'.
\]

5. **Pressure update:**

\[
p \leftarrow p + p'.
\]

6. Re-apply BCs and IB to \(\mathbf{u}^{*}\).

### 6.4 Step D — Accept new time level

\[
\mathbf{u}^{n+1}=\mathbf{u}^{*},\qquad
p^{n+1}=p-\langle p\rangle
\]

(mean pressure set to zero; only gradients matter for incompressible flow).

### 6.5 Why PISO rather than a single projection?

A single fractional step uses one Poisson solve with \(\mathbf{H}(\mathbf{u}^{n})\) frozen. Extra correctors with an updated \(\mathbf{H}(\mathbf{u}^{*})\) reduce splitting error between the momentum residual and the divergence-free constraint, which is the defining idea of PISO relative to a pure projection method.

---

## 7. Boundary conditions and immersed cylinder

### 7.1 Far-field / domain boundaries

| Location | Condition | Meaning |
|----------|-----------|---------|
| Inlet \(x=0\) | \(u=U_\infty\), \(v=0\), \(w=0\) | uniform freestream |
| Outlet \(x=L\) | \(\partial_x(u,v,w)=0\) (copy from interior) | convective / Neumann outflow |
| \(y=0\) and \(y=W\) | \(u=U_\infty\), \(v=0\), \(\partial_y w\approx 0\) | freestream / slip sides |
| \(z=0\) and \(z=H\) | \(\partial_z u\approx 0\), \(\partial_z v\approx 0\), \(w=0\) | slip span ends (surrogate for periodic span) |

These are **not** viscous channel walls; the setup is an external flow past a cylinder in a truncated box.

### 7.2 Immersed boundary (cylinder)

Solid cells: all pressure cells whose centre \((x_i,y_j)\) satisfies

\[
(x_i-c_x)^2+(y_j-c_y)^2 \le R^2
\]

for every \(k\).

Any velocity face that borders at least one solid cell is treated as solid and set to zero (stationary cylinder):

\[
u=v=w=0 \quad\text{on solid-adjacent faces}.
\]

Divergence in solid cells is forced to zero so those cells do not drive the Poisson right-hand side.

This is a simple discrete-forcing / masking immersed-boundary approach: geometry is stair-stepped on the Cartesian mesh; fidelity improves as \(\Delta x,\Delta y\) decrease relative to \(D\).

### 7.3 Field sanitation

After updates, non-finite values are cleared and velocities are clipped to a large bound (order \(8\,U_\infty\)) to prevent rare blow-ups on coarse grids.

---

## 8. Force coefficients

Drag and lift coefficients are estimated from pressure on a thin band of **fluid** cells just outside the cylinder:

\[
C_D=\frac{F_x}{\tfrac12\rho U_\infty^2\,D\,H},\qquad
C_L=\frac{F_y}{\tfrac12\rho U_\infty^2\,D\,H}.
\]

With outward unit normal in the \(xy\)-plane \(\mathbf{n}=(n_x,n_y)\) from the cylinder centre through each band cell, and cell area scale \(A\sim\min(\Delta x,\Delta y)\,\Delta z\),

\[
\mathbf{F}
\approx
\sum_{\text{band}} (-p)\,A\,\mathbf{n}.
\]

Viscous surface stress is **not** included in this estimate; it is a pressure-dominated approximation suitable for monitoring, not a high-precision force balance.

---

## 9. Initialisation

1. Set all \(u\)-faces to \(U_\infty\), all \(v\)- and \(w\)-faces to \(0\), pressure to \(0\).
2. Apply BCs and IB.
3. Add a very small random perturbation to \(v\) (amplitude \(\sim 10^{-3}U_\infty\)) so that the symmetric wake can bifurcate into vortex shedding.
4. Advance with the algorithm of Section 6 until \(t=t_{\mathrm{end}}\).

---

## 10. End-to-end algorithm (reproduction checklist)

For each time step \(\Delta t\):

1. **Inputs known:** \(\mathbf{u}^{n}\), \(p^{n}\), grid, \(\rho\), \(\nu\), \(C_s\), \(N_c\), solid mask.
2. **LES:** compute staggered \(|S|\), \(\nu_t\), \(\nu_{\mathrm{eff}}\) at centres; interpolate \(\nu_{\mathrm{eff}}\) to faces; apply stability clips.
3. **Predictor (explicit):** \(\mathbf{u}^{*}=\mathbf{u}^{n}+\Delta t(\mathbf{H}(\mathbf{u}^{n})-\nabla p^{n}/\rho)\); apply BC/IB.
4. **For** \(m=1\) to \(N_c\):
   - if \(m>1\): rebuild \(\mathbf{H}(\mathbf{u}^{*})\), re-predict from \(\mathbf{u}^{n}\) with current \(p\), apply BC/IB;
   - form \(d=\nabla\cdot\mathbf{u}^{*}\);
   - solve \(\nabla^2 p'=(\rho/\Delta t)\,d\) (implicit);
   - \(\mathbf{u}^{*}\leftarrow\mathbf{u}^{*}-(\Delta t/\rho)\nabla p'\), \(p\leftarrow p+p'\);
   - apply BC/IB.
5. **Store** \(\mathbf{u}^{n+1},\ p^{n+1}\); optionally compute \(C_D,C_L\) and mid-span diagnostics.

**Time integration summary**

| Term | Treatment |
|------|-----------|
| Convection \((\mathbf{u}\cdot\nabla)\mathbf{u}\) | Explicit (in \(\mathbf{H}\)) |
| Viscous \(\nu_{\mathrm{eff}}\nabla^2\mathbf{u}\) | Explicit (in \(\mathbf{H}\)) |
| Old pressure \(\nabla p^{n}\) in predictor | Explicit |
| Pressure correction \(p'\) | Implicit Poisson |
| Overall | Explicit momentum + PISO projection |

---

## 11. Quantities to monitor for a correct implementation

- **Mass conservation:** \(\max|\nabla\cdot\mathbf{u}|\) in fluid cells should drop after each projection (boundaries/IB can leave local residuals).
- **CFL:** keep \(\mathrm{CFL}\) moderate; reduce \(\Delta t\) if the solution diverges.
- **Wake physics at Re = 3900:** after a few convective times, expect a separated wake, shear layers, and (on adequate grids and longer times) unsteady shedding; mean \(C_D\) on fine LES grids is classically near order \(1\) (coarse grids over-predict drag).
- **Energy of \(\nu_t\):** \(\nu_t\) should be small in quiet freestream and larger in shear layers/wake.

---

## 12. Default numerical parameters (reference case)

| Parameter | Value |
|-----------|--------|
| \(\mathrm{Re}_D\) | \(3900\) |
| \(D\), \(U_\infty\), \(\rho\) | \(1,\ 1,\ 1\) |
| \(\nu\) | \(1/3900\) |
| Domain | \(20\times 12\times\pi\) (in units of \(D\)) |
| Grid (production) | \(128\times 80\times 24\) |
| \(\Delta t\) | \(0.002\) |
| \(t_{\mathrm{end}}\) | \(20\) (convective times) |
| \(C_s\) | \(0.1\) |
| PISO correctors \(N_c\) | \(2\) |
| Convection scheme | 1st-order upwind |
| Momentum time scheme | Forward Euler (explicit) |
| Pressure | Implicit Poisson (CG) |

---

## 13. Conceptual flowchart

```text
Inputs (Re, domain, grid, Δt, Cs, Nc)
        │
        ▼
Incompressible NS  →  filter / LES  →  Smagorinsky ν_t
        │
        ▼
MAC staggered discretisation (faces u,v,w; centre p)
        │
        ▼
Each time step:
   explicit H(u) [convection + diffusion]
   + PISO: Poisson for p' (implicit) + velocity corrections
        │
        ▼
BC + immersed cylinder mask
        │
        ▼
Outputs: fields, Cd, Cl, wake diagnostics
```

---

## 14. References (methods)

- Harlow & Welch (1965): staggered MAC grid for incompressible flow.
- Smagorinsky (1963): eddy-viscosity closure for unresolved scales.
- Issa (1986): PISO — pressure-implicit with splitting of operators.
- Standard projection / fractional-step ideas (Chorin; later variants): divergence-free enforcement via Poisson pressure.

This document matches the mathematical structure of the `cylinder_re3900` solver: filtered incompressible NS, Smagorinsky LES on a staggered strain-rate evaluation, explicit face momentum residuals, and an Issa-type PISO pressure–velocity coupling with an immersed circular cylinder at \(\mathrm{Re}=3900\).
