# Theory: from the Navier–Stokes equations to `ns_staggered_euler.py`

This note derives the discrete equations that `ns_staggered_euler.py` solves on
the staggered O–H multiblock mesh produced by `mesh_from_known_sizes.py`, term by
term, and then walks through exactly what happens in one time step and how the
plots are produced. Function names in `code font` refer to the implementation.

The time-stepping / projection structure follows Griebel, Dornseifer &
Neunhoeffer, *Numerical Simulation in Fluid Dynamics* (SIAM, 1998), as
implemented in <https://github.com/CaptainProton42/NavierStokes>. The spatial
operators are the finite-volume (FV) equivalents of their finite differences,
because our cells are not Cartesian.

---

## 1. Governing equations

Incompressible, constant-property, 2-D flow. With velocity
\(\mathbf{u}=(u,v)\), pressure \(p\), density \(\rho\) and kinematic viscosity
\(\nu\):

**Continuity (mass)**

\[
\nabla\cdot\mathbf{u} = \frac{\partial u}{\partial x}+\frac{\partial v}{\partial y} = 0
\tag{1}
\]

**Momentum (conservative form)**

\[
\frac{\partial \mathbf{u}}{\partial t}
+ \nabla\cdot(\mathbf{u}\otimes\mathbf{u})
= -\frac{1}{\rho}\nabla p + \nu\nabla^2\mathbf{u}
\tag{2}
\]

Written per component, this is what Griebel solve:

\[
\frac{\partial u}{\partial t}
+ \frac{\partial (u^2)}{\partial x}+\frac{\partial (uv)}{\partial y}
= -\frac{1}{\rho}\frac{\partial p}{\partial x}
+ \nu\left(\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2}\right)
\]
\[
\frac{\partial v}{\partial t}
+ \frac{\partial (uv)}{\partial x}+\frac{\partial (v^2)}{\partial y}
= -\frac{1}{\rho}\frac{\partial p}{\partial y}
+ \nu\left(\frac{\partial^2 v}{\partial x^2}+\frac{\partial^2 v}{\partial y^2}\right)
\]

Non-dimensionalising with the cylinder diameter \(D\) and free-stream speed
\(U_\infty\) gives \(\nu = U_\infty D / Re\); in the code `self.nu = U_inf*D/Re`
with \(D = U_\infty = \rho = 1\) and \(Re = 150\).

There is no evolution equation for \(p\). The pressure is whatever field is
needed to make the velocity satisfy (1) at every instant — this is what the
projection method exploits.

### Boundary conditions

| patch | velocity | pressure |
|---|---|---|
| `inlet` (x = 0) | \(u = U_\infty,\ v = v_{in}(t)\) (Dirichlet) | \(\partial p/\partial n = 0\) |
| `outlet` (x = 24 D) | \(\partial \mathbf{u}/\partial n = 0\), flux balanced | \(\partial p/\partial n = 0\) |
| `wall_top`, `wall_bottom`, `cylinder` | \(\mathbf{u} = 0\) (no-slip) | \(\partial p/\partial n = 0\) |

\(v_{in}(t) = \varepsilon U_\infty \sin(\pi t/T)\) for \(t<T=2\) is a small
pulse that breaks the exact mirror symmetry of the mesh (see §6).

---

## 2. The staggered finite-volume mesh

`mesh_from_known_sizes.py` builds a conforming multiblock quad mesh (O-ring
around the cylinder, transition block, Cartesian blocks upstream/downstream)
and `extract_staggered` exports it as a **MAC / staggered** arrangement:

* **cell centres** \(P\) — where \(p\) (and the cell-averaged \(u, v\)) live;
* **face centres** \(f\) — where the face-normal velocity \(U_n = \mathbf{u}\cdot\mathbf{n}\) lives;
* per face: unit normal \(\mathbf{n}_f\) (pointing from *owner* cell to
  *neighbour* cell), length \(L_f\), owner index, neighbour index (−1 on the
  boundary), boundary-patch id.

`build_topology` derives the quantities the discrete operators need:

* cell area \(V_P = \tfrac12\sum_{f\in P}(\mathbf{x}_f\cdot\mathbf{n}_f)\,L_f\)
  (divergence theorem applied to \(\mathbf{x}\));
* owner–neighbour distance \(d_{PN}=|\mathbf{x}_N-\mathbf{x}_P|\), and on
  boundary faces \(d = 2|\mathbf{x}_f-\mathbf{x}_P|\) so that \(d/2\) is the
  centre-to-face distance;
* `hmin` \(=\min_P\sqrt{V_P}\), the smallest cell size, which sets the time step.

Why staggered: storing \(U_n\) on faces and \(p\) in cells couples pressure
and velocity at adjacent locations, so the classic odd–even ("checkerboard")
pressure decoupling of collocated grids cannot occur, exactly as in the
Cartesian MAC scheme of Griebel.

### The basic identity: Gauss' theorem

Every spatial operator below is obtained by integrating over a cell and
converting the volume integral of a divergence into a sum over the cell's
faces:

\[
\int_{V_P}\nabla\cdot\mathbf{F}\,dV=\oint_{\partial V_P}\mathbf{F}\cdot\mathbf{n}\,dS
\;\;\approx\;\;\sum_{f\in P} s_{P,f}\,(\mathbf{F}_f\cdot\mathbf{n}_f)\,L_f
\tag{3}
\]

with \(s_{P,f}=+1\) if \(P\) owns the face and \(-1\) if \(P\) is its neighbour.
The helper `accumulate(flux, own, nei, Nc)` performs exactly this signed sum
for every cell at once (`np.add.at` on owners, subtract on neighbours).

---

## 3. Discretising each term

Throughout, \(\phi\) stands for either velocity component \(u\) or \(v\), and
\(\phi_f\) for its value interpolated to a face.

### 3.1 Time derivative — explicit (forward) Euler

\[
\frac{\partial\phi}{\partial t}\;\approx\;\frac{\phi^{n+1}_P-\phi^{n}_P}{\Delta t}
\tag{4}
\]

First-order, explicit: every right-hand-side term is evaluated with the
*known* field at level \(n\). This makes the momentum update a cheap explicit
formula but imposes the stability limit on \(\Delta t\) in §4.

### 3.2 Convective term — Griebel donor-cell / \(\gamma\) blend

Integrate \(\nabla\cdot(\mathbf{u}\phi)\) over the cell and apply (3):

\[
\big[\nabla\cdot(\mathbf{u}\phi)\big]_P\;\approx\;
\frac{1}{V_P}\sum_{f\in P} s_{P,f}\;U_{n,f}\,\phi_f\,L_f
\tag{5}
\]

\(U_{n,f}L_f\) is the volume flux through the face (known, on the staggered
face), and \(\phi_f\) is the transported quantity at the face. How \(\phi_f\)
is obtained from the two cell values \(\phi_P\) (owner) and \(\phi_N\)
(neighbour) is the whole difference between the schemes:

\[
\phi_f=\tfrac12(\phi_P+\phi_N)\;-\;\tfrac12\,\gamma\,\operatorname{sign}(U_{n,f})\,(\phi_N-\phi_P)
\tag{6}
\]

equivalently \(\phi_f=(1-\gamma)\,\phi_f^{\text{central}}+\gamma\,\phi_f^{\text{upwind}}\), where

* \(\phi_f^{\text{central}}=\tfrac12(\phi_P+\phi_N)\) — **central difference**,
  second-order, non-dissipative, but produces oscillations when the cell
  Péclet number \(U h/\nu\) exceeds ≈2;
* \(\phi_f^{\text{upwind}}=\phi_P\) if \(U_{n,f}\ge 0\) else \(\phi_N\) —
  **upwind / donor cell**, first-order, always bounded, but adds numerical
  viscosity of order \(\tfrac12|U|h\).

The code (`donor_cell_face`) implements (6) and the three CLI options are

| `--scheme` | \(\gamma\) | behaviour |
|---|---|---|
| `central` | 0 | pure central (used for the shedding runs) |
| `upwind` | 1 | pure donor cell |
| `donor` | \(\max(\text{CFL})=\max\!\big(|U|\Delta t/h_{\min}\big)\), capped at 1 | Griebel's adaptive blend (their default) |

This is the unstructured form of `integration.c` in the reference, where the
same \(\gamma\) blends \((uu)_x,\ (uv)_y,\ (uv)_x,\ (vv)_y\) on the Cartesian
grid. `_convective` evaluates (5) for both \(u\) and \(v\) and returns
\(-\nabla\cdot(\mathbf{u}\otimes\mathbf{u})\) (note the sign) so it can simply
be added to the predictor.

On boundary faces \(\phi_f\) is the Dirichlet value (\(U_\infty, 0\) at the
inlet; 0 on walls) and at the outlet it is the owner value (zero gradient).

### 3.3 Viscous term — central face gradients

Integrate \(\nu\nabla^2\phi=\nabla\cdot(\nu\nabla\phi)\) and apply (3):

\[
\big[\nu\nabla^2\phi\big]_P\;\approx\;
\frac{\nu}{V_P}\sum_{f\in P}s_{P,f}\,
\big(\nabla\phi\cdot\mathbf{n}\big)_f\,L_f,
\qquad
\big(\nabla\phi\cdot\mathbf{n}\big)_f\approx\frac{\phi_N-\phi_P}{d_{PN}}
\tag{7}
\]

The face-normal gradient is the two-point central difference between the
cell centres on either side — the same \((\phi_{i+1}-2\phi_i+\phi_{i-1})/\delta x^2\)
stencil as Griebel's `d2udx2`, but written as a flux balance so it works on
skewed ring cells. On boundary faces the gradient uses the boundary value:
\((\phi_{bc}-\phi_P)/(d/2)\). Implemented in `_laplacian`. No non-orthogonal
correction is applied; this is first-order on strongly skewed cells, which is
one reason the safety factor \(\tau\) is kept at 0.4.

### 3.4 Pressure gradient — Green–Gauss

\[
\big[\nabla p\big]_P\;\approx\;\frac{1}{V_P}\sum_{f\in P}s_{P,f}\,p_f\,\mathbf{n}_f\,L_f,
\qquad p_f=\tfrac12(p_P+p_N)
\tag{8}
\]

(`_grad`). On boundaries \(p_f=p_P\), i.e. the Neumann condition
\(\partial p/\partial n=0\). This gradient is only used to correct the
cell-centred \(u,v\); the face flux correction uses the compact two-point form
\((p_N-p_P)/d_{PN}\), see §4.

### 3.5 Divergence — face fluxes

\[
\big[\nabla\cdot\mathbf{u}\big]_P\;\approx\;\frac{1}{V_P}\sum_{f\in P}s_{P,f}\,U_{n,f}\,L_f
\tag{9}
\]

(`_div`). Because \(U_n\) lives on the faces, this is exact for the discrete
flux field and is the quantity driven to machine zero by the projection.

### 3.6 Pressure Poisson equation

Substituting the corrected velocity (§4, eq. 12) into the discrete continuity
(9) and requiring it to vanish gives, for every cell,

\[
\sum_{f\in P}s_{P,f}\,\frac{L_f}{d_{PN}}\,(p_N-p_P)\;=\;\frac{\rho}{\Delta t}\,V_P\,
\big[\nabla\cdot\mathbf{u}^*\big]_P
\tag{10}
\]

which is the FV analogue of Griebel's
\(\nabla^2 p = \frac{1}{\Delta t}\nabla\cdot\mathbf{u}^*\). Assembled once in
`_build_poisson` as a sparse matrix \(A\): off-diagonal \(+L_f/d_{PN}\),
diagonal \(-\sum_f L_f/d_{PN}\). Properties:

* **symmetric** (each face contributes the same coefficient to both cells);
* **singular** (all-Neumann: rows sum to zero, so \(p\) is defined up to a constant);
* **compatibility**: a solution exists only if the right-hand side sums to zero,
  i.e. net inflow equals net outflow. `_balance_outlet` enforces this by
  shifting the zero-gradient outlet flux before the solve; without it the
  residual mass imbalance appears as a uniform spurious divergence.

Solvers (`--poisson`):

| option | method |
|---|---|
| `lu` (default) | one dof pinned to remove the null space, SuperLU factorisation once, two triangular solves per step (≈2 ms on 37k cells) |
| `cg` | Jacobi-preconditioned conjugate gradient on the mean-projected operator |
| `sor` | successive over-relaxation, \(\omega=1.7\) — the original Griebel method |

---

## 4. One time step (Chorin / Griebel projection)

Given \(u^n, v^n\) (cells) and \(U_n^n\) (faces), `step()` does:

**0. Time step.** `stable_dt()`:

\[
\Delta t=\tau\,\min\!\left(\frac{h_{\min}}{|u|_{\max}},\ \frac{Re\,h_{\min}^2}{4}\right),\qquad \tau=0.4
\tag{11}
\]

The first term is the CFL (convective) limit, the second the diffusive limit
of explicit Euler. \(\gamma\) for the `donor` scheme is set from the same CFL.

**1. Boundary conditions** on \(U_n\) (`_enforce_Un_bc`): inlet Dirichlet,
walls zero.

**2. Predictor — momentum without pressure.** Explicit Euler for (2) with the
pressure gradient *omitted*:

\[
F_P = u_P^n+\Delta t\Big(\nu\nabla^2u-\nabla\cdot(\mathbf{u}u)\Big)_P^n,\qquad
G_P = v_P^n+\Delta t\Big(\nu\nabla^2v-\nabla\cdot(\mathbf{u}v)\Big)_P^n
\]

\(F, G\) are Griebel's names. These are a provisional velocity
\(\mathbf{u}^*=(F,G)\) that carries the right momentum but is **not**
divergence-free.

**3. Provisional face flux.** \(F, G\) are interpolated to faces with (6)
(central, \(\gamma=0\)) and projected on the normal:
\(U_n^*=F_f\,n_x+G_f\,n_y\). Boundary values are imposed and the outlet is
mass-balanced.

**4. Pressure Poisson.** Solve (10) for \(p^{n+1}\) with the right-hand side
\(\frac{\rho}{\Delta t}V_P\,[\nabla\cdot\mathbf{u}^*]_P\).

**5. Correction — make it divergence-free.**

\[
U_{n,f}^{\,n+1}=U_{n,f}^{*}-\frac{\Delta t}{\rho}\,\frac{p_N-p_P}{d_{PN}},
\qquad
u^{n+1}_P=F_P-\frac{\Delta t}{\rho}\big[\partial_x p\big]_P,\quad
v^{n+1}_P=G_P-\frac{\Delta t}{\rho}\big[\partial_y p\big]_P
\tag{12}
\]

The face correction uses the same \((p_N-p_P)/d_{PN}\) that was used to build
\(A\), so after this step \([\nabla\cdot\mathbf{u}^{n+1}]_P=0\) to solver
precision (the log column `max|div|` is ≈1e-12 with `lu`). The cell velocities
are corrected with the Green–Gauss gradient (8) for the convective/viscous
terms of the next step.

Why this works: taking the divergence of (12) and requiring it to vanish is
*exactly* equation (10). The pressure is therefore not a prognostic variable
but a Lagrange multiplier that enforces (1); since \(F,G\) contain no old
pressure, the Poisson solution is the pressure itself (`self.p = p_corr`), not
an increment.

**6. Advance.** \(t\leftarrow t+\Delta t\); compute diagnostics (`max|div|`,
\(|u|_{\max}\), \(C_d, C_l\) from `force_coefficients`: pressure \(\sum p\,\mathbf{n}L\)
plus wall shear \(\mu\,u_t/(d/2)\) on the cylinder faces), store a frame if
due, and go back to step 0 with \(u^{n+1},v^{n+1},U_n^{n+1}\).

### Summary of the discretisation choices

| term | method | order | code |
|---|---|---|---|
| \(\partial_t\) | explicit Euler | 1 | `step` |
| convection | face flux with \(\gamma\)-blend: central / upwind / donor | 2 / 1 / mixed | `_convective`, `donor_cell_face` |
| diffusion | central two-point face gradient | 2 (orthogonal), 1 (skewed) | `_laplacian` |
| pressure gradient | Green–Gauss (cells), two-point (faces) | 2 | `_grad`, `_project_Un` |
| divergence | face-flux sum | exact for \(U_n\) | `_div` |
| pressure Poisson | symmetric FV Laplacian, LU / PCG / SOR | — | `_build_poisson`, `_solve_poisson` |
| coupling | Chorin projection (predict → Poisson → correct) | 1 in time | `step`, `_project_Un` |

---

## 5. Initial condition

`_init_solenoidal`: \(u=U_\infty,\ v=0\) everywhere (impulsive start), plus a
small transverse seed \(v=\varepsilon U_\infty e^{-r^2/0.5^2}\) centred one
diameter behind the cylinder (`--perturb`, \(\varepsilon=0.1\)). This field is
projected once with (10)–(12) so the first step starts divergence-free. The
impulsive start is why \(C_d\) and \(C_l\) show a large spike in the first
few steps (the flow has to be brought to rest on the cylinder in one \(\Delta t\)).

## 6. Why a perturbation is needed

The Hopf bifurcation to vortex shedding occurs at \(Re\approx 47\); at
\(Re=150\) the steady symmetric wake is an *unstable* solution, but it is still
an exact solution. The O–H mesh is exactly mirror-symmetric about \(y=0\), and
so are all the operators above, so with a symmetric initial condition the
discrete solution stays symmetric (\(C_l\equiv 0\)) until round-off has grown
by a factor of ~\(10^{13}\) — in the central run that took until \(tU/D\approx120\).
A \(10\%\) seed in the near wake brings this forward to \(tU/D\approx 20\);
it is removed by the flow within a few time units and does not affect the
saturated limit cycle (both runs give \(C_l\) amplitude 0.75).

---

## 7. What the plots show

### Wake frames (`wake_*.png`, GIF, MP4)

**No streamlines are plotted.** Each frame is a snapshot of the *instantaneous
Eulerian fields at one time level*, and the next frame is the same picture for
a later time level — the movie is a sequence of independent snapshots, nothing
is tracked between them.

* **Colour.** Every quadrilateral cell of the mesh is drawn (Matplotlib
  `PolyCollection` over the exported `quads`) and filled with a single colour
  given by the cell-centred value at that instant. Top panel:
  \(\omega_z=\partial_x v-\partial_y u\), computed with the Green–Gauss
  gradient (8) applied to \(u\) and \(v\) (`vorticity`); red/blue are the
  counter-rotating vortices of the von Kármán street ("Vorticity plot").
  Bottom panel: the speed \(|\mathbf{u}|=\sqrt{u^2+v^2}\) ("Velocity plot"),
  so the wake (slow, dark) and the accelerated flow past the shoulders (fast,
  bright) stand out from the uniform free stream \(|\mathbf{u}|=U_\infty\).
* **Arrows.** On a subset of cells (every ~90th, ≈400 arrows) the vector
  \((u_P, v_P)\) at that instant is drawn with `quiver`; direction shows the
  flow direction, length is proportional to speed. They are *not* streamlines
  or particle paths — a streamline would require integrating
  \(d\mathbf{x}/ds=\mathbf{u}(\mathbf{x})\) through the velocity field, which
  is not done.
* **Time.** A frame is written every `--frame-every` steps (100 steps ≈ 0.45
  time units on the fine mesh, ≈11 frames per shedding period). The GIF loops
  over the last ~4 periods so it appears continuous; the MP4 covers the
  development from the symmetric wake (two straight shear layers) through the
  growing oscillation to the saturated street.

### Forces plot (`forces_*.png`)

\(C_d(t)\) and \(C_l(t)\) from `force_coefficients` at every step. Shedding
shows as a sinusoidal \(C_l\) at the shedding frequency and a \(C_d\) ripple
at twice that frequency (one vortex from each side per cycle). The Strouhal
number \(St=fD/U_\infty\) is estimated from the upward zero crossings of
\(C_l\) over the last 40 % of the run.

### Solution plot (`plot_*.png`)

Left: the speed \(|\mathbf{u}|\) at the final time ("Velocity plot"). Right: the continuity
residual \(\max_P|\nabla\cdot\mathbf{u}|\) versus time on a log scale — a check
that the projection is doing its job (≈1e-12 with the direct solver).
