# 12 steps to the cylinder solver

The same road as Lorena Barba's *CFD Python: 12 steps to Navier–Stokes*
(<https://github.com/barbagroup/CFDPython>), but every step ends with
"…and this is where it lives in `ns_staggered_euler.py`". If you have done
the 12 steps, each piece of the cylinder solver is something you have
already coded; the only new ingredient (Step 13) is that the grid is a
multiblock O–H mesh instead of `numpy.linspace`, so `i+1` becomes
"the neighbour across this face".

Notation as in the notebooks: superscript \(n\) = time level, subscript
\(i\) (or \(i,j\)) = grid point, \(\Delta t\), \(\Delta x\), \(\Delta y\) = step sizes.

---

## Step 1 — 1-D linear convection

\[
\frac{\partial u}{\partial t}+c\,\frac{\partial u}{\partial x}=0
\]

Forward difference in time, backward difference in space:

\[
\frac{u_i^{n+1}-u_i^n}{\Delta t}+c\,\frac{u_i^n-u_{i-1}^n}{\Delta x}=0
\quad\Longrightarrow\quad
u_i^{n+1}=u_i^n-c\frac{\Delta t}{\Delta x}\left(u_i^n-u_{i-1}^n\right)
\]

```python
for n in range(nt):
    un = u.copy()
    for i in range(1, nx):
        u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
```

**What we learn.** The explicit update needs *only old values* on the
right-hand side, so each step is a formula, not an equation to solve.
Using \(u_{i-1}\) (the point the wave is coming *from*) is **upwinding**:
stable, but it smears the step profile — numerical diffusion.

**In the cylinder solver.** Forward Euler in time is exactly what `step()`
does. The upwind choice reappears in Step 2 as the \(\gamma=1\) end of the
donor-cell blend.

---

## Step 2 — 1-D nonlinear convection

\[
\frac{\partial u}{\partial t}+u\,\frac{\partial u}{\partial x}=0
\qquad\Longrightarrow\qquad
u_i^{n+1}=u_i^n-u_i^n\frac{\Delta t}{\Delta x}\left(u_i^n-u_{i-1}^n\right)
\]

The wave speed is now the solution itself. Written in *conservative* form,
\(\partial_t u+\partial_x(u^2)=0\), the thing being transported through the
face between \(i\) and \(i+1\) is the flux \((u^2)_{i+\frac12}\), and the
question becomes: **what value of \(u\) do we use at the face?**

* average of the two sides, \(\tfrac12(u_i+u_{i+1})\) — **central**, second
  order, but it lets 2Δx wiggles through;
* the upstream side, \(u_i\) if the flow goes \(i\to i+1\) — **upwind**,
  first order, always bounded.

Griebel's donor-cell scheme (and CaptainProton42's `integration.c`) writes
both at once:

\[
u_{i+\frac12}=\tfrac12\left(u_i+u_{i+1}\right)-\tfrac12\,\gamma\,
\operatorname{sign}\!\left(u_{i+\frac12}\right)\left(u_{i+1}-u_i\right)
\]

\(\gamma=0\) is central, \(\gamma=1\) is upwind, anything between is a blend.

**In the cylinder solver.** `donor_cell_face()` is this formula with
\(i\to\) owner cell, \(i+1\to\) neighbour cell, and the face velocity
\(U_n\) deciding the sign. `--scheme central|upwind|donor` sets
\(\gamma=0,\ 1,\ \max(\text{CFL})\). The central scheme is the one that gave
vortex shedding; upwind damped it away, which is Step 1's numerical
diffusion showing up as a physical-looking but wrong steady wake.

---

## CFL condition (the bonus notebook)

Running Step 1 with more grid points *for the same \(\Delta t\)* blows up.
The explicit scheme is only stable if information travels less than one cell
per step:

\[
\sigma=\frac{u\,\Delta t}{\Delta x}\le\sigma_{\max}
\qquad\Longrightarrow\qquad
\Delta t=\sigma_{\max}\frac{\Delta x}{u}
\]

**In the cylinder solver.** `stable_dt()` computes
\(\Delta t=\tau\min\!\big(h_{\min}/|u|_{\max},\ Re\,h_{\min}^2/4\big)\). The
first term is this CFL condition with \(\Delta x\to h_{\min}\) (the smallest
cell, next to the cylinder) and \(\sigma_{\max}\to\tau=0.4\). The second
term comes from Step 3.

---

## Step 3 — 1-D diffusion

\[
\frac{\partial u}{\partial t}=\nu\frac{\partial^2u}{\partial x^2}
\]

Second derivative by the central difference (Taylor expansions of
\(u_{i+1}\) and \(u_{i-1}\) added together):

\[
\frac{\partial^2u}{\partial x^2}\approx\frac{u_{i+1}-2u_i+u_{i-1}}{\Delta x^2}
\qquad\Longrightarrow\qquad
u_i^{n+1}=u_i^n+\nu\frac{\Delta t}{\Delta x^2}\left(u_{i+1}^n-2u_i^n+u_{i-1}^n\right)
\]

Stability now needs \(\nu\Delta t/\Delta x^2\le\tfrac12\) — halving
\(\Delta x\) quarters the allowed \(\Delta t\).

**In the cylinder solver.** `_laplacian()`. The stencil
\(u_{i+1}-2u_i+u_{i-1}\) is the same as *"sum over the two faces of
(neighbour − me)"*; on a general cell it becomes
\(\sum_f (\phi_N-\phi_P)\,L_f/d_{PN}\), divided by the cell area. The
diffusive limit \(Re\,h_{\min}^2/4\) in `stable_dt()` is
\(\nu\Delta t/\Delta x^2\le\tfrac14\) in 2-D with \(\nu=1/Re\). It is why the
fine mesh (\(h_{\min}=0.018\)) runs at \(\Delta t\approx0.0045\).

---

## Step 4 — Burgers' equation

\[
\frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}=\nu\frac{\partial^2u}{\partial x^2}
\]

Steps 2 and 3 glued together on the right-hand side:

\[
u_i^{n+1}=u_i^n
-u_i^n\frac{\Delta t}{\Delta x}\left(u_i^n-u_{i-1}^n\right)
+\nu\frac{\Delta t}{\Delta x^2}\left(u_{i+1}^n-2u_i^n+u_{i-1}^n\right)
\]

**What we learn.** Convection and diffusion are *independent* terms
evaluated from the old field and added. Nothing couples them inside a step.

**In the cylinder solver.** This is Griebel's predictor, line for line:

```python
cu, cv = self._convective(self.u, self.v, self.Un, gamma)   # −∇·(u u), −∇·(u v)
lu = self._laplacian(self.u, ...)                           # ν∇²u
lv = self._laplacian(self.v, ...)                           # ν∇²v
F = self.u + dt * (lu + cu)                                 # Burgers step for u
G = self.v + dt * (lv + cv)                                 # Burgers step for v
```

\(F, G\) are "Burgers velocities": correct momentum, but nobody has asked
them to be divergence-free yet. That is the pressure's job (Step 11).

---

## Steps 5–8 — the same four equations in 2-D

Step 5 (linear convection), 6 (nonlinear), 7 (diffusion), 8 (Burgers) add
the \(j\) index and the second component \(v\). For 2-D Burgers:

\[
\begin{aligned}
u_{i,j}^{n+1}=u_{i,j}^n
&-u_{i,j}^n\frac{\Delta t}{\Delta x}\left(u_{i,j}^n-u_{i-1,j}^n\right)
-v_{i,j}^n\frac{\Delta t}{\Delta y}\left(u_{i,j}^n-u_{i,j-1}^n\right)\\
&+\nu\frac{\Delta t}{\Delta x^2}\left(u_{i+1,j}^n-2u_{i,j}^n+u_{i-1,j}^n\right)
+\nu\frac{\Delta t}{\Delta y^2}\left(u_{i,j+1}^n-2u_{i,j}^n+u_{i,j-1}^n\right)
\end{aligned}
\]

and the mirror image for \(v\).

**What we learn.** Each grid point talks to its four neighbours
\((i\pm1,j),(i,j\pm1)\) through its four faces. The \(x\)-terms use the
east/west pair, the \(y\)-terms the north/south pair. In the array-operations
notebook the loops become slices, `u[1:-1,1:-1] - u[:-2,1:-1]`.

**In the cylinder solver.** A quad cell on the O–H mesh also has four faces
and four neighbours, but they are not aligned with \(x\) and \(y\). So
instead of separate \(x\) and \(y\) stencils, every face contributes
\(\text{flux}\times L_f\) and we sum over the four faces (`accumulate()`).
That one sum *is* the two stencils above, rotated to whatever direction the
face happens to point in. The transported flux through a face is
\(U_{n,f}\,\phi_f\,L_f\) with \(U_n\) the staggered face velocity and
\(\phi_f\) from Step 2.

---

## Step 9 — Laplace equation

\[
\frac{\partial^2p}{\partial x^2}+\frac{\partial^2p}{\partial y^2}=0
\quad\Longrightarrow\quad
p_{i,j}=\frac{\Delta y^2\left(p_{i+1,j}+p_{i-1,j}\right)+\Delta x^2\left(p_{i,j+1}+p_{i,j-1}\right)}{2\left(\Delta x^2+\Delta y^2\right)}
\]

**What we learn.** No time derivative: this is an *equilibrium* problem.
Iterating the formula (Jacobi) until \(p\) stops changing is a *solver*, not
a time march. Boundary conditions can be Dirichlet (\(p\) given) or Neumann
(\(\partial p/\partial n=0\): copy the neighbour, `p[:, -1] = p[:, -2]`).

**In the cylinder solver.** All four boundaries are Neumann
(\(\partial p/\partial n=0\)), which has two consequences you can see in the
code: \(p\) is only defined up to a constant (`p -= p.mean()`), and the
equations are only solvable if the right-hand side sums to zero
(`_balance_outlet`).

---

## Step 10 — Poisson equation

\[
\frac{\partial^2p}{\partial x^2}+\frac{\partial^2p}{\partial y^2}=b
\quad\Longrightarrow\quad
p_{i,j}=\frac{\Delta y^2\left(p_{i+1,j}+p_{i-1,j}\right)+\Delta x^2\left(p_{i,j+1}+p_{i,j-1}\right)-b_{i,j}\Delta x^2\Delta y^2}{2\left(\Delta x^2+\Delta y^2\right)}
\]

Same iteration as Step 9 with a source \(b\). Every sweep is one matrix–vector
product with the 5-point Laplacian; convergence takes \(O(N)\) sweeps.

**In the cylinder solver.** `_build_poisson()` assembles the 5-point (here
4-face) Laplacian once as a sparse matrix \(A\): off-diagonal \(L_f/d_{PN}\),
diagonal minus the sum. Then `_solve_poisson()` offers three ways to solve
\(A\,p=b\):

| `--poisson` | what it is in 12-steps language |
|---|---|
| `sor` | Step 10's sweep with over-relaxation \(\omega=1.7\) — Griebel's original |
| `cg` | conjugate gradient: a smarter iteration that uses the symmetry of \(A\) |
| `lu` (default) | factorise \(A\) once, then every step is two triangular solves (≈2 ms) — possible because the mesh, hence \(A\), never changes |

---

## Step 11 — cavity flow: where the pressure comes from

Navier–Stokes with the pressure gradient added to Step 8:

\[
\frac{\partial\mathbf{u}}{\partial t}+(\mathbf{u}\cdot\nabla)\mathbf{u}
=-\frac{1}{\rho}\nabla p+\nu\nabla^2\mathbf{u},
\qquad\nabla\cdot\mathbf{u}=0
\]

There is no equation that says \(\partial p/\partial t=\dots\). Barba's
notebook gets one by **taking the divergence of the momentum equation and
demanding that \(\nabla\cdot\mathbf{u}^{n+1}=0\)**. That yields a Poisson
equation (Step 10) for \(p\):

\[
\nabla^2p^{n+1}=\frac{\rho}{\Delta t}\nabla\cdot\mathbf{u}^{*}
\quad(\text{Griebel form; Barba keeps the extra }\rho\,(u_x^2+2u_yv_x+v_y^2)\text{ terms that cancel if }\nabla\cdot\mathbf{u}^n=0\text{ exactly})
\]

where \(\mathbf{u}^*=(F,G)\) is the Burgers velocity from Step 4/8. The
discrete step is then:

1. \(F,G\): Step 8 (Burgers) — no pressure.
2. \(b=\frac{\rho}{\Delta t}\left(\frac{F_{i+1,j}-F_{i-1,j}}{2\Delta x}+\frac{G_{i,j+1}-G_{i,j-1}}{2\Delta y}\right)\).
3. Solve Step 10 for \(p^{n+1}\).
4. Correct:
\[
u^{n+1}_{i,j}=F_{i,j}-\frac{\Delta t}{\rho}\frac{p_{i+1,j}-p_{i-1,j}}{2\Delta x},\qquad
v^{n+1}_{i,j}=G_{i,j}-\frac{\Delta t}{\rho}\frac{p_{i,j+1}-p_{i,j-1}}{2\Delta y}
\]

Plug (4) into \(\nabla\cdot\mathbf{u}^{n+1}\) and you recover (3): the
pressure is precisely the field that removes the divergence of \(F,G\).

**What we learn.** The pressure is not marched in time, it is *solved for*
at every step as a constraint. The \(p_{i+1}-p_{i-1}\) over \(2\Delta x\) in
Barba's collocated grid skips point \(i\), which is why their cavity solution
can show a faint checkerboard; the staggered (MAC) grid of Griebel avoids
this by putting \(u\) on the faces between pressure points so the gradient is
\((p_{i+1}-p_i)/\Delta x\).

**In the cylinder solver.** `step()` is this list:

```python
F = self.u + dt * (lu + cu);  G = self.v + dt * (lv + cv)        # 1. Burgers
Un_star = uf*n_x + vf*n_y                                       #    F,G → face flux
Un_new  = self._project_Un(Un_star, dt)                          # 2.+3. b = ρ/dt·V·div(U*), solve A p = b
                                                                 # 4a. Un -= dt/ρ·(p_N − p_P)/d_PN  (staggered!)
gx, gy  = self._grad(p_corr)                                     # 4b. cell velocities
self.u  = F - (dt/rho)*gx;  self.v = G - (dt/rho)*gy
```

After 4a the face divergence is \(10^{-12}\) — the `max|div|` column in the
log is the cavity notebook's "is \(\nabla\cdot\mathbf{u}=0\)?" check, printed
every 20 steps.

---

## Step 12 — channel flow: inflow, outflow, and keeping mass

Step 12 drives a channel with a body force and periodic ends. Our cylinder
case is the same channel with an **inlet** and an **outlet** instead:

* inlet: \(u=U_\infty,\ v=0\) (Dirichlet) — `_enforce_Un_bc`;
* outlet: \(\partial\mathbf{u}/\partial n=0\) (copy the inside value, as the
  Neumann BC in Step 9) — the face flux is taken from the owner cell;
* top/bottom walls and the cylinder: \(\mathbf{u}=0\) (no-slip, as the cavity
  walls in Step 11).

**What we learn (the hard way).** With a zero-gradient outlet, what flows out
is not automatically what flows in. The Neumann Poisson problem of Step 9
then has no solution, and the iteration quietly settles on a field whose
divergence is a non-zero constant everywhere. In this solver that showed up
as `max|div| ≈ 1e-2` concentrated in the smallest cells. `_balance_outlet`
shifts the outlet flux by a constant so inflow = outflow before every Poisson
solve, after which the divergence is at round-off.

---

## Step 13 — from `linspace` to the O–H mesh

Everything above was derived on a Cartesian grid with indices \(i,j\). The
cylinder mesh from `mesh_from_known_sizes.py` is a multiblock mesh (rings
around the cylinder, a transition block, Cartesian blocks up- and downstream),
so there is no global \(i,j\). The translation is mechanical:

| 12-steps (Cartesian) | staggered O–H mesh |
|---|---|
| point \((i,j)\) | cell \(P\) with area \(V_P\) (`cell_centers`, `vol`) |
| neighbour \((i+1,j)\) | the cell across face \(f\): owner/neighbour pair (`face_owner`, `face_neigh`) |
| \(\Delta x\) between them | \(d_{PN}=\lvert\mathbf{x}_N-\mathbf{x}_P\rvert\) (`d_pn`) |
| face width \(\Delta y\) | \(L_f\) (`face_lengths`) |
| \(u_{i+\frac12,j}\) on the east face | \(U_{n,f}=\mathbf{u}\cdot\mathbf{n}_f\) on *that* face (`Un`) |
| "sum of \(x\)-stencil and \(y\)-stencil" | one sum over the cell's faces, \(\sum_f s_{P,f}(\cdot)_f L_f / V_P\) (`accumulate`) |
| \(\Delta x,\Delta y\) in the CFL rule | \(h_{\min}=\min\sqrt{V_P}\) |

The pressure stays at the cell centres and the velocity used for continuity
stays on the faces, so this is still the MAC grid of Griebel — just with
cells that are curved around the cylinder.

---

## Reading the plots (the notebooks' `pyplot.contourf` + `quiver`)

Barba's cavity and channel notebooks finish with

```python
pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
```

— filled colour for a scalar field, arrows for the velocity at every second
grid point, drawn from the arrays at the *final* time step. Our frames are
the same two calls, repeated:

* `contourf` becomes a `PolyCollection`: every mesh cell is filled with the
  colour of its own value (vorticity \(\omega_z=\partial_xv-\partial_yu\) on
  top, \(u-U_\infty\) below), because the cells are quads of varying size and
  shape rather than a regular \(X,Y\) grid;
* `quiver` draws \((u_P,v_P)\) at roughly every 90th cell;
* one such picture is saved every 100 time steps and the pictures are played
  in sequence as the GIF/MP4.

So, exactly as in the notebooks, each image is the field *at one instant*,
coloured and arrowed point by point; the next image is a later instant. There
are no streamlines (`pyplot.streamplot` is not used); the arrows show the
local velocity vector, not particle paths.
