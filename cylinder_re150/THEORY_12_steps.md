# 12 steps to the cylinder wake

How `ns_staggered_euler.py` solves flow past the cylinder at \(Re=150\),
told one step at a time: write the equation, discretise it on *our* mesh,
rearrange for the unknown, code it, look at what comes out. Each step adds
one piece; by Step 11 the whole solver is on the page, and Step 12 makes the
pictures.

Notation. \(P\) = a cell (pressure point), \(N\) = the cell across face
\(f\), \(\mathbf{n}_f\) the unit normal pointing \(P\to N\), \(L_f\) the face
length, \(d_{PN}\) the distance between the two cell centres, \(V_P\) the cell
area. \(U_{n,f}=\mathbf{u}\cdot\mathbf{n}_f\) is the velocity *through* face
\(f\). Superscript \(n\) is the time level. Arrays in the code:
`own[f]`, `nei[f]` (−1 on a boundary), `n[f]`, `L[f]`, `d_pn[f]`, `vol[P]`.

---

## Step 1 — what we are solving and where the unknowns sit

Incompressible Navier–Stokes with \(D=U_\infty=\rho=1\), \(\nu=1/Re\):

\[
\nabla\cdot\mathbf{u}=0,\qquad
\frac{\partial\mathbf{u}}{\partial t}+\nabla\cdot(\mathbf{u}\otimes\mathbf{u})
=-\nabla p+\nu\nabla^2\mathbf{u}
\]

The mesh from `mesh_from_known_sizes.py` is loaded as flat arrays — there
is no `i, j`; a cell knows its neighbours only through its faces:

```python
S = load_staggered("mesh_output/staggered_fine.npz")
P   = S["cell_centers"]      # (Nc, 2)  → p, u, v live here
Fc  = S["face_centers"]      # (Nf, 2)  → Un lives here
n   = S["face_normals"]      # (Nf, 2)  owner → neighbour
L   = S["face_lengths"]      # (Nf,)
own = S["face_owner"]        # (Nf,)  cell on the back side of n
nei = S["face_neigh"]        # (Nf,)  cell on the front side, -1 if boundary
```

Three fields are marched in time:

```python
u  = np.full(Nc, U_inf)      # cell-centred x-velocity
v  = np.zeros(Nc)            # cell-centred y-velocity
Un = np.zeros(Nf)            # face-normal velocity  (the staggered unknown)
p  = np.zeros(Nc)            # pressure, cell-centred
```

\(p\) in the cells and \(U_n\) on the faces is the staggered (MAC)
arrangement: the pressure difference across a face and the velocity through
that face sit at the same place.

**One tool for everything.** Every spatial derivative below is "integrate
over the cell, turn it into a sum over the cell's faces":

\[
\frac{1}{V_P}\int_{V_P}\nabla\cdot\mathbf{F}\,dV
=\frac{1}{V_P}\sum_{f\in P}s_{P,f}\,(\mathbf{F}\cdot\mathbf{n})_f\,L_f,
\qquad s_{P,f}=\begin{cases}+1&P=\text{owner}\\-1&P=\text{neighbour}\end{cases}
\]

```python
def accumulate(flux, own, nei, Nc):      # Σ_f s_{P,f} · flux_f  for every cell
    out = np.zeros(Nc)
    np.add.at(out, own, flux)            # + for the owner
    m = nei >= 0
    np.add.at(out, nei[m], -flux[m])     # − for the neighbour
    return out
```

Everything that follows is this function called with a different `flux`.

---

## Step 2 — the time derivative

Forward (explicit) Euler, the same for every cell:

\[
\frac{\partial\phi}{\partial t}\approx\frac{\phi_P^{n+1}-\phi_P^{n}}{\Delta t}
\quad\Longrightarrow\quad
\phi_P^{n+1}=\phi_P^{n}+\Delta t\cdot\big(\text{RHS evaluated with level-}n\text{ fields}\big)
\]

So the right-hand side is always a *formula* in known numbers, and the new
field is a one-line `numpy` expression. The cost is a limit on \(\Delta t\)
(Step 9).

---

## Step 3 — the convective term, three ways

\[
\big[\nabla\cdot(\mathbf{u}\,\phi)\big]_P
=\frac{1}{V_P}\sum_{f\in P}s_{P,f}\;U_{n,f}\,\phi_f\,L_f
\]

\(U_{n,f}L_f\) is the volume flux through the face — we have it, it is the
staggered unknown. \(\phi_f\) is the value of the transported quantity *at
the face*, and we only have it in the two cells \(P\) and \(N\). This is the
choice between the schemes:

\[
\phi_f=\underbrace{\tfrac12(\phi_P+\phi_N)}_{\text{central}}
\;-\;\tfrac12\,\gamma\,\operatorname{sign}(U_{n,f})\,(\phi_N-\phi_P)
\]

* \(\gamma=0\): the average — **central difference**. Second order, no
  built-in damping.
* \(\gamma=1\): \(\phi_P\) if the flow goes \(P\to N\), else \(\phi_N\) —
  **upwind / donor cell**. First order, adds a numerical viscosity
  \(\approx\tfrac12|U_n|\,d_{PN}\).
* \(0<\gamma<1\): blend. `--scheme donor` sets
  \(\gamma=\max_f|U_n|\Delta t/h_{\min}\) (the CFL number), Griebel's
  adaptive rule.

```python
def donor_cell_face(phi, Un, own, nei, interior, gamma):
    phi_f = phi[own].copy()                        # boundary faces: owner value
    o, nn = own[interior], nei[interior]
    central = 0.5 * (phi[o] + phi[nn])
    upwind  = np.where(Un[interior] >= 0.0, phi[o], phi[nn])
    phi_f[interior] = (1 - gamma) * central + gamma * upwind
    return phi_f

def _convective(u, v, Un, gamma):                  # returns −∇·(u u), −∇·(u v)
    uf = donor_cell_face(u, Un, own, nei, interior, gamma)
    vf = donor_cell_face(v, Un, own, nei, interior, gamma)
    # boundary faces: uf, vf = (U_inf, 0) at the inlet, (0, 0) on walls/cylinder
    ru = accumulate(Un * uf * L, own, nei, Nc)
    rv = accumulate(Un * vf * L, own, nei, Nc)
    return -ru / vol, -rv / vol
```

**What we see.** Run to \(tU/D=150\) on the 37k-cell mesh: with `donor`
(\(\gamma\approx0.36\)) the lift stays \(C_l\equiv0\) — the numerical
viscosity holds the wake in its steady, symmetric state. With `central` the
same mesh sheds vortices. The scheme choice is not cosmetic.

---

## Step 4 — the viscous term

\[
\big[\nu\nabla^2\phi\big]_P
=\frac{\nu}{V_P}\sum_{f\in P}s_{P,f}\;\Big(\frac{\partial\phi}{\partial n}\Big)_f L_f,
\qquad
\Big(\frac{\partial\phi}{\partial n}\Big)_f\approx\frac{\phi_N-\phi_P}{d_{PN}}
\]

Central two-point gradient across each face; on a boundary face the
boundary value is used over half the distance,
\((\phi_{bc}-\phi_P)/(d/2)\).

```python
def _laplacian(phi, phi_bc):
    g = np.empty(Nf)
    g[interior] = (phi[nei[interior]] - phi[own[interior]]) / d_pn[interior]
    g[bnd]      = (phi_bc[bnd] - phi[own[bnd]]) / (0.5 * d_pn[bnd])
    return accumulate(nu * g * L, own, nei, Nc) / vol
```

On a Cartesian cell this reduces to the familiar
\(\nu(\phi_{i+1}-2\phi_i+\phi_{i-1})/\Delta x^2\); here the four faces may
point in any direction and the sum takes care of it. (No correction for
skewed ring cells — one reason the safety factor in Step 9 is 0.4, not 0.5.)

---

## Step 5 — the predictor: momentum without pressure

Steps 2 + 3 + 4 together, leaving the pressure out on purpose:

\[
F_P=u_P^n+\Delta t\Big[\nu\nabla^2u-\nabla\cdot(\mathbf{u}u)\Big]_P^n,
\qquad
G_P=v_P^n+\Delta t\Big[\nu\nabla^2v-\nabla\cdot(\mathbf{u}v)\Big]_P^n
\]

```python
cu, cv = _convective(u, v, Un, gamma)
F = u + dt * (_laplacian(u, u_bc) + cu)
G = v + dt * (_laplacian(v, v_bc) + cv)
```

\(F,G\) (Griebel's names) have the right momentum but, because nothing has
enforced continuity, \(\nabla\cdot(F,G)\neq0\). The pressure's only job is
to fix that.

To use them on the faces we need the flux they imply:

\[
U^*_{n,f}=F_f\,n_{x,f}+G_f\,n_{y,f},\qquad F_f=\tfrac12(F_P+F_N)
\]

```python
uf, vf  = face_values(F, G, gamma=0.0)      # central average + boundary values
Un_star = uf * n[:, 0] + vf * n[:, 1]
```

---

## Step 6 — the divergence, and the equation for the pressure

Divergence of a cell is the net flux out of it:

\[
\big[\nabla\cdot\mathbf{u}\big]_P=\frac{1}{V_P}\sum_{f\in P}s_{P,f}\,U_{n,f}\,L_f
\]

```python
def _div(Un):
    return accumulate(Un * L, own, nei, Nc) / vol
```

Now the key move. We *want* the end-of-step flux to be

\[
U^{\,n+1}_{n,f}=U^{*}_{n,f}-\Delta t\,\frac{p_N-p_P}{d_{PN}}
\]

(the pressure gradient across the face, acting on the velocity through the
face — a two-point difference, no point skipped) **and** we want
\(\nabla\cdot\mathbf{u}^{n+1}=0\). Put the first into the second:

\[
\frac{1}{V_P}\sum_{f}s_{P,f}\Big(U^*_{n,f}-\Delta t\frac{p_N-p_P}{d_{PN}}\Big)L_f=0
\;\;\Longrightarrow\;\;
\boxed{\;\sum_{f\in P}s_{P,f}\,\frac{L_f}{d_{PN}}\,(p_N-p_P)=\frac{V_P}{\Delta t}\big[\nabla\cdot\mathbf{u}^*\big]_P\;}
\]

One linear equation per cell, unknowns \(p\) in that cell and its
neighbours: a Poisson equation. Its matrix never changes (the mesh does not
move), so it is assembled once:

```python
def _build_poisson():                 # A p = rhs,  A symmetric
    o, c = own[interior], nei[interior]
    g = L[interior] / d_pn[interior]
    rows = np.r_[o, o, c, c]; cols = np.r_[c, o, o, c]; data = np.r_[g, -g, g, -g]
    return sparse.coo_matrix((data, (rows, cols)), shape=(Nc, Nc)).tocsr()
```

Two facts about \(A\) decide how we solve it. Every boundary is
\(\partial p/\partial n=0\) (no face coefficient on boundary faces), so rows
sum to zero: \(p\) is only defined up to a constant, and a solution exists
only if `rhs.sum() == 0`, i.e. if as much fluid leaves through the outlet as
enters at the inlet.

---

## Step 7 — boundary conditions on the flux

Before the divergence is taken, `Un_star` is made to respect the boundaries:

```python
def _enforce_Un_bc(Un):
    Un[inlet] = U_inf * n[inlet, 0] + v_in(t) * n[inlet, 1]   # prescribed
    Un[wall_top] = Un[wall_bottom] = Un[cylinder] = 0.0        # no-slip
    return Un                                                  # outlet: left as is (zero gradient)

def _balance_outlet(Un):                 # make rhs.sum() == 0
    Q_other = np.sum(Un[~outlet & bnd] * L[~outlet & bnd])
    Q_out   = np.sum(Un[outlet] * L[outlet])
    Un[outlet] += -(Q_other + Q_out) / np.sum(L[outlet])
    return Un
```

**What we saw without `_balance_outlet`.** The solver ran, but
`max|div|` sat at \(10^{-2}\), largest in the tiny cells on the cylinder:
the unbalanced mass had nowhere to go and was spread over the domain as a
constant divergence. With it, `max|div|` is \(10^{-12}\).

\(v_{in}(t)=0.1\,U_\infty\sin(\pi t/2)\) for \(t<2\), together with a small
transverse seed in the initial field, breaks the mirror symmetry of the mesh
(Step 10).

---

## Step 8 — solve for \(p\), then correct

```python
rhs = (1.0 / dt) * vol * _div(Un_star)
rhs -= rhs.mean()                       # exact compatibility after Step 7
p = solve(A, rhs)                       # lu | cg | sor
p -= p.mean()                           # fix the arbitrary constant
```

Three solvers for the same system (`--poisson`):

| option | what it does | cost on 37k cells |
|---|---|---|
| `sor` | Gauss–Seidel sweeps with \(\omega=1.7\) (Griebel's original) | thousands of sweeps |
| `cg` | Jacobi-preconditioned conjugate gradient on the mean-free operator | ≈1000 iterations, 390 ms |
| `lu` | pin one cell, `splu(A)` once at start-up, then two triangular solves | **2 ms** |

Then the correction, face flux first, cell velocities second:

\[
U^{\,n+1}_{n,f}=U^*_{n,f}-\Delta t\,\frac{p_N-p_P}{d_{PN}},
\qquad
u_P^{n+1}=F_P-\Delta t\,[\partial_xp]_P,\quad
v_P^{n+1}=G_P-\Delta t\,[\partial_yp]_P
\]

with the cell gradient from the face averages (Green–Gauss):
\([\nabla p]_P=\frac{1}{V_P}\sum_f s_{P,f}\,\tfrac12(p_P+p_N)\,\mathbf{n}_fL_f\).

```python
Un = Un_star.copy()
Un[interior] -= dt * (p[nei[interior]] - p[own[interior]]) / d_pn[interior]
Un = _enforce_Un_bc(Un)

pf = 0.5 * (p[own] + p[nei]);  pf[bnd] = p[own[bnd]]
gx = accumulate(pf * n[:, 0] * L, own, nei, Nc) / vol
gy = accumulate(pf * n[:, 1] * L, own, nei, Nc) / vol
u = F - dt * gx
v = G - dt * gy
```

Because the face correction uses the same \((p_N-p_P)/d_{PN}\) that built
\(A\), `_div(Un)` is now zero to solver precision — that is the
`max|div|=7.9e-13` column in the log, every 20 steps. \(F,G\) contained no
old pressure, so this \(p\) *is* the pressure at the new level, not an
increment (`self.p = p`).

---

## Step 9 — choosing \(\Delta t\)

Explicit Euler is stable only if, in every cell, fluid moves less than one
cell per step and diffusion acts over less than one cell per step. The
smallest cell is the binding one:

\[
\Delta t=\tau\,\min\!\left(\frac{h_{\min}}{|u|_{\max}},\;\frac{Re\,h_{\min}^2}{4}\right),
\qquad h_{\min}=\min_P\sqrt{V_P},\quad\tau=0.4
\]

```python
def stable_dt():
    umax = max(np.abs(Un).max(), U_inf)
    return tau * min(hmin / umax, Re * hmin**2 / 4)
```

**Numbers.** Fine mesh: \(h_{\min}=0.0177\) (first ring on the cylinder),
\(|u|_{\max}\approx1.58\) during shedding → \(\Delta t\approx0.0045\),
about 1,100 steps per shedding period, 26,600 steps to \(tU/D=120\). The
far-field cells are 0.12 wide and could take a 7× larger step; a single
global explicit \(\Delta t\) does not let them.

---

## Step 10 — the initial condition and the symmetry problem

\(u=U_\infty,\ v=0\) everywhere (impulsive start), projected once with
Steps 6–8 so the first step begins divergence-free.

**What we saw.** The O–H mesh is exactly mirror-symmetric about \(y=0\), and
so is every operator above. Starting symmetric, the solution *stays*
symmetric: \(C_l\) printed as `-0.000` for 120 time units while the wake
grew into the long, steady, unstable bubble of the symmetric branch. Only at
\(tU/D\approx120\) had round-off grown enough to start the oscillation.
A small seed in the initial field,

```python
r2 = (P[:, 0] - 1.0)**2 + P[:, 1]**2            # one diameter behind the cylinder
v += perturb * U_inf * np.exp(-r2 / 0.5**2)     # perturb = 0.1
```

brings the onset to \(tU/D\approx20\). The seed is washed out within a few
time units; both runs end on the same limit cycle (\(C_l\) amplitude 0.75).

---

## Step 11 — the time loop

Everything above, in the order it runs:

```python
while t < t_end:
    dt    = stable_dt()                                   # Step 9
    gamma = scheme_gamma(dt)                              # Step 3
    Un    = _enforce_Un_bc(Un)                            # Step 7
    cu, cv = _convective(u, v, Un, gamma)                 # Step 3
    F = u + dt * (_laplacian(u, u_bc) + cu)               # Steps 4–5
    G = v + dt * (_laplacian(v, v_bc) + cv)
    Un_star = face_normal(F, G)                           # Step 5
    Un_star = _balance_outlet(_enforce_Un_bc(Un_star))    # Step 7
    rhs = vol / dt * _div(Un_star); rhs -= rhs.mean()     # Step 6
    p = solve(A, rhs); p -= p.mean()                      # Step 8
    Un = Un_star - dt * grad_face(p); Un = _enforce_Un_bc(Un)
    u, v = F - dt * gx(p), G - dt * gy(p)
    t += dt
    Cd, Cl = forces_on_cylinder(p, u, v)                  # diagnostics
    if step % 100 == 0: save_frame(u, v, t)               # Step 12
```

Forces: pressure \(\sum_{f\in\text{cyl}}p_P\,\mathbf{n}_fL_f\) plus wall
shear \(\nu\,u_t/(d/2)\) on the cylinder faces, divided by
\(\tfrac12\rho U_\infty^2D\).

**What comes out** (fine mesh, `central`): after the impulsive-start
transient, \(C_l\) grows exponentially, saturates at \(\pm0.75\) by
\(tU/D\approx30\) with period \(1/St\approx4.8\) (\(St\approx0.20\)–\(0.21\)), \(C_d\)
oscillates at twice that frequency around 1.64. Reference values for an
unbounded cylinder are \(St\approx0.18\), \(C_d\approx1.33\); the 10 %
blockage of the 12 D-high channel and the first-order time scheme push both up.

---

## Step 12 — the pictures

Each frame is the state at one time level, drawn cell by cell:

```python
coll = PolyCollection(S["quads"], array=field, cmap="RdBu_r")   # one colour per cell
ax.add_collection(coll)
idx = np.arange(0, Nc, Nc // 400)                               # ~400 cells
ax.quiver(P[idx, 0], P[idx, 1], u[idx], v[idx])                  # the velocity vector there
```

* top panel: \(\omega_z=\partial_xv-\partial_yu\), from the same Green–Gauss
  gradient as Step 8 — red and blue are the two senses of rotation, so the
  alternating street is unmistakable;
* bottom panel ("Velocity plot"): the speed \(|\mathbf{u}|\), so the wake
  (slow) is dark and the shoulders (fast) bright; arrows are \((u_P,v_P)\) in
  the sampled cells.

A frame is saved every 100 steps (≈0.45 time units, ~11 per shedding
period). The GIF is the last ~4 periods looped; the MP4 is the whole history.
Nothing is traced between frames: the next frame is simply the next saved
state, coloured and arrowed again. There are no streamlines; an arrow is the
local velocity vector at that instant, not a path.

---

### The twelve steps on one line

mesh arrays → explicit Euler → \(\gamma\)-blended convective flux →
central viscous flux → predictor \(F,G\) → divergence ⇒ Poisson for \(p\) →
BCs + mass balance → solve & correct (faces, then cells) → CFL/diffusive
\(\Delta t\) → symmetric start + seed → loop → cell-coloured snapshots.
