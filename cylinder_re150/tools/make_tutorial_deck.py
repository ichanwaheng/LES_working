"""Build the beginner's reproducible walkthrough deck."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Circle, Rectangle
from matplotlib.collections import PolyCollection

from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptkit import Deck, NAVY, GREY, ACC, GREEN, LIGHT, TMP

ROOT = HERE.parent
ART = ROOT / "demo"
OUT = ROOT / "demo" / "ns_staggered_tutorial.pptx"

I = Inches


# ====================================================================== assets
def sketch_domain():
    p = TMP / "sk_domain.png"
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.add_patch(Rectangle((0, 0), 24, 12, fc="#eef2f7", ec="k", lw=1.5))
    ax.add_patch(Circle((6, 6), 0.5, fc="#cccccc", ec="k", lw=1.5))
    ax.text(6, 3.4, "cylinder D = 1\nat (6 D, 6 D), no-slip", ha="center", fontsize=12)
    ax.text(-0.3, 6, "inlet (x = 0)\nu = U∞ = 1\nv = 0", ha="right", va="center", fontsize=12, color="#2a7f3a")
    ax.text(24.3, 6, "outlet (x = 24)\n∂u/∂n = 0\n∂v/∂n = 0", ha="left", va="center", fontsize=12, color="#B23A2A")
    ax.text(12, 12.4, "wall_top (y = 12): u = v = 0", ha="center", fontsize=12)
    ax.text(12, -0.9, "wall_bottom (y = 0): u = v = 0", ha="center", fontsize=12)
    for y in (2.5, 4.5, 7.5, 9.5):
        ax.add_patch(FancyArrowPatch((0.5, y), (2.5, y), arrowstyle="->", mutation_scale=16, color="#2a7f3a"))
    ax.text(17, 1.6, "x ∈ [0, 24],  y ∈ [0, 12]  (lengths in D)", fontsize=12, ha="center")
    ax.text(17, 10.2, "Re = U∞ D / ν = 150  →  ν = 1/150", fontsize=12, ha="center")
    ax.set_xlim(-5.5, 29.5); ax.set_ylim(-1.6, 13.4); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(p, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.02); plt.close(fig); return p


def sketch_bl():
    p = TMP / "sk_bl.png"
    d_min, r, d_max = 0.015, 1.15, 0.12
    n = int(np.floor(np.log(d_max / d_min) / np.log(r) + 1e-9)) + 1
    k = np.arange(n + 1); t = d_min * (r ** k - 1) / (r - 1)
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.add_patch(Rectangle((-0.1, 0), 0.1, 0.5, fc="#bbbbbb", ec="k"))
    ax.text(-0.05, 0.25, "wall", rotation=90, ha="center", va="center", fontsize=10)
    for i in range(n):
        ax.add_patch(Rectangle((t[i], 0), t[i + 1] - t[i], 0.5, fc="#dbe7f5" if i < 4 else "#f3f4f6", ec="#1F3A5F", lw=0.8))
    for i in range(4):
        ax.add_patch(Rectangle((t[-1] + i * d_max, 0), d_max, 0.5, fc="white", ec="#777", lw=0.8))
    ax.annotate("", (t[0], 0.58), (t[1], 0.58), arrowprops=dict(arrowstyle="<->", lw=1)); ax.text(t[1] / 2, 0.62, "d_min = 0.015", fontsize=13, ha="left")
    ax.annotate("", (t[-1], 0.58), (t[-1] + d_max, 0.58), arrowprops=dict(arrowstyle="<->", lw=1)); ax.text(t[-1] + d_max / 2, 0.62, "d_max = 0.12", fontsize=13, ha="center")
    ax.text(t[-1] / 2, -0.12, f"boundary-layer stack: n = {n} cells, each 1.15× the previous, total T = {t[-1]:.3f}", fontsize=13, ha="center")
    ax.text(t[-1] / 2, -0.24, "shaded: the first 4 cells cover the physical boundary layer δ ≈ D/√Re ≈ 0.08", fontsize=13, ha="center", color="#1F3A5F")
    ax.set_xlim(-0.15, t[-1] + 4 * d_max + 0.05); ax.set_ylim(-0.34, 0.8); ax.axis("off")
    fig.tight_layout(); fig.savefig(p, dpi=200, transparent=True); plt.close(fig); return p


def sketch_cell():
    p = TMP / "sk_cell.png"
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    P = np.array([[0, 0], [2.0, 0.1], [2.1, 1.7], [0.0, 1.6]])
    N = np.array([[2.0, 0.1], [4.6, -0.1], [4.8, 1.6], [2.1, 1.7]])
    ax.add_patch(Polygon(P, fc="#dbe7f5", ec="k", lw=1.5)); ax.add_patch(Polygon(N, fc="#f7e3df", ec="k", lw=1.5))
    cP, cN = P.mean(0), N.mean(0)
    ax.plot(*cP, "o", color="#1F3A5F", ms=9); ax.text(cP[0], cP[1] + 0.22, "P (owner)", fontsize=13, fontweight="bold", ha="center")
    ax.plot(*cN, "o", color="#B23A2A", ms=9); ax.text(cN[0] + 0.35, cN[1] + 0.45, "N (neighbour)", fontsize=13, fontweight="bold", ha="center")
    f0, f1 = P[1], P[2]; fc = 0.5 * (f0 + f1)
    t = f1 - f0; nrm = np.array([t[1], -t[0]]); nrm /= np.linalg.norm(nrm)
    ax.plot([f0[0], f1[0]], [f0[1], f1[1]], color="#2a7f3a", lw=4); ax.plot(*fc, "s", color="#2a7f3a", ms=9)
    ax.add_patch(FancyArrowPatch(fc, fc + 0.7 * nrm, arrowstyle="->", mutation_scale=18, lw=2, color="#2a7f3a"))
    ax.text(fc[0] + 0.75, fc[1] - 0.05, r"$\mathbf{n}_f$", fontsize=15, color="#2a7f3a")
    ax.text(fc[0], -0.45, r"face $f$: stores $U_{n,f}$;  length $L_f$", fontsize=12, color="#2a7f3a", ha="center")
    ax.plot([cP[0], cN[0]], [cP[1], cN[1]], "k--", lw=1)
    ax.text(cP[0] + 0.55, cP[1] - 0.3, r"$d_{PN}$", fontsize=14, ha="center")
    ax.text(cP[0], cP[1] - 0.6, r"$p_P,\ u_P,\ v_P,\ V_P$", fontsize=12, ha="center", color="#1F3A5F")
    ax.set_xlim(-0.3, 5.1); ax.set_ylim(-0.75, 2.2); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(p, dpi=200, transparent=True); plt.close(fig); return p


def sketch_checkerboard():
    p = TMP / "sk_checker.png"
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6))
    x = np.arange(6)
    pc = np.array([1, -1, 1, -1, 1, -1], float)
    for ax, title in zip(axes, ["collocated: u and p at the same points", "staggered: U$_n$ on the faces between p"]):
        ax.bar(x, pc, color=["#B23A2A" if v > 0 else "#1F3A5F" for v in pc], width=0.5)
        ax.axhline(0, color="k", lw=0.8); ax.set_ylim(-1.6, 1.9); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=12); ax.text(2.5, 1.45, "p", fontsize=13, ha="center")
    axes[0].text(2.5, -1.45, "(p$_{i+1}$ − p$_{i-1}$)/2Δx = 0: invisible", fontsize=12, ha="center", color="#B23A2A")
    for xi in x[:-1] + 0.5:
        axes[1].plot(xi, 0, "s", color="#2a7f3a", ms=7)
    axes[1].text(2.5, -1.45, "(p$_{i+1}$ − p$_i$)/Δx = ±2/Δx: corrected", fontsize=12, ha="center", color="#2a7f3a")
    for ax in axes:
        for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig(p, dpi=200, transparent=True); plt.close(fig); return p


def sketch_face_value():
    p = TMP / "sk_face.png"
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = [0, 1]; phi = [1.0, 0.3]
    ax.plot(x, phi, "o", color="#1F3A5F", ms=10)
    ax.text(0, 1.12, r"$\phi_P$", ha="center", fontsize=15); ax.text(1, 0.42, r"$\phi_N$", ha="center", fontsize=15)
    ax.axvline(0.5, color="#2a7f3a", lw=3); ax.text(0.5, -0.02, "face f", ha="center", color="#2a7f3a", fontsize=13)
    ax.plot(0.5, 0.65, "D", color="#555", ms=9); ax.text(0.58, 0.68, r"central $\frac{1}{2}(\phi_P+\phi_N)$", fontsize=12)
    ax.plot(0.5, 1.0, "s", color="#B23A2A", ms=9); ax.text(0.58, 0.97, r"upwind $\phi_P$  ($U_n>0$)", fontsize=12, color="#B23A2A")
    ax.add_patch(FancyArrowPatch((0.05, 0.15), (0.45, 0.15), arrowstyle="->", mutation_scale=16, lw=2, color="#2a7f3a"))
    ax.text(0.25, 0.2, r"$U_{n,f}$", ha="center", color="#2a7f3a", fontsize=13)
    ax.set_xlim(-0.3, 1.5); ax.set_ylim(-0.1, 1.3); ax.axis("off")
    fig.tight_layout(); fig.savefig(p, dpi=200, transparent=True); plt.close(fig); return p


def sketch_flow():
    p = TMP / "sk_flow.png"
    fig, ax = plt.subplots(figsize=(4.2, 6.2))
    steps = [("1  Δt from CFL / diffusion", "#eef2f7"), ("2  BCs on U_n", "#eef2f7"),
             ("3  F, G = u + Δt(ν∇²u − ∇·(uu))", "#f7e3df"), ("4  U_n* = F n_x + G n_y, BCs,\n    balance outlet", "#f7e3df"),
             ("5  rhs = V/Δt · div(U_n*)", "#e6f1e0"), ("6  solve A p = rhs", "#e6f1e0"),
             ("7  U_n = U_n* − Δt (p_N−p_P)/d_PN", "#dbe7f5"), ("8  u = F − Δt ∂p/∂x,  v = G − Δt ∂p/∂y", "#dbe7f5"),
             ("9  t += Δt;  Cd, Cl;  frame", "#eef2f7")]
    y = 8.6
    for txt, col in steps:
        ax.add_patch(Rectangle((0, y), 4, 0.75, fc=col, ec="k", lw=0.8))
        ax.text(0.15, y + 0.375, txt, va="center", fontsize=9.5, family="monospace")
        if y > 0.6:
            ax.add_patch(FancyArrowPatch((2, y), (2, y - 0.25), arrowstyle="->", mutation_scale=12, lw=1.2))
        y -= 1.0
    ax.add_patch(FancyArrowPatch((4.1, 0.95), (4.1, 9.0), arrowstyle="->", mutation_scale=12, lw=1.2, connectionstyle="arc3,rad=-0.35", color="#555"))
    ax.text(4.25, 5.0, "next step", rotation=90, va="center", fontsize=9, color="#555")
    ax.set_xlim(-0.1, 4.8); ax.set_ylim(0.4, 9.6); ax.axis("off")
    fig.tight_layout(); fig.savefig(p, dpi=200, transparent=True); plt.close(fig); return p


def near_wake_zoom():
    p = TMP / "zoom_wake.png"
    import ns_staggered_euler as m
    S = m.load_staggered(ROOT / "mesh_output" / "staggered_fine.npz")
    s = m.GriebelStaggeredNS(S, Re=150, scheme="central")
    s.load_state(ROOT / "output_ns" / "fine_24x12" / "fields_central_fine.npz")
    xc, yc = S["xc"], S["yc"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (f, lab, lo, hi, cm) in zip(axes, [(np.hypot(s.u, s.v), "Velocity plot (cell values, mesh edges drawn)", 0.0, 1.6, "viridis"), (s.vorticity(), "Vorticity plot", -3, 3, "RdBu_r")]):
        c = PolyCollection(S["quads"], array=f, cmap=cm, edgecolors="k", linewidths=0.08); c.set_clim(lo, hi); ax.add_collection(c)
        ax.add_patch(Circle((xc, yc), 0.5, fc="white", ec="k"))
        ax.set_xlim(xc - 0.9, xc + 2.6); ax.set_ylim(yc - 1.2, yc + 1.2); ax.set_aspect("equal"); ax.set_title(lab, fontsize=11)
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); return p


def dt_history():
    p = TMP / "dt_hist.png"
    d = np.load(ROOT / "output_ns" / "fine_24x12" / "fields_central_fine.npz")
    t = d["hist_t"]; dt = np.diff(t)
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.plot(t[1:], dt, color="#1F3A5F", lw=1.2); ax.set_xlabel("t U / D"); ax.set_ylabel("Δt"); ax.set_ylim(0, 0.006)
    ax.axhline(0.4 * 150 * 0.01774 ** 2 / 4, color="#B23A2A", ls="--", lw=1); ax.text(60, 0.0049, "τ·Re h²/4 (diffusive limit)", color="#B23A2A", fontsize=9)
    ax.grid(alpha=0.3); ax.set_title("Δt actually used (fine mesh, central)", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(p, dpi=170); plt.close(fig); return p


SK_DOMAIN, SK_BL, SK_CELL, SK_CHK, SK_FACE, SK_FLOW = sketch_domain(), sketch_bl(), sketch_cell(), sketch_checkerboard(), sketch_face_value(), sketch_flow()
ZOOM, DTH = near_wake_zoom(), dt_history()

# ====================================================================== deck
D = Deck("Flow past a cylinder at Re = 150 — reproducible walkthrough (cylinder_re150/)")

# ---- title
s = D.section("CYLINDER_RE150 — FROM ZERO TO VORTEX SHEDDING",
              "Flow past a cylinder at Re = 150",
              "A step-by-step walkthrough you can reproduce: build the mesh, extract the staggered grid, discretise the\n"
              "Navier–Stokes equations, march in time, check the answer, make the pictures.")
D.pic(s, ART / "ns_fine_shedding_frame.png", I(9.3), I(0.45), height=I(2.6))
tb = s.shapes.add_textbox(I(0.9), I(6.2), I(8), I(0.9)); tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run(); r.text = ("Method: Griebel, Dornseifer & Neunhoeffer (1998) projection scheme, as in github.com/CaptainProton42/NavierStokes,\n"
                                          "with finite-volume operators on the multiblock O–H mesh of mesh_from_known_sizes.py")
r.font.size = Pt(11); from pptx.dml.color import RGBColor; r.font.color.rgb = RGBColor(0xBB, 0xC8, 0xD8)

# ---- what you get / need
s = D.slide("0", "What you will build, and what you need")
D.bullets(s, [
    "**What you will build",
    (1, "a 2-D incompressible Navier–Stokes solver that produces the von Kármán vortex street behind a cylinder"),
    (1, "on a body-fitted multiblock mesh (26,444 cells) with small cells on the walls and large cells far away"),
    (1, "with checks: mass conservation to 1e-12, lift/drag history, Strouhal number ≈ 0.21"),
    "**What you need",
    (1, "Python 3.10+, numpy, scipy, matplotlib  (pip install numpy scipy matplotlib);  ffmpeg only for the mp4"),
    (1, "the two files:  mesh_from_known_sizes.py  (mesh + staggered grid)   and   ns_staggered_euler.py  (solver)"),
    (1, "≈ 6 min of CPU for the fine run to tU/D = 120 on one core (12 ms per step × 26,600 steps)"),
    "**What you should already know",
    (1, "what a partial derivative, a divergence and a gradient are;  numpy arrays and fancy indexing (a[idx])"),
    (1, "nothing about CFD — every discretisation is derived here before it is used"),
], I(0.6), I(1.3), I(7.4), I(5.5), size=14)
D.table(s, [
    ["symbol", "meaning", "value here"],
    ["D", "cylinder diameter (length unit)", "1"],
    ["U∞", "inlet speed (velocity unit)", "1"],
    ["ρ", "density", "1"],
    ["Re", "U∞ D / ν", "150"],
    ["ν", "kinematic viscosity = 1/Re", "0.00667"],
    ["t", "time in units of D/U∞", "0 … 120"],
    ["St", "shedding frequency × D / U∞", "≈ 0.18 (lit.)"],
    ["Cd, Cl", "force / (½ ρ U∞² D)", "≈ 1.33, ±0.5 (lit.)"],
], I(8.3), I(1.4), I(4.6), [I(0.9), I(2.3), I(1.4)], size=11)

# ---- roadmap
s = D.slide("0", "Roadmap")
D.table(s, [
    ["part", "what we do", "file / function", "steps"],
    ["A", "state the physical problem and the equations", "—", "A1–A2"],
    ["B", "build a mesh from three numbers: d_min, d_max, growth", "mesh_from_known_sizes.py", "B1–B4"],
    ["C", "turn the quads into a staggered grid: cells, faces, owner/neighbour", "extract_staggered, build_topology", "C1–C4"],
    ["D", "discretise each term of Navier–Stokes on that grid", "_convective, _laplacian, _div, _grad", "D1–D5"],
    ["E", "derive the pressure equation and assemble/solve it", "_build_poisson, _solve_poisson", "E1–E3"],
    ["F", "boundary conditions, Δt, initial condition", "_enforce_Un_bc, _balance_outlet, stable_dt", "F1–F3"],
    ["G", "one complete time step, then the loop", "step(), main()", "G1–G3"],
    ["H", "run it: commands, flags, console output, files", "CLI", "H1–H3"],
    ["I", "check the answer: divergence, Cd/Cl, St, literature", "force_coefficients, plot_forces", "I1–I2"],
    ["J", "make the pictures and understand what they show", "save_wake_frame, build_gif", "J1–J2"],
    ["K", "pitfalls we hit (so you don't), exercises, glossary, file map", "—", "K1–K3"],
], I(0.6), I(1.3), I(12.1), [I(0.7), I(5.4), I(4.4), I(1.6)], size=12)
D.text(s, "Each step slide is self-contained: the formula, the few lines of code that implement it, and the numbers you should see on the fine mesh.", I(0.6), I(5.6), I(12.1), I(0.6), size=13, italic=True)

# ======================================================== PART A
D.section("PART A", "The physical problem", "What flows where, and which equations describe it")

s = D.slide("A1", "The physical problem")
D.pic(s, SK_DOMAIN, I(0.6), I(1.2), height=I(3.0))
D.bullets(s, [
    "A channel 24 D long and 12 D high, with a cylinder of diameter D centred at (6 D, 6 D) — 6 D from the inlet, mid-height",
    "Fluid enters from the left (x = 0) at speed U∞, leaves on the right (x = 24), cannot slip on the top/bottom walls (y = 0, 12) or the cylinder",
    "Everything is non-dimensional: lengths in D, speeds in U∞, time in D/U∞ (\"convective time units\")",
    "The only parameter is Re = U∞D/ν = 150",
    "**What should happen (experiments & literature)",
    (1, "below Re ≈ 47 the wake is steady and symmetric; above it the wake oscillates — vortices peel off alternately from top and bottom"),
    (1, "at Re = 150: shedding frequency St = f D/U∞ ≈ 0.18, mean drag Cd ≈ 1.33, lift amplitude ≈ ±0.5"),
    (1, "our channel is only 12 D high (8.3 % blockage) → expect St and Cd a little higher than these"),
], I(0.6), I(4.3), I(12.1), I(2.7), size=13, space=3)
D.callout(s, "Goal: reproduce the oscillating wake and measure St, Cd, Cl.", I(8.4), I(1.5), I(4.5), I(1.0), size=14, fill=LIGHT)

s = D.slide("A2", "The governing equations, term by term")
D.eq(s, r"\nabla\cdot\mathbf{u}=0", I(0.6), I(1.3), height=I(0.45))
D.text(s, "continuity (mass): what flows into any region flows out again — the fluid is incompressible", I(3.2), I(1.3), I(9.5), I(0.5), size=13)
D.eq(s, r"\frac{\partial\mathbf{u}}{\partial t}+\nabla\cdot(\mathbf{u}\otimes\mathbf{u})=-\frac{1}{\rho}\nabla p+\nu\nabla^2\mathbf{u}", I(0.6), I(1.95), height=I(0.8))
D.table(s, [
    ["term", "name", "meaning", "how we discretise it (Part D)"],
    ["∂u/∂t", "time derivative", "how fast the velocity in a cell changes", "forward Euler: (uⁿ⁺¹ − uⁿ)/Δt"],
    ["∇·(u⊗u)", "convection", "momentum carried by the flow through the cell faces", "flux through faces, face value central or upwind"],
    ["−∇p/ρ", "pressure gradient", "fluid is pushed from high to low pressure", "pressure difference across each face"],
    ["ν∇²u", "diffusion (viscosity)", "momentum spreads from fast to slow neighbours", "central difference across each face"],
], I(0.6), I(2.95), I(12.1), [I(1.3), I(2.0), I(4.4), I(4.4)], size=12)
D.bullets(s, [
    "Written per component with u = (u, v): two momentum equations (for u and for v) plus one constraint (continuity)",
    "Unknowns: u, v, p — three fields. But there are only two evolution equations (for u and v). There is no ∂p/∂t anywhere",
    "**Key idea for the whole method: the pressure is not marched in time. At each step it is whatever field is needed to make the new velocity satisfy ∇·u = 0.",
    "With D = U∞ = ρ = 1 the only number left is ν = 1/Re = 1/150",
], I(0.6), I(4.9), I(12.1), I(2.1), size=13)

# ======================================================== PART B  mesh
D.section("PART B", "Building the mesh from three numbers", "mesh_from_known_sizes.py: d_min, d_max, growth ratio — everything else is derived")

s = D.slide("B1", "Why not a uniform grid?")
D.bullets(s, [
    "The thin layer where the fluid slows to zero on the cylinder (the boundary layer) is about δ ≈ D/√Re ≈ 0.08 D thick",
    "To resolve it we need cells of about 0.015 D there. A uniform 0.015 grid over 24 D × 12 D is 1.3 million cells — and Δt would be tiny everywhere",
    "Far from the cylinder the flow varies over ~1 D; cells of 0.1–0.2 D are plenty",
    "**So: small cells on the walls, growing geometrically outward, uniform large cells elsewhere. Three numbers describe this:",
    (1, "d_min — the wall-normal size of the first cell on every wall (cylinder and channel walls)"),
    (1, "growth_ratio r — each cell in the layer is r × the previous one (1.1–1.2 is usual; we use 1.15)"),
    (1, "d_max — the cell size everywhere outside the layer; the layer stops when a cell would exceed d_max"),
    "Everything else (number of layer cells, block sizes, cell counts) is computed from these",
], I(0.6), I(1.3), I(12.1), I(3.0), size=14)
D.pic(s, SK_BL, I(0.6), I(4.1), width=I(7.6))
D.eq(s, r"n=\left\lfloor\frac{\ln(d_{\max}/d_{\min})}{\ln r}\right\rfloor+1", I(8.5), I(4.3), height=I(0.75), max_width=I(4.4))
D.eq(s, r"t_k=d_{\min}\frac{r^k-1}{r-1}\quad(k=0..n),\qquad T=t_n", I(8.5), I(5.3), height=I(0.7), max_width=I(4.4))

s = D.slide("B2", "Boundary-layer stack: the numbers for the fine mesh")
D.code(s, """def boundary_layer(d_min, d_max, r):
    n = int(np.floor(np.log(d_max / d_min) / np.log(r) + 1e-9)) + 1
    k = np.arange(n + 1)
    t = d_min * (r**k - 1.0) / (r - 1.0)     # cumulative thickness
    return dict(n=n, r=r, t=t, T=t[-1], d_last=d_min * r**(n - 1))

>>> boundary_layer(0.015, 0.12, 1.15)
n = 15 cells,  T = 0.714,  d_last = 0.106
t = [0, 0.015, 0.032, 0.052, 0.075, 0.101, 0.131, 0.166, 0.206,
     0.252, 0.305, 0.365, 0.435, 0.515, 0.608, 0.714]""", I(0.6), I(1.3), I(7.4), I(3.2), size=11)
D.bullets(s, [
    "15 cells grow from 0.015 to 0.106 (the last one still ≤ d_max = 0.12) over a thickness of 0.71 D",
    "The physical boundary layer (≈ 0.08 D) is covered by the first 4 cells — enough for a laminar Re = 150 demo",
    "The same stack is used on the cylinder (radially) and on the top/bottom walls (vertically)",
    "Changing d_min changes everything downstream: cell count, smallest cell h_min, and therefore Δt (Part F)",
], I(8.3), I(1.3), I(4.6), I(3.4), size=12)
D.table(s, [
    ["preset", "d_min", "d_max", "cells", "h_min", "Δt", "use"],
    ["--quick", "0.05", "0.30", "4,848", "0.06", "0.02", "smoke test, steady wake, ~1 min"],
    ["--fine", "0.015", "0.12", "26,444", "0.0177", "0.0045", "vortex shedding (this deck), ~5 min"],
    ["INPUT (file default)", "0.006", "0.06", "~95,000", "0.006", "0.001", "production; needs LU and patience"],
], I(0.6), I(4.75), I(12.1), [I(2.0), I(1.1), I(1.1), I(1.3), I(1.1), I(1.1), I(4.4)], size=11)
D.callout(s, "Rule of thumb: halve d_min → ×4 cells near walls and ÷4 in Δt (diffusive limit) → ×16 cost.", I(0.6), I(6.3), I(12.1), I(0.55), size=12, fill=LIGHT)

s = D.slide("B3", "Block layout: an O-ring inside a Cartesian channel")
D.pic(s, ROOT / "mesh_output" / "mesh_fine.png", I(0.6), I(1.25), height=I(4.3))
D.bullets(s, [
    "O-block (blue/cyan): 152 radial lines × (15 layer cells + transition cells). The layer cells follow the stack t_k; the transition cells go straight from the end of the layer to a square of half-side a",
    "Square: side 2a with a = 2.20 chosen automatically as the smallest value for which the size jump at the ring→square interface stays below max_jump = 2.0",
    "Cartesian blocks (grey): grid lines at spacing ≈ d_max, with the same layer stack against the top and bottom walls (orange)",
    "The blocks share nodes exactly along their interfaces → conforming mesh, no hanging nodes, every face has at most two cells",
    "**Checks printed by the mesher: no inverted cells (min area 3.1e-4), 0 unclassified boundary edges, interface size jumps 1.99 and 1.82 (limit 2.0)",
], I(6.2), I(1.35), I(6.8), I(4.4), size=12)
D.text(s, "Why an O-ring: cells stay aligned with the cylinder surface, so the wall-normal direction is a grid direction and the thin first cell is well shaped (0.015 × 0.021) instead of a staircase.", I(0.6), I(5.75), I(12.1), I(0.9), size=12, italic=True)

s = D.slide("B4", "Run the mesher")
D.code(s, """cd cylinder_re150
python mesh_from_known_sizes.py          # uses INPUT at the top of the file (d_min = 0.006)

# or let the solver build a preset for you (same mesher, different numbers):
python ns_staggered_euler.py --fine --build-mesh --steps 1 --no-gif
#   MESH_PRESETS["fine"] = dict(d_min=0.015, d_max=0.12, growth_ratio=1.15, max_jump=2.0, a=None)""", I(0.6), I(1.3), I(12.1), I(1.6), size=11)
D.text(s, "Console output you should see (abridged):", I(0.6), I(3.0), I(6), I(0.4), size=12, bold=True)
D.code(s, """[PASS] size jumps at block interfaces   ring->transition 1.992, transition->rectangles 1.823 (limit 2.0)
[PASS] no inverted cells                min area 3.146e-04
[PASS] conforming (no hanging nodes)    0 unclassified boundary edges
STAGGERED GRID
cells (P locations)     : 26,444
faces (velocity locs)   : 53,286  (796 boundary, 52,490 interior)
boundary patches        : ['cylinder', 'inlet', 'outlet', 'wall_bottom', 'wall_top']
    cylinder  152 faces   inlet  120   outlet  120   wall_bottom  202   wall_top  202
written to ./mesh_output/mesh_fine.(vtk|msh|png)
written to ./mesh_output/staggered_fine.(npz|png)""", I(0.6), I(3.4), I(12.1), I(2.4), size=10)
D.table(s, [
    ["file", "what it is", "used by"],
    ["mesh_output/staggered_fine.npz", "the arrays on the next slides — the only input the solver needs", "ns_staggered_euler.py"],
    ["mesh_output/mesh_fine.png / staggered_fine.png", "pictures of the mesh and of the P / face locations", "you (sanity check)"],
    ["mesh_output/mesh_fine.vtk / .msh", "the same mesh for ParaView / Gmsh", "optional"],
], I(0.6), I(5.9), I(12.1), [I(3.6), I(6.0), I(2.5)], size=11)

# ======================================================== PART C  staggered grid
D.section("PART C", "From quads to a staggered grid", "cells, faces, owner / neighbour — the data structure every operator uses")

s = D.slide("C1", "Cells and faces: the owner / neighbour description")
D.pic(s, SK_CELL, I(0.6), I(1.3), width=I(5.6))
D.bullets(s, [
    "Every quad is a cell P with centre x_P and area V_P",
    "Every edge shared by two quads is an interior face f; an edge on the boundary is a boundary face",
    "Each face stores exactly one pair of cells: the owner (first cell found) and the neighbour (the other one, −1 if boundary)",
    "The unit normal n_f points from owner to neighbour (outward on the boundary); L_f is the edge length; d_PN = |x_N − x_P|",
    "**How extract_staggered builds this: list all 4 edges of all cells as sorted node pairs (a, b); sort; identical pairs are the same face → first cell = owner, second = neighbour; a pair appearing once is a boundary face, named by which boundary its midpoint lies on",
], I(6.4), I(1.35), I(6.6), I(3.6), size=12)
D.code(s, """# edge key: sorted node pair -> unique faces
codes = a * n_node + b                 # a < b
order = np.argsort(codes)
# equal codes = same face: owner = first cell, neighbour = second (or -1)""", I(6.4), I(5.0), I(6.6), I(1.2), size=10)
D.text(s, "Because of this, i±1 / j±1 never appear in the solver. 'The cell on the other side of this face' is nei[f] (or own[f]).", I(0.6), I(5.9), I(5.6), I(1.0), size=12, italic=True)

s = D.slide("C2", "The arrays in staggered_fine.npz")
D.table(s, [
    ["array", "shape", "meaning"],
    ["cell_centers", "(Nc, 2)", "x_P — where p, u, v are stored  (Nc = 26,444)"],
    ["quads", "(Nc, 4, 2)", "the four corners of each cell (only for plotting)"],
    ["face_centers", "(Nf, 2)", "x_f — where U_n is stored  (Nf = 53,286)"],
    ["face_normals", "(Nf, 2)", "unit n_f, owner → neighbour, outward on the boundary"],
    ["face_lengths", "(Nf,)", "L_f, the 2-D 'area' of the face"],
    ["face_owner", "(Nf,)", "index of the owner cell"],
    ["face_neigh", "(Nf,)", "index of the neighbour cell, −1 on the boundary"],
    ["face_bc", "(Nf,)", "−1 interior, else index into bc_names"],
    ["bc_names", "(5,)", "['cylinder', 'inlet', 'outlet', 'wall_bottom', 'wall_top']"],
    ["D, xc, yc, d_min, d_max, a", "scalars", "geometry and sizing, for titles and checks"],
], I(0.6), I(1.3), I(7.6), [I(2.4), I(1.1), I(4.1)], size=11)
D.code(s, """S = load_staggered("mesh_output/staggered_fine.npz")
own, nei = S["face_owner"], S["face_neigh"]
interior = nei >= 0                       # 52,490 True
bnd      = ~interior                      #    796 True
masks = {name: S["face_bc"] == i
         for i, name in enumerate(S["bc_names"])}
# masks["inlet"], masks["cylinder"], ... boolean (Nf,)""", I(8.5), I(1.4), I(4.5), I(2.4), size=10)
D.bullets(s, [
    "All loops over cells or faces are numpy fancy-indexing: phi[own] is 'the owner's value on every face'",
    "A field on cells is an array of length Nc; a field on faces is an array of length Nf — never mix them up",
], I(8.5), I(4.0), I(4.5), I(2.0), size=12)

s = D.slide("C3", "Where the unknowns live, and why (MAC staggering)")
D.pic(s, ROOT / "mesh_output" / "staggered_fine.png", I(0.6), I(1.25), width=I(7.0))
D.pic(s, SK_CHK, I(7.8), I(1.3), width=I(5.2))
D.bullets(s, [
    "p at cell centres (blue dots, left) — one scalar per control volume",
    "U_n = u·n on face centres (red/green dots, right) — the velocity through each face",
    "u_P, v_P (Cartesian components at the centres) are also kept: the convective and viscous terms need both components on every face, and averaging the two cells is the simplest way to get them on a non-Cartesian mesh",
    "**Why staggered: continuity of a cell is an exact sum of its face velocities, and the pressure gradient driving a face velocity is the two-point difference (p_N − p_P)/d_PN across that very face. A p field that alternates +1, −1 cell to cell (a 'checkerboard') is invisible to a collocated 2Δx gradient but produces a huge face gradient here and is corrected at once.",
], I(7.8), I(3.3), I(5.2), I(3.7), size=11)

s = D.slide("C4", "Derived geometry, and the one function that does every sum")
D.eq(s, r"V_P=\frac{1}{2}\sum_{f\in P}s_{P,f}\,(\mathbf{x}_f\cdot\mathbf{n}_f)\,L_f\qquad d_{PN}=|\mathbf{x}_N-\mathbf{x}_P|\ \ (\mathrm{boundary:}\ 2|\mathbf{x}_f-\mathbf{x}_P|)\qquad h_{\min}=\min_P\sqrt{V_P}", I(0.6), I(1.3), height=I(0.6))
D.code(s, """def accumulate(flux, own, nei, Nc):   # out[P] = sum_{f in P} s_{P,f} * flux[f]
    out = np.zeros(Nc)
    np.add.at(out, own, flux)          # +flux to the owner (n points out of it)
    m = nei >= 0
    np.add.at(out, nei[m], -flux[m])   # -flux to the neighbour (n points into it)
    return out

contrib = 0.5 * (Fc[:, 0]*n[:, 0] + Fc[:, 1]*n[:, 1]) * L
vol  = accumulate(contrib, own, nei, Nc)   # cell areas (Gauss on x)
hmin = np.sqrt(vol).min()                  # 0.01774 on the fine mesh""", I(0.6), I(2.1), I(8.0), I(2.8), size=11)
D.bullets(s, [
    "s_{P,f} = +1 when P owns f (n points out of P), −1 when P is the neighbour (n points into P)",
    "np.add.at adds into repeated indices correctly (plain out[own] += flux would not)",
    "The area formula is the divergence theorem applied to the vector field x (∇·x = 2) — a first check that the data structure is right: the areas must be positive and sum to the domain area 24·12 − π/4 = 287.21",
    "Convection, diffusion, divergence and gradient (Part D) are all accumulate() with a different flux",
], I(0.6), I(5.0), I(8.0), I(2.0), size=12)
D.callout(s, "Gauss' theorem, the only piece of calculus we need:\n\n∫_V ∇·F dV = ∮ F·n dS ≈ Σ_f s_{P,f} (F·n)_f L_f\n\n'the net amount leaving a cell is the sum of what crosses its faces'", I(8.9), I(2.1), I(4.1), I(2.6), size=12, fill=LIGHT)

# ======================================================== PART D  discretisation
D.section("PART D", "Discretising the equations", "one term at a time, each one a sum over faces")

s = D.slide("D1", "The finite-volume recipe")
D.bullets(s, [
    "**1. Integrate the equation over one cell P and divide by its area V_P → an equation for the cell average",
    "**2. Every term of the form ∇·(something) becomes, by Gauss' theorem, a sum over the faces of (something)·n L_f",
    "**3. The 'something' is only known at cell centres; decide how to get it at the face (this is where the schemes differ)",
    "**4. Do it for every cell at once with accumulate()",
], I(0.6), I(1.3), I(12.1), I(2.0), size=15)
D.eq(s, r"\frac{1}{V_P}\int_{V_P}\nabla\cdot\mathbf{F}\,dV=\frac{1}{V_P}\sum_{f\in P}s_{P,f}\,(\mathbf{F}\cdot\mathbf{n})_f\,L_f", I(0.6), I(3.4), height=I(0.9))
D.table(s, [
    ["term", "F (what crosses the face)", "face value needed", "code"],
    ["convection ∇·(u φ)", "U_n φ  — flux times transported quantity", "φ_f (central / upwind)", "_convective"],
    ["diffusion ∇·(ν∇φ)", "ν ∂φ/∂n", "(φ_N − φ_P)/d_PN", "_laplacian"],
    ["continuity ∇·u", "U_n", "already on the face", "_div"],
    ["pressure gradient ∇p = ∇·(p I)", "p n", "½(p_P + p_N)", "_grad"],
], I(0.6), I(4.5), I(12.1), [I(3.0), I(4.0), I(2.8), I(2.3)], size=12)
D.text(s, "On a rectangular cell with faces ±x, ±y these sums reduce exactly to the familiar (φ_{i+1} − 2φ_i + φ_{i−1})/Δx² stencils. The face form is just the same thing written so that skewed, curved cells work too.", I(0.6), I(6.2), I(12.1), I(0.8), size=12, italic=True)

s = D.slide("D2", "Time derivative: forward (explicit) Euler")
D.eq(s, r"\frac{\partial\phi}{\partial t}\approx\frac{\phi_P^{\,n+1}-\phi_P^{\,n}}{\Delta t}\quad\Longrightarrow\quad\phi_P^{\,n+1}=\phi_P^{\,n}+\Delta t\cdot\mathrm{RHS}_P(\phi^{\,n})", I(0.6), I(1.35), height=I(0.8))
D.bullets(s, [
    "'Explicit' = the right-hand side is evaluated with the field we already have (level n). The new value is a formula, not an equation to solve",
    "The same line of code updates all 26,444 cells at once: u_new = u + dt * rhs",
    "Accuracy: first order — the error per unit time is proportional to Δt. With Δt = 0.0045 that is small compared with the spatial error of the mesh",
    "**Stability (Part F): an explicit step is only stable if Δt is small enough that fluid moves less than one cell per step and viscosity acts over less than one cell per step. The smallest cell decides.",
    "This is exactly the scheme of the Griebel book / the CaptainProton42 code; the 'projection' trick of Part E makes it work with the incompressibility constraint",
], I(0.6), I(2.5), I(12.1), I(3.5), size=14)
D.callout(s, "Reading the superscripts: φⁿ is the field at time t, φⁿ⁺¹ at t + Δt. Subscript P = cell P.", I(0.6), I(6.0), I(12.1), I(0.6), size=12, fill=LIGHT)

s = D.slide("D3", "Convection: momentum carried through the faces")
D.eq(s, r"[\nabla\cdot(\mathbf{u}\,\phi)]_P=\frac{1}{V_P}\sum_{f\in P}s_{P,f}\;U_{n,f}\,L_f\;\phi_f\qquad(\phi=u\ \mathrm{or}\ v)", I(0.6), I(1.25), height=I(0.8))
D.bullets(s, [
    "U_{n,f} L_f is the volume flow through face f — we have it (staggered unknown)",
    "φ_f is how much u (or v) each unit of that flow carries — we must interpolate it from the two cells:",
], I(0.6), I(2.15), I(7.6), I(0.9), size=12, space=2)
D.eq(s, r"\phi_f=\frac{1}{2}(\phi_P+\phi_N)-\frac{1}{2}\,\gamma\,\mathrm{sgn}(U_{n,f})\,(\phi_N-\phi_P)", I(0.6), I(2.95), height=I(0.55), max_width=I(6.0))
D.callout(s, "What we saw: on the same mesh 'donor' never shed (its extra viscosity holds the wake steady); 'central' sheds. Use central for this problem.", I(6.9), I(3.08), I(6.1), I(0.62), size=11, fill=LIGHT)
D.table(s, [
    ["γ", "scheme", "face value φ_f", "accuracy", "character", "--scheme"],
    ["0", "central", "average of the two cells", "2nd order", "no damping; wiggles if U h/ν > 2", "central"],
    ["1", "upwind (donor cell)", "the upstream cell's value", "1st order", "always bounded; extra viscosity ≈ ½|U_n| d_PN", "upwind"],
    ["CFL", "Griebel adaptive", "blend with γ = max|U_n| Δt / h_min ≈ 0.36", "between", "reference default", "donor"],
], I(0.6), I(3.75), I(12.1), [I(0.6), I(1.9), I(3.6), I(1.1), I(3.7), I(1.2)], size=10)
D.code(s, """phi_f = phi[own].copy()                                   # boundary faces: owner value (overwritten by BC below)
central = 0.5 * (phi[o] + phi[nn])                        # o = own[interior], nn = nei[interior]
upwind  = np.where(Un[interior] >= 0.0, phi[o], phi[nn])  # Un > 0 means flow from owner to neighbour
phi_f[interior] = (1 - gamma) * central + gamma * upwind
conv = -accumulate(Un * phi_f * L, own, nei, Nc) / vol    # minus sign: the term moves to the right-hand side""", I(0.6), I(5.15), I(12.1), I(1.45), size=10)
D.pic(s, SK_FACE, I(8.6), I(1.15), height=I(1.9))
D.text(s, "Boundary faces: φ_f = (U∞, 0) at the inlet, 0 on the walls and the cylinder, the owner value at the outlet.", I(0.6), I(6.65), I(12.1), I(0.4), size=11, italic=True)

s = D.slide("D4", "Diffusion: viscosity across the faces")
D.eq(s, r"[\nu\nabla^2\phi]_P=\frac{\nu}{V_P}\sum_{f\in P}s_{P,f}\left(\frac{\partial\phi}{\partial n}\right)_f L_f,\qquad\left(\frac{\partial\phi}{\partial n}\right)_f\approx\frac{\phi_N-\phi_P}{d_{PN}}", I(0.6), I(1.3), height=I(0.95))
D.code(s, """g = np.empty(Nf)
g[interior] = (phi[nei[interior]] - phi[own[interior]]) / d_pn[interior]   # centre to centre
g[bnd]      = (phi_bc[bnd] - phi[own[bnd]]) / (0.5 * d_pn[bnd])           # centre to wall
lap = accumulate(nu * g * L, own, nei, Nc) / vol""", I(0.6), I(2.4), I(8.0), I(1.3), size=11)
D.bullets(s, [
    "The normal gradient is the slope between the two cell centres — a central difference, second order on orthogonal cells",
    "On a boundary face the 'other side' is the wall itself (distance d/2 = centre-to-face), with the known wall value φ_bc: u_bc = U∞ at the inlet, 0 on walls and cylinder; v_bc = 0 on all of them; outlet: zero gradient (g = 0)",
    "On skewed ring cells the line P→N is not exactly normal to the face; we ignore that (no 'non-orthogonal correction'). It costs some accuracy there, not stability — but it is why the safety factor τ is kept at 0.4",
    "ν = 1/150: diffusion is weak compared with convection away from walls, strong inside the boundary layer where cells are small — exactly where it matters",
], I(0.6), I(3.9), I(12.1), I(3.0), size=13)

s = D.slide("D5", "Divergence and gradient")
D.eq(s, r"[\nabla\cdot\mathbf{u}]_P=\frac{1}{V_P}\sum_{f\in P}s_{P,f}\,U_{n,f}\,L_f", I(0.6), I(1.3), height=I(0.8))
D.code(s, """def _div(Un):
    return accumulate(Un * L, own, nei, Nc) / vol      # net outflow per unit area""", I(0.6), I(2.2), I(6.0), I(0.7), size=11)
D.eq(s, r"[\nabla p]_P=\frac{1}{V_P}\sum_{f\in P}s_{P,f}\,p_f\,\mathbf{n}_f\,L_f,\qquad p_f=\frac{1}{2}(p_P+p_N)\ \ (\mathrm{boundary:}\ p_P)", I(0.6), I(3.1), height=I(0.8))
D.code(s, """pf = p[own].copy();  pf[interior] = 0.5 * (p[own[interior]] + p[nei[interior]])
gx = accumulate(pf * n[:, 0] * L, own, nei, Nc) / vol
gy = accumulate(pf * n[:, 1] * L, own, nei, Nc) / vol""", I(0.6), I(4.0), I(8.0), I(1.0), size=11)
D.bullets(s, [
    "The divergence uses the face velocity directly — it is exact for the discrete flux field. This is the quantity we drive to zero",
    "The cell gradient (Green–Gauss) is only used to correct the cell-centred u, v (Part G). The face flux is corrected with the compact (p_N − p_P)/d_PN instead — see the next part for why that matters",
    "p_f = p_P on the boundary expresses ∂p/∂n = 0 there",
], I(0.6), I(5.2), I(12.1), I(1.8), size=13)

# ======================================================== PART E  pressure
D.section("PART E", "The pressure equation", "why there is one, how to derive it, how to solve it")

s = D.slide("E1", "Deriving the pressure equation (the heart of the method)")
D.text(s, "Step 1 — predict. Advance the momentum equation without the pressure (Burgers' equation). Call the result F, G (Griebel's names):", I(0.6), I(1.25), I(12.1), I(0.5), size=13, bold=True)
D.eq(s, r"F_P=u_P^{\,n}+\Delta t\,[\nu\nabla^2u-\nabla\cdot(\mathbf{u}u)]_P^{\,n},\qquad G_P=v_P^{\,n}+\Delta t\,[\nu\nabla^2v-\nabla\cdot(\mathbf{u}v)]_P^{\,n},\qquad U^*_{n,f}=F_f\,n_{x,f}+G_f\,n_{y,f}", I(0.6), I(1.7), height=I(0.6))
D.text(s, "Step 2 — demand. The true new flux must include the pressure push across the face, and the new field must be divergence-free:", I(0.6), I(2.45), I(12.1), I(0.5), size=13, bold=True)
D.eq(s, r"U^{\,n+1}_{n,f}=U^{*}_{n,f}-\Delta t\,\frac{p_N-p_P}{d_{PN}}\qquad\mathrm{and}\qquad\frac{1}{V_P}\sum_{f\in P}s_{P,f}\,U^{\,n+1}_{n,f}\,L_f=0", I(0.6), I(2.9), height=I(0.75))
D.text(s, "Step 3 — substitute the first into the second. The only unknowns left are the pressures of P and its neighbours:", I(0.6), I(3.75), I(12.1), I(0.5), size=13, bold=True)
D.eq(s, r"\sum_{f\in P}s_{P,f}\,\frac{L_f}{d_{PN}}\,(p_N-p_P)\;=\;\frac{V_P}{\Delta t}\,[\nabla\cdot\mathbf{u}^{*}]_P\qquad\mathrm{for\ every\ cell}\ P", I(0.6), I(4.2), height=I(0.9), fontsize=24)
D.bullets(s, [
    "This is a discrete Poisson equation ∇²p = (1/Δt) ∇·u* — the same one Griebel write, but obtained directly on our faces",
    "Read it as: 'find the pressure whose push through the faces removes exactly the divergence that the predictor created'",
    "Because F, G contained no pressure at all, the solution p is the pressure itself at the new time, not a correction to an old pressure",
    "Nc = 26,444 equations in 26,444 unknowns, each coupling a cell to its 4 neighbours → a sparse linear system A p = rhs",
], I(0.6), I(5.2), I(12.1), I(1.9), size=13)

s = D.slide("E2", "Assembling the matrix A")
D.code(s, """o, c = own[interior], nei[interior]
g = L[interior] / d_pn[interior]                 # one conductance per interior face
rows = np.r_[o, o, c, c]
cols = np.r_[c, o, o, c]
data = np.r_[g, -g, g, -g]                       # row o: +g at col c, -g at col o   (and vice versa)
A = sparse.coo_matrix((data, (rows, cols)), shape=(Nc, Nc)).tocsr()
# built ONCE in __init__ : the mesh never changes""", I(0.6), I(1.3), I(8.0), I(2.0), size=11)
D.bullets(s, [
    "Each interior face adds the same conductance g_f = L_f/d_PN to four entries: it couples P and N symmetrically → A is symmetric",
    "Row P: off-diagonals +g_f for each neighbour, diagonal −Σ g_f. Boundary faces add nothing (∂p/∂n = 0)",
    "**Consequence 1 — rows sum to zero: A·(constant) = 0. The pressure is only defined up to a constant; we fix it by subtracting the mean",
    "**Consequence 2 — a solution exists only if the right-hand side sums to zero: Σ_P V_P div(u*) = 0 = net flow out of the domain. Inflow must equal outflow before we solve (Part F)",
    "Check: A @ np.ones(Nc) is zero to round-off; A − A.T is exactly zero",
], I(0.6), I(3.5), I(8.0), I(3.4), size=13)
D.callout(s, "Analogy: A is the conductance matrix of a resistor network — one resistor L/d per face, p the node voltages, rhs the injected currents. Kirchhoff says the injected currents must sum to zero.", I(8.9), I(1.4), I(4.1), I(2.2), size=12, fill=LIGHT)
D.table(s, [["Nc", "nnz(A)", "memory"], ["26,444", "131,424", "1.7 MB"]], I(8.9), I(3.9), I(4.1), [I(1.3), I(1.4), I(1.4)], size=11)

s = D.slide("E3", "Solving A p = rhs — three ways, one default")
D.code(s, """rhs = (1.0 / dt) * vol * _div(Un_star)
rhs -= rhs.mean()                       # exactly compatible (after the outlet balance)
if poisson == "lu":
    rhs[0] = 0.0;  p = lu.solve(rhs)    # lu = splu(A, row 0 -> [1 0 0 ...]), built once
elif poisson == "cg":
    p, info = cg(A_meanfree, rhs, M=jacobi, rtol=1e-8, maxiter=2000)
else:                                   # "sor"
    for it in range(max_it): Gauss-Seidel sweep with omega = 1.7 until the residual < eps
p -= p.mean()                           # pin the free constant""", I(0.6), I(1.3), I(8.0), I(2.4), size=10)
D.table(s, [
    ["--poisson", "idea", "work per time step (26k cells)", "when to use"],
    ["sor", "sweep cell by cell, over-relax (ω = 1.7). Griebel's original", "thousands of sweeps, seconds", "to see the textbook method"],
    ["cg", "conjugate gradient, Jacobi preconditioner, on the mean-free operator", "≈ 1000 iterations, ~300 ms", "very large meshes (no factorisation memory)"],
    ["lu (default)", "pin one cell, factorise A once (SuperLU), two triangular solves per step", "2 ms", "whenever the mesh fits in memory"],
], I(0.6), I(3.9), I(12.1), [I(1.3), I(5.0), I(2.9), I(2.9)], size=11)
D.bullets(s, [
    "'Pin one cell': replace row 0 of A by the identity row. That removes the constant null space; because the other rows sum to zero, a mean-free rhs is still solved exactly by all of them",
    "LU is exact to round-off: after the correction, max|∇·u| = 1e-12. CG/SOR stop at a tolerance and leave 1e-8 … 1e-6",
    "The factorisation costs 60 ms once and 1.6 M non-zeros — negligible. It made the 26,600-step run take 6 min instead of 3 h",
], I(0.6), I(5.5), I(12.1), I(1.6), size=12)
D.callout(s, "Lesson: the pressure solve is 98 % of the cost of an explicit projection code. Pick the solver before anything else.", I(8.9), I(1.4), I(4.1), I(1.2), size=12, fill=LIGHT)

# ======================================================== PART F  BC, dt, IC
D.section("PART F", "BCs, time step, initial condition", "the parts that decide whether the run is right, stable, and actually sheds")

s = D.slide("F1", "Boundary conditions — where each one is applied")
D.table(s, [
    ["patch", "velocity", "how it enters the code", "pressure"],
    ["inlet (x = 0)", "u = U∞, v = v_in(t)", "U_n set directly on inlet faces; u_bc in the Laplacian; φ_f at the face", "∂p/∂n = 0 (no face in A)"],
    ["outlet (x = 24)", "∂u/∂n = 0, ∂v/∂n = 0", "U_n* taken from the owner cell; then shifted so outflow = inflow", "∂p/∂n = 0"],
    ["wall_top, wall_bottom", "u = v = 0", "U_n = 0; φ_bc = 0 in the Laplacian; φ_f = 0 in convection", "∂p/∂n = 0"],
    ["cylinder", "u = v = 0", "same as walls", "∂p/∂n = 0"],
], I(0.6), I(1.3), I(12.1), [I(2.0), I(2.2), I(5.4), I(2.5)], size=11)
D.code(s, """def _enforce_Un_bc(Un):
    Un[inlet] = U_inf * n[inlet, 0] + v_in(t) * n[inlet, 1]
    Un[wall_top] = Un[wall_bottom] = Un[cylinder] = 0.0
    return Un                                   # outlet faces untouched

def _balance_outlet(Un):                        # inflow must equal outflow
    oth = bnd & ~outlet                                      # every boundary face except the outlet
    Q_other = np.sum(Un[oth] * L[oth])                       # their net outward flux
    Q_out   = np.sum(Un[outlet] * L[outlet])
    Un[outlet] += -(Q_other + Q_out) / np.sum(L[outlet])     # uniform shift
    return Un""", I(0.6), I(3.3), I(7.6), I(2.6), size=10)
D.callout(s, "What went wrong before _balance_outlet existed:\nthe run looked fine, but max|div| sat at 1e-2 in the smallest cells. The ~0.1 % mass imbalance of the zero-gradient outlet had nowhere to go and was spread over the whole domain as a constant source.\n\nWith it: 1e-12. Always check Σ rhs = 0 for a Neumann problem.", I(8.5), I(3.3), I(4.5), I(2.6), size=11, fill=LIGHT)
D.text(s, "Order of application in a step: BCs on U_n → predictor → BCs on U_n* → balance outlet → Poisson → correct → BCs on U_n again (the correction only touches interior faces).", I(0.6), I(6.2), I(12.1), I(0.8), size=12, italic=True)

s = D.slide("F2", "Choosing the time step")
D.eq(s, r"\Delta t=\tau\,\min\left(\frac{h_{\min}}{|u|_{\max}},\ \frac{Re\,h_{\min}^{2}}{4}\right),\qquad h_{\min}=\min_P\sqrt{V_P}=0.0177,\qquad\tau=0.4", I(0.6), I(1.3), height=I(0.85))
D.bullets(s, [
    "**h_min / |u|_max — the CFL (Courant) condition: in one step fluid may not cross more than one cell. Violating it makes explicit convection blow up",
    "**Re h_min² / 4 = h_min²/(4ν) — the diffusive limit of explicit Euler in 2-D: viscosity may not 'reach' further than one cell per step",
    "**τ — safety factor for what the two simple limits ignore (their interaction, the nonlinear term, skewed cells). Griebel use 0.5; 0.4 here",
    "Both limits are set by the smallest cell, which sits on the cylinder. The far field has 0.12 cells and 'wastes' most of each step — the price of a single global explicit Δt",
    "|u|_max ≈ 1.4 after the start, 1.5–1.6 during shedding → Δt stays between 0.0044 and 0.0047: adaptive in name, practically constant",
    "Fine mesh: 25,700 steps to t = 120, ≈ 1,000 steps per shedding period. --dt fixes it by hand, --tau changes the margin",
], I(0.6), I(2.3), I(7.2), I(4.6), size=12)
D.pic(s, DTH, I(8.0), I(2.3), width=I(5.0))
D.code(s, """def stable_dt():
    umax = max(np.abs(Un).max(), U_inf)
    return tau * min(hmin / umax, Re * hmin**2 / 4)""", I(8.0), I(4.7), I(5.0), I(1.0), size=10)

s = D.slide("F3", "Initial condition, and why a symmetric start never sheds")
D.bullets(s, [
    "Start from u = U∞, v = 0 everywhere (impulsive start) and project once (Parts E/G) so the first step begins divergence-free",
    "**Problem we hit: the mesh is exactly mirror-symmetric about the cylinder centre-line y = y_c, and so is every operator. A symmetric start therefore stays symmetric forever in exact arithmetic — Cl printed as −0.000 for 120 time units while the wake grew into the long, steady (and unstable) symmetric bubble",
    "Round-off eventually breaks the symmetry, but it has to grow by 10¹³ — on our mesh the oscillation only started at tU/D ≈ 120 (left plot)",
    "**Fix: add a small, localised, non-symmetric kick that the physics will amplify if shedding is real — a transverse velocity bump one diameter behind the cylinder (at x_c + D, y_c), plus a 2-time-unit inlet pulse",
    "With it, shedding is established by tU/D ≈ 20–30 (right plot). The seed is washed out within a few time units; both runs end on the same limit cycle (Cl = ±0.75 on that domain). That is the test that the seed only triggered, and did not create, the oscillation",
], I(0.6), I(1.3), I(12.1), I(2.7), size=12)
D.code(s, """r2 = (P[:, 0] - (xc + D))**2 + (P[:, 1] - yc)**2
v += perturb * U_inf * np.exp(-r2 / 0.5**2)     # perturb = 0.1   (CLI: --perturb, 0 = off)
v_in(t) = perturb * U_inf * sin(pi * t / 2)  for t < 2""", I(0.6), I(3.7), I(8.4), I(0.85), size=10)
D.pic(s, ART / "forces_noseed_40x10.png", I(0.6), I(4.95), height=I(2.05))
D.pic(s, ART / "forces_seed_40x10.png", I(6.9), I(4.95), height=I(2.05))
D.text(s, "no seed (--perturb 0): onset at t ≈ 120   [earlier 40 D × 10 D domain]", I(0.6), I(4.65), I(6), I(0.3), size=11, bold=True)
D.text(s, "seed 0.1: onset at t ≈ 20, same final amplitude   [same domain]", I(6.9), I(4.65), I(6), I(0.3), size=11, bold=True)

# ======================================================== PART G  the step
D.section("PART G", "One complete time step", "and the loop around it")

s = D.slide("G1", "One time step, in order")
D.pic(s, SK_FLOW, I(0.5), I(1.2), height=I(5.8))
D.bullets(s, [
    "**1 Δt** from the CFL/diffusive rule (F2); γ for the donor scheme from the same CFL",
    "**2 BCs** on the face flux U_n (F1)",
    "**3 Predictor** F = u + Δt(ν∇²u − ∇·(uu)), G likewise (D2–D4): explicit, no pressure",
    "**4 Provisional flux** U_n* = F_f n_x + G_f n_y from the face-averaged F, G; impose BCs; balance the outlet",
    "**5 Right-hand side** rhs = (V/Δt) div(U_n*), minus its mean",
    "**6 Solve** A p = rhs (E3) — the only linear system in the step",
    "**7 Correct the faces** U_n = U_n* − Δt (p_N − p_P)/d_PN → div(U_n) = 1e-12",
    "**8 Correct the cells** u = F − Δt ∂p/∂x, v = G − Δt ∂p/∂y with the Green–Gauss gradient (D5)",
    "**9 Advance** t += Δt; compute Cd, Cl on the cylinder faces; save a frame every N steps",
], I(4.6), I(1.3), I(8.4), I(4.5), size=13)
D.callout(s, "Why faces are corrected with (p_N − p_P)/d_PN and cells with the Green–Gauss gradient: the face formula is the one that appears in the Poisson equation, so it zeroes the divergence exactly. The cell gradient only feeds the next predictor and does not need to be exact.", I(4.6), I(5.7), I(8.4), I(1.2), size=11, fill=LIGHT)

s = D.slide("G2", "The same step as code (step(), lightly abridged)")
D.code(s, """def step(self, dt=None):
    dt = dt or self.stable_dt()                                   # 1
    gamma = self._gamma(dt)
    self.Un = self._enforce_Un_bc(self.Un)                        # 2

    cu, cv = self._convective(self.u, self.v, self.Un, gamma)     # 3  (-div(u u), -div(u v))
    lu = self._laplacian(self.u, U_inf_dirichlet_u=True)          #    nu lap u
    lv = self._laplacian(self.v, U_inf_dirichlet_u=False)         #    nu lap v
    F = self.u + dt * (lu + cu)
    G = self.v + dt * (lv + cv)

    uf, vf = self._face_velocity_cartesian(F, G, self.Un, gamma=0.0)   # 4
    Un_star = uf * n[:, 0] + vf * n[:, 1]
    Un_new = self._project_Un(Un_star, dt)      # BCs, balance outlet, rhs, solve, face correction  (5-7)
    p = self._p_corr

    gx, gy = self._grad(p)                                        # 8
    self.u = F - (dt / self.rho) * gx
    self.v = G - (dt / self.rho) * gy
    self.p = p                   # the Poisson solution IS the pressure (F, G had none)
    self.Un = Un_new
    self.t += dt                                                  # 9
    Cd, Cl = self.force_coefficients()
    return dict(t=self.t, dt=dt, gamma=gamma, max_div=abs(self._div(self.Un)).max(),
                max_u=abs(self.u).max(), Cd=Cd, Cl=Cl)""", I(0.6), I(1.3), I(12.1), I(5.6), size=10)

s = D.slide("G3", "The loop, the forces, and the restart")
D.code(s, """solver = GriebelStaggeredNS(S, Re=150, scheme="central", poisson="lu", perturb=0.1)
hist, k = [], 0
while solver.t < t_end:
    k += 1
    info = solver.step()                        # one step (G2)
    hist.append(info)                           # t, dt, max_div, max_u, Cd, Cl
    if k % every == 0:                          # every = 40 with --fine (--frame-every)
        save_wake_frame(out / f"wake_{k:05d}_t{solver.t:08.4f}.png", solver)
save_fields(out / "fields_central_fine.npz", solver, hist)   # u, v, p, Un, t + history
plot_forces(out / "forces_central_fine.png", hist)           # Cd(t), Cl(t), St""", I(0.6), I(1.3), I(7.6), I(2.8), size=10)
D.eq(s, r"(C_d,\,C_l)=\frac{2}{\rho U_\infty^2 D}\sum_{f\in\mathrm{cyl}}(p_P\,\mathbf{n}_f+\mu\frac{u_{t,P}}{d_f/2}\,\mathbf{t}_f)L_f", I(0.6), I(4.3), height=I(0.8), max_width=I(7.6))
D.bullets(s, [
    "Forces on the cylinder = pressure (p_P n L on each cylinder face) + wall shear (viscosity × tangential velocity of the first cell / half its distance). Divide by ½ρU∞²D → Cd (x) and Cl (y)",
    "Cl oscillating in time is the signature of shedding; its frequency gives St; Cd oscillates at twice that frequency",
    "**Restart: fields_*.npz holds u, v, p, U_n, t and the history; --restart continues from it on the same mesh (we ran 0→150, then 150→240)",
], I(0.6), I(5.2), I(7.6), I(1.9), size=12)
D.callout(s, "Numbers on the fine mesh (LU, central):\n• 12 ms per step\n• 25,725 steps to t = 120 → 5.6 min\n• frames ≈ 0.5 s each (266 at --frame-every 100)\n• run to t = 240 with --restart: +5 min", I(8.5), I(1.4), I(4.5), I(2.4), size=12, fill=LIGHT)

# ======================================================== PART H  running
D.section("PART H", "Running it", "commands, what the console shows, which files appear")

s = D.slide("H1", "Commands, from smoke test to shedding")
D.code(s, """cd cylinder_re150

# 1. smoke test (6,630-cell mesh, 40 time units, ~1 min): is everything wired up?
python ns_staggered_euler.py --quick --scheme donor --t-end 40 --demo

# 2. the real thing: fine mesh, central convection, 120 time units (~6 min + frames)
python ns_staggered_euler.py --fine --scheme central --demo
#    -> output_ns/fields_central_fine.npz, plot_central_fine.png, forces_central_fine.png,
#       wake_*.png, cylinder_wake_central.gif, demo/cylinder_wake_griebel.gif

# 3. more time, same mesh (continue from the saved state)
python ns_staggered_euler.py --fine --scheme central --t-end 240 \\
       --restart output_ns/fields_central_fine.npz --gif-from 220

# 4. experiments
python ns_staggered_euler.py --fine --scheme upwind         # watch the wake stay steady
python ns_staggered_euler.py --fine --scheme central --perturb 0   # watch Cl stay 0 for 120 units
python ns_staggered_euler.py --fine --poisson cg --steps 200       # feel the solver cost
python ns_staggered_euler.py --fine --Re 40 --t-end 60             # below the shedding threshold""", I(0.6), I(1.3), I(12.1), I(4.4), size=11)
D.bullets(s, [
    "Every run prints its settings first (next slide) — check mesh, scheme, Re and poisson before waiting six minutes",
    "The mesh is built once per preset and cached in mesh_output/; delete the npz or pass --build-mesh to regenerate",
], I(0.6), I(5.9), I(12.1), I(1.0), size=13)

s = D.slide("H1b", "Command-line flags")
D.table(s, [
    ["flag", "meaning", "default"],
    ["--quick / --fine", "mesh preset (built on first use with mesh_from_known_sizes, then reused from mesh_output/)", "INPUT mesh (d_min 0.006)"],
    ["--build-mesh / --mesh PATH", "force a rebuild / use another staggered_*.npz", "—"],
    ["--scheme", "central | upwind | donor (adaptive γ)", "donor"],
    ["--t-end / --steps", "stop at physical time / at step count, whichever comes first", "120 / 200000 with --fine"],
    ["--dt / --tau", "fixed time step / safety factor in the adaptive rule", "adaptive / 0.4"],
    ["--Re / --U-inf", "Reynolds number / inlet speed", "150 / 1"],
    ["--poisson", "lu | cg | sor", "lu"],
    ["--perturb", "shedding-trigger amplitude × U∞ (near-wake seed + inlet pulse); 0 = off", "0.1"],
    ["--frame-every", "save a wake frame every N steps", "40 with --fine (100 used for the long runs here)"],
    ["--gif-from / --gif-max-frames / --gif-fps", "only frames with t ≥ this go into the GIF / cap / speed", "t_end − ~4 periods / 60 / 8"],
    ["--no-gif", "skip frames and GIF (fastest)", "off"],
    ["--restart PATH", "continue from a fields_*.npz of a previous run on the same mesh", "—"],
    ["--out-dir", "where everything is written", "cylinder_re150/output_ns"],
    ["--demo", "also copy the GIF and forces plot to cylinder_re150/demo/", "off"],
], I(0.6), I(1.3), I(12.1), [I(3.4), I(5.9), I(2.8)], size=12)

s = D.slide("H2", "What the console shows, and how to read it")
D.code(s, """GRIEBEL NS on known-sizes staggered mesh
mesh     : mesh_output/staggered_fine.npz
cells    : 26,444   faces: 53,286   hmin=0.01774
sizing   : d_min=0.015  d_max=0.12  a=2.204
scheme   : central   Re=150  nu=0.00666667  poisson=lu
steps    : <=200000  t_end=120
------------------------------------------------------------------------
step     1  t=0.0047  dt=0.004719  g=0.000  max|div|=5.3e-12    max|u|=1.87   Cd=320.1  Cl=-6.3
step  1500  t=6.9399  dt=0.004719  g=0.000  max|div|=4.6e-12    max|u|=1.396  Cd=1.262  Cl=-0.100
step  5840  t=27.199  dt=0.004438  g=0.000  max|div|=7.9e-12    max|u|=1.600  Cd=1.556  Cl=+0.594
step 25725  t=120.00  dt=0.004673  g=0.000  max|div|=7.8e-12    max|u|=1.519  Cd=1.533  Cl=+0.066
------------------------------------------------------------------------
done in 335.1s  (13.0 ms/step)  t=120.005
forces   : Cd_mean=1.564  Cl_amp=0.712  St=0.207 (ref ~0.18)
wrote output_ns/cylinder_wake_central.gif  (44 of 44 frames, 5.6 MB)""", I(0.6), I(1.3), I(12.1), I(3.6), size=10)
D.table(s, [
    ["column", "what it tells you", "healthy value"],
    ["dt", "the adaptive step (F2)", "0.0045–0.0047 after the start"],
    ["g (γ)", "upwind blend: 0 for central, 1 for upwind, CFL for donor", "as chosen"],
    ["max|div|", "largest cell divergence after the projection — THE correctness check", "~1e-11 (lu); 1e-8 (cg)"],
    ["max|u|", "fastest fluid anywhere", "≈ 1.4 steady wake, ≈ 1.5–1.6 while shedding; growing without bound = instability"],
    ["Cd, Cl", "drag and lift coefficients", "huge for a few steps (impulsive start), then Cd ≈ 1.6, Cl oscillating"],
], I(0.6), I(5.0), I(12.1), [I(1.5), I(6.0), I(4.6)], size=10)

s = D.slide("H3", "Files produced")
D.table(s, [
    ["file (output_ns/)", "content", "look at it to…"],
    ["wake_{step}_t{time}.png", "two panels: 'Vorticity plot' (ω_z) and 'Velocity plot' (|u| with arrows)", "see the flow at one instant"],
    ["cylinder_wake_central.gif", "the last ~4 shedding periods, looped (≤ 60 frames)", "see the shedding"],
    ["forces_central_fine.png", "Cd(t), Cl(t), late-time Cd mean, Cl amplitude, St", "measure the result"],
    ["plot_central_fine.png", "final Velocity plot and max|div|(t) on a log scale", "confirm mass conservation held all run"],
    ["fields_central_fine.npz", "u, v, p, Un, t, hist_t, hist_Cd, hist_Cl, hist_div, hist_umax", "post-process yourself / --restart"],
    ["demo/cylinder_wake_griebel.gif (--demo)", "copy of the GIF for the README", "share"],
], I(0.6), I(1.3), I(12.1), [I(3.4), I(5.4), I(3.3)], size=11)
D.code(s, """import numpy as np
d = np.load("output_ns/fields_central_fine.npz")
t, Cl = d["hist_t"], d["hist_Cl"]
late = t > 0.6 * t[-1]
zc = np.where(np.diff(np.sign(Cl[late] - Cl[late].mean())) > 0)[0]   # upward zero crossings
St = 1.0 / np.mean(np.diff(t[late][zc]))                             # f D / U with D = U = 1
print("St =", St, " Cl amp =", 0.5 * np.ptp(Cl[late]), " Cd mean =", d["hist_Cd"][late].mean())""", I(0.6), I(4.3), I(12.1), I(2.0), size=10)

# ======================================================== PART I  checking
D.section("PART I", "Checking the answer", "three levels: is it consistent, is it converged, is it physical")

s = D.slide("I1", "Level 1 — internal consistency checks (do these first)")
D.table(s, [
    ["check", "how", "expected", "if not"],
    ["mesh is valid", "mesher [PASS] lines", "no inverted cells, 0 unclassified edges", "increase a / max_jump, reduce d_max"],
    ["cell areas", "vol.sum() vs 24·12 − π/4 = 287.21", "equal to 1e-10", "owner/neighbour or normal orientation wrong"],
    ["A is symmetric & singular", "abs(A − A.T).max(), A @ ones", "0, 0", "assembly sign error"],
    ["uniform flow is preserved", "run with no cylinder forces: u stays U∞", "exactly", "BC or flux error"],
    ["mass conservation", "max|div| column", "1e-12 (lu)", "outlet not balanced / solver tolerance"],
    ["symmetry (perturb 0)", "Cl", "exactly 0", "an operator breaks mirror symmetry"],
    ["Δt", "dt column vs τ·min(h/u, Re h²/4)", "equal", "—"],
], I(0.6), I(1.3), I(12.1), [I(2.6), I(3.6), I(2.9), I(3.0)], size=11)
D.bullets(s, [
    "These cost nothing and catch 90 % of bugs. All of them were used while writing this solver — two of them (mass conservation, symmetry) found real bugs",
    "max|div| is printed every 20 steps precisely so that a broken projection is noticed in the first minute, not after six",
], I(0.6), I(4.6), I(12.1), I(1.5), size=13)

s = D.slide("I2", "Level 2 & 3 — is it converged, is it physical?")
D.pic(s, ART / "forces_griebel_fine.png", I(0.6), I(1.3), height=I(3.4))
D.table(s, [
    ["quantity", "this run (fine, central)", "literature, unbounded cylinder", "why different"],
    ["St", "0.207", "0.18 (Williamson 1996)", "8.3 % blockage raises St; 1st-order time"],
    ["Cd mean", "1.56", "1.33", "blockage (+10–15 %), coarse far field"],
    ["Cl amplitude", "0.71", "≈ 0.5", "blockage; central scheme on 0.12 cells"],
    ["Cd oscillation", "2 × lift frequency", "2 × lift frequency", "✓ one vortex from each side per cycle"],
    ["onset", "Hopf-type exponential growth → limit cycle", "same", "✓"],
], I(6.3), I(1.4), I(6.7), [I(1.5), I(1.6), I(1.9), I(1.7)], size=10)
D.bullets(s, [
    "**Converged in time? Cl amplitude and period constant for the last ~15 cycles → yes (a limit cycle)",
    "**Converged in space? Not tested here. The honest test is to rerun with d_min = 0.010 and d_max = 0.08 and see St, Cd move by less than a few %. Expect the far-field d_max to matter most for Cd",
    "**Physical? The qualitative picture (alternating street, St ≈ 0.2, Cd ≈ 1.3–1.6, Cd at 2f) is right; the quantitative gaps are explained by the 12 D channel height and first-order Euler — both deliberate simplifications of the setup, not bugs",
    "To compare with the literature properly: widen the channel to 30 D (blockage 3 %) and refine the far field",
], I(6.3), I(4.35), I(6.7), I(2.7), size=11, space=3)

# ======================================================== PART J  pictures
D.section("PART J", "Making and reading the pictures", "what a frame is, and what it is not")

s = D.slide("J1", "How a frame is made — and what it is not")
D.code(s, """def save_wake_frame(path, solver):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True)
    _draw_vorticity(ax0, solver);  ax0.set_title("Vorticity plot")          # top
    _draw_velocity(ax1, solver, vmax=1.6);  ax1.set_title("Velocity plot")  # bottom: |u|
    ...
def _draw_field(ax, solver, field, vlim, quiver=True):
    coll = PolyCollection(S["quads"], array=field, cmap="RdBu_r")  # one colour per CELL
    coll.set_clim(-vlim, vlim); ax.add_collection(coll)
    if quiver:
        idx = np.arange(0, Nc, Nc // 400)                 # ~400 sampled cells
        ax.quiver(P[idx, 0], P[idx, 1],
                  solver.u[idx], solver.v[idx])           # the velocity VECTOR there""", I(0.6), I(1.3), I(7.2), I(3.0), size=10)
D.bullets(s, [
    "Colour: every mesh cell is filled with its own value at this instant — 'Vorticity plot': ω_z = ∂v/∂x − ∂u/∂y (red/blue = the two senses of rotation); 'Velocity plot': the speed |u| = √(u²+v²) (free stream ≈ 1, wake dark, shoulders bright). Titles and colour-bar labels carry only the plot type",
    "Arrows: the velocity vector (u_P, v_P) of ~400 sampled cells at this instant — direction and magnitude of the flow there",
    "Time: one frame every 40 steps by default (≈ 0.18 time units, ~27 frames per shedding period; the long runs here used --frame-every 100); the GIF loops the last ~4 periods (--gif-from), the MP4 is the whole history",
    "**What it is NOT: there are no streamlines or particle paths (plt.streamplot is never called). Nothing is tracked from one frame to the next — the next frame is simply the next saved state, coloured and arrowed again",
], I(0.6), I(4.4), I(7.2), I(2.6), size=11)
D.pic(s, ART / "ns_fine_shedding_frame.png", I(8.0), I(1.3), height=I(5.6))

s = D.slide("J2", "Reading the near wake on the actual mesh")
D.pic(s, ZOOM, I(0.6), I(1.3), width=I(12.1))
D.bullets(s, [
    "Cell values drawn with their edges: the solution is smooth across the O-ring → square → Cartesian block interfaces — no checkerboard, no jumps at the size steps",
    "Left: the recirculating wake (|u| ≈ 0, dark) is pulled to one side — this frame is at a lift maximum",
    "Right: the two shear layers (opposite-sign vorticity) roll up alternately into the street; each vortex is about one diameter across and resolved by ~10 cells",
], I(0.6), I(6.0), I(12.1), I(1.1), size=11)

# ======================================================== PART K  pitfalls etc
D.section("PART K", "Pitfalls, exercises, glossary", "what went wrong for us, so it doesn't for you")

s = D.slide("K1", "Pitfalls we hit — symptom → cause → fix")
D.table(s, [
    ["symptom", "cause", "fix", "how we noticed"],
    ["Cd = 500 while the flow looked fine", "p was accumulated (p += p_corr) although F, G carry no old pressure", "p = p_corr: the Poisson solution is the pressure", "unphysical number in the forces plot"],
    ["max|div| stuck at 1e-2", "zero-gradient outlet → inflow ≠ outflow → Neumann problem unsolvable; the mean was spread as a uniform source", "shift outlet flux so Σ rhs = 0 (_balance_outlet)", "the max|div| column; largest in the smallest cells"],
    ["no shedding for 120 time units", "exactly symmetric mesh + operators; Cl ≡ 0", "small localised seed + inlet pulse (--perturb)", "Cl printed as −0.000 forever"],
    ["still no shedding with --scheme donor", "upwind blend adds viscosity ≈ ½|U|h → effective Re too low", "use central for this problem", "compare donor vs central at t = 150"],
    ["390 ms per step, hours per run", "Jacobi-CG needs ~1000 iterations on 37k cells", "sparse LU once, 2 ms per solve", "cProfile: 98 % in cg()"],
    ["CG diverged on the big mesh", "pinned row made the matrix non-symmetric for CG", "solve the singular symmetric system with a mean projection (or use LU)", "info ≠ 0, divergence growing"],
    ["20 MB GIF", "266 frames at 900 px", "window to ~4 periods, ≤ 60 frames, 640 px", "file size"],
    ["frames overwrote each other", "file name had t to 2 decimals; early steps collided", "include the step index in the name", "fewer frames than expected"],
], I(0.6), I(1.3), I(12.1), [I(2.6), I(3.9), I(3.2), I(2.4)], size=12)
D.bullets(s, [
    "The common thread: every one of these was caught by a cheap printed diagnostic (max|div|, Cd, Cl, ms/step) or by looking at the picture — not by reading code. Print them from the first step on",
], I(0.6), I(6.1), I(12.1), I(0.8), size=13)

s = D.slide("K2", "Exercises — each one changes one thing")
D.table(s, [
    ["try", "command", "what to look for", "what you learn"],
    ["lower Re", "--fine --Re 40 --t-end 80", "Cl decays to 0; steady symmetric wake with two bubbles", "the shedding threshold (Re ≈ 47)"],
    ["upwind", "--fine --scheme upwind", "no shedding; shorter, fatter wake", "numerical viscosity"],
    ["no trigger", "--fine --scheme central --perturb 0", "Cl = 0 for ~120 units, then growth", "symmetry and instability"],
    ["bigger step", "--fine --tau 0.9", "max|u| grows, NaN", "the CFL / diffusive limits"],
    ["coarser mesh", "--quick --scheme central --t-end 120", "weak or no shedding", "resolution of the shear layers"],
    ["solver cost", "--fine --poisson cg --steps 100  vs  --poisson lu", "ms/step", "where the time goes"],
    ["blockage", "edit INPUT: y_max=30, yc=15 (30 D high)", "St and Cd move toward 0.18 / 1.33", "wall confinement"],
    ["frame content", "edit _draw_velocity to colour by p", "pressure minima at the vortex cores", "what the fields mean"],
], I(0.6), I(1.3), I(12.1), [I(1.5), I(4.2), I(3.7), I(2.7)], size=11)
D.bullets(s, [
    "**Natural next steps for the code: treat diffusion implicitly (removes the h² limit on Δt), second-order time stepping (Adams–Bashforth / RK), non-orthogonal correction in _laplacian, a proper outflow (convective) condition, mesh-refinement study",
], I(0.6), I(5.2), I(12.1), I(1.5), size=12)

s = D.slide("K3", "Glossary")
D.table(s, [
    ["term", "meaning here"],
    ["cell P / neighbour N / face f", "control volume where p lives / the cell across a face / the edge between two cells"],
    ["owner, neighbour", "the two cells of a face; the normal points owner → neighbour; neighbour = −1 on the boundary"],
    ["staggered (MAC) grid", "pressure in cells, normal velocity U_n on faces"],
    ["U_n", "velocity component through a face, u·n; the quantity that is made divergence-free"],
    ["finite volume (FV)", "discretisation by integrating over cells and summing fluxes over faces (Gauss' theorem)"],
    ["central / upwind / donor cell", "face value = average of both cells / the upstream cell / Griebel's blend of the two"],
    ["explicit Euler", "new = old + Δt × rate evaluated with the old field"],
    ["predictor F, G", "velocity after convection and diffusion but before the pressure acts"],
    ["projection", "solving a Poisson equation for p and subtracting Δt∇p so that ∇·u = 0"],
    ["CFL number", "|u| Δt / h: how many cells the fluid crosses per step; must be < 1 for an explicit scheme"],
    ["h_min, d_PN, L_f, V_P", "smallest cell size / distance between two cell centres / face length / cell area"],
    ["Neumann condition", "∂φ/∂n = 0 on the boundary (zero gradient); for p on every boundary here"],
    ["Cd, Cl, St", "drag and lift coefficients F/(½ρU∞²D); Strouhal number f D/U∞ of the lift oscillation"],
    ["von Kármán street", "the staggered double row of alternating vortices behind a bluff body"],
], I(0.6), I(1.3), I(12.1), [I(3.3), I(8.8)], size=11)

s = D.slide("", "File map")
D.table(s, [
    ["file", "role"],
    ["cylinder_re150/mesh_from_known_sizes.py", "mesher: INPUT → O–H multiblock quads → verify → extract_staggered → mesh_output/staggered_*.npz"],
    ["cylinder_re150/ns_staggered_euler.py", "solver: load_staggered, build_topology, GriebelStaggeredNS (step, _project_Un, …), plotting, save_wake_gif, CLI"],
    ["cylinder_re150/tools/compare_nasa.py", "post-processing: Cd, Cl, St against the NASA Study #1 reference (VALIDATION_NASA.md)"],
    ["cylinder_re150/THEORY_12_steps.md", "the method as twelve short steps (this deck in text form)"],
    ["cylinder_re150/THEORY_ns_staggered.md", "compact term-by-term derivation"],
    ["cylinder_re150/demo/", "cylinder_wake_griebel.gif, forces_griebel_fine.png, ns_fine_shedding_frame.png, these slides"],
    ["cylinder_re150/mesh_output/ (generated)", "staggered_{quick,fine,result}.npz, mesh pictures, vtk/msh"],
    ["cylinder_re150/output_ns/ (generated)", "frames, GIF, fields npz, forces and solution plots"],
], I(0.6), I(1.3), I(12.1), [I(4.0), I(8.1)], size=12)
D.callout(s, "Reference for the method: M. Griebel, T. Dornseifer, T. Neunhoeffer, Numerical Simulation in Fluid Dynamics: A Practical Introduction, SIAM 1998 — chapter 3 (the MAC/projection scheme, donor-cell convection, SOR) — and its C implementation at github.com/CaptainProton42/NavierStokes (integration.c, main.c).", I(0.6), I(4.6), I(12.1), I(1.2), size=12, fill=LIGHT)

out = D.save(OUT)
print("wrote", out, out.stat().st_size / 1e6, "MB", D.n, "slides")
