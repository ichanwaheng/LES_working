"""Build the '12 steps to the cylinder wake' deck with python-pptx.

Equations are rendered with matplotlib mathtext to PNG so they look right
in PowerPoint without an equation editor.
"""
from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Circle
import numpy as np

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "demo"
TMP = Path("/tmp/ppt_assets"); TMP.mkdir(exist_ok=True)
OUT = ROOT / "demo" / "ns_staggered_12_steps.pptx"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREY = RGBColor(0x55, 0x55, 0x55)
ACC = RGBColor(0xB2, 0x3A, 0x2A)
CODE_BG = RGBColor(0xF3, 0xF4, 0xF6)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


# ---------------------------------------------------------------- helpers
def eq_png(tex: str, fontsize: int = 22, color: str = "#1F3A5F") -> Path:
    h = hashlib.md5((tex + str(fontsize)).encode()).hexdigest()[:10]
    p = TMP / f"eq_{h}.png"
    if p.exists():
        return p
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=fontsize, color=color)
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    return p


def add_title(slide, step: str, title: str):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(1.0))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = step + "  "
    r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = ACC
    r = p.add_run(); r.text = title
    r.font.size = Pt(26); r.font.bold = True; r.font.color.rgb = NAVY
    ln = slide.shapes.add_shape(1, Inches(0.5), Inches(1.05), Inches(12.3), Emu(18000))
    ln.fill.solid(); ln.fill.fore_color.rgb = NAVY; ln.line.fill.background()


def add_bullets(slide, items, left, top, width, height, size=16, color=GREY):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    for it in items:
        lvl = 0
        if isinstance(it, tuple):
            lvl, it = it
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = lvl
        r = p.add_run(); r.text = ("• " if lvl == 0 else "– ") + it
        r.font.size = Pt(size - 2 * lvl); r.font.color.rgb = color
        p.space_after = Pt(6)
    return tb


def add_code(slide, code: str, left, top, width, height, size=11):
    box = slide.shapes.add_shape(1, left, top, width, height)
    box.fill.solid(); box.fill.fore_color.rgb = CODE_BG
    box.line.color.rgb = RGBColor(0xD0, 0xD0, 0xD0)
    tf = box.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = Inches(0.15)
    tf.margin_top = tf.margin_bottom = Inches(0.1)
    first = True
    for line in code.rstrip("\n").split("\n"):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run(); r.text = line if line else " "
        r.font.size = Pt(size); r.font.name = "Courier New"
        r.font.color.rgb = RGBColor(0x20, 0x20, 0x20)
    return box


def add_eq(slide, tex, left, top, width=None, height=None, fontsize=22, max_width=None):
    """Insert a rendered equation, scaled to fit both the given height and the slide width."""
    from PIL import Image
    p = eq_png(tex, fontsize)
    with Image.open(p) as im:
        w_px, h_px = im.size
    if max_width is None:
        max_width = prs.slide_width - left - Inches(0.4)
    if width is not None:
        max_width = min(max_width, width)
    if height is None:
        height = Inches(0.6)
    # scale by height, then shrink if too wide
    h = height
    w = int(h * w_px / h_px)
    if w > max_width:
        w = int(max_width)
        h = int(w * h_px / w_px)
    return slide.shapes.add_picture(str(p), left, top, width=w, height=h)


def add_pic(slide, path, left, top, width=None, height=None):
    kw = {}
    if width is not None: kw["width"] = width
    if height is not None: kw["height"] = height
    return slide.shapes.add_picture(str(path), left, top, **kw)


def add_note(slide, text, left, top, width, height, size=14, color=ACC, bold=True):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame; tf.word_wrap = True
    r = tf.paragraphs[0].add_run(); r.text = text
    r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold
    return tb


def footer(slide, i):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.3))
    r = tb.text_frame.paragraphs[0].add_run()
    r.text = f"ns_staggered_euler.py on the known-sizes O–H staggered mesh   ·   {i}"
    r.font.size = Pt(10); r.font.color.rgb = RGBColor(0x99, 0x99, 0x99)


# ---------------------------------------------------------------- sketches
def sketch_cell() -> Path:
    p = TMP / "sketch_cell.png"
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    P = np.array([[0, 0], [2.0, 0.2], [2.2, 1.6], [0.1, 1.4]])
    N = np.array([[2.0, 0.2], [3.9, -0.1], [4.2, 1.5], [2.2, 1.6]])
    ax.add_patch(Polygon(P, fc="#dbe7f5", ec="k", lw=1.5))
    ax.add_patch(Polygon(N, fc="#f7e3df", ec="k", lw=1.5))
    cP, cN = P.mean(0), N.mean(0)
    ax.plot(*cP, "o", color="#1F3A5F", ms=9); ax.text(cP[0] - 0.15, cP[1] + 0.2, "P", fontsize=16, fontweight="bold")
    ax.plot(*cN, "o", color="#B23A2A", ms=9); ax.text(cN[0] + 0.1, cN[1] + 0.2, "N", fontsize=16, fontweight="bold")
    f0, f1 = P[1], P[2]
    fc = 0.5 * (f0 + f1)
    t = f1 - f0; nrm = np.array([t[1], -t[0]]); nrm /= np.linalg.norm(nrm)
    ax.plot([f0[0], f1[0]], [f0[1], f1[1]], color="#2a7f3a", lw=4)
    ax.plot(*fc, "s", color="#2a7f3a", ms=9)
    ax.add_patch(FancyArrowPatch(fc, fc + 0.8 * nrm, arrowstyle="->", mutation_scale=18, lw=2, color="#2a7f3a"))
    ax.text(fc[0] + 0.85 * nrm[0], fc[1] + 0.85 * nrm[1] - 0.1, r"$\mathbf{n}_f$", fontsize=15, color="#2a7f3a")
    ax.text(fc[0] - 0.05, fc[1] - 0.45, r"face $f$: $U_{n,f}$, $L_f$", fontsize=13, color="#2a7f3a", ha="center")
    ax.plot([cP[0], cN[0]], [cP[1], cN[1]], "k--", lw=1)
    ax.text(0.5 * (cP[0] + cN[0]), 0.5 * (cP[1] + cN[1]) - 0.35, r"$d_{PN}$", fontsize=14, ha="center")
    ax.text(cP[0] - 0.1, cP[1] - 0.45, r"$p_P,\ u_P,\ v_P,\ V_P$", fontsize=12, ha="center", color="#1F3A5F")
    ax.set_xlim(-0.4, 4.6); ax.set_ylim(-0.9, 2.1); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(p, dpi=200, transparent=True); plt.close(fig)
    return p


def sketch_face_value() -> Path:
    p = TMP / "sketch_face.png"
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = [0, 1, 2]; phi = [1.0, 0.3, 0.7]
    ax.plot(x, phi, "o", color="#1F3A5F", ms=10)
    for xi, lab in zip(x, [r"$\phi_P$", r"$\phi_N$", ""]):
        ax.text(xi, 1.12, lab, ha="center", fontsize=15)
    ax.axvline(0.5, color="#2a7f3a", lw=3); ax.text(0.5, -0.02, "face f", ha="center", color="#2a7f3a", fontsize=13)
    ax.plot(0.5, 0.65, "D", color="#555", ms=9); ax.text(0.58, 0.68, r"central $\frac{1}{2}(\phi_P+\phi_N)$", fontsize=12)
    ax.plot(0.5, 1.0, "s", color="#B23A2A", ms=9); ax.text(0.58, 0.97, r"upwind $\phi_P$  ($U_n>0$)", fontsize=12, color="#B23A2A")
    ax.add_patch(FancyArrowPatch((0.05, 0.15), (0.45, 0.15), arrowstyle="->", mutation_scale=16, lw=2, color="#2a7f3a"))
    ax.text(0.25, 0.2, r"$U_{n,f}$", ha="center", color="#2a7f3a", fontsize=13)
    ax.set_xlim(-0.3, 1.5); ax.set_ylim(-0.1, 1.3); ax.axis("off")
    fig.tight_layout(); fig.savefig(p, dpi=200, transparent=True); plt.close(fig)
    return p


def sketch_projection() -> Path:
    p = TMP / "sketch_proj.png"
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    boxes = [("u^n, U_n^n", "#dbe7f5"), ("F, G\n(no pressure)", "#f7e3df"),
             ("solve\nA p = rhs", "#e6f1e0"), ("u^{n+1}, U_n^{n+1}\n∇·U = 0", "#dbe7f5")]
    labels = ["convect +\ndiffuse (Δt)", "div(U*)", "− Δt ∇p"]
    for k, (txt, col) in enumerate(boxes):
        ax.add_patch(plt.Rectangle((k * 1.6, 0), 1.2, 1, fc=col, ec="k"))
        ax.text(k * 1.6 + 0.6, 0.5, txt.replace("^n", "ⁿ").replace("^{n+1}", "ⁿ⁺¹"), ha="center", va="center", fontsize=12)
        if k < 3:
            ax.add_patch(FancyArrowPatch((k * 1.6 + 1.22, 0.5), (k * 1.6 + 1.58, 0.5), arrowstyle="->", mutation_scale=16, lw=2))
            ax.text(k * 1.6 + 1.4, 1.1, labels[k], ha="center", fontsize=10, color="#555")
    ax.set_xlim(-0.1, 6.1); ax.set_ylim(-0.2, 1.6); ax.axis("off")
    fig.tight_layout(); fig.savefig(p, dpi=200, transparent=True); plt.close(fig)
    return p


# ---------------------------------------------------------------- slides
n = 0

# 0 title
s = prs.slides.add_slide(BLANK); n += 1
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
tb = s.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.5), Inches(1.5))
r = tb.text_frame.paragraphs[0].add_run(); r.text = "12 steps to the cylinder wake"
r.font.size = Pt(48); r.font.bold = True; r.font.color.rgb = RGBColor(255, 255, 255)
tb = s.shapes.add_textbox(Inches(0.8), Inches(2.7), Inches(11.5), Inches(1.5))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = "How ns_staggered_euler.py solves flow past a cylinder at Re = 150 on the known-sizes O–H staggered mesh"
r.font.size = Pt(22); r.font.color.rgb = RGBColor(0xDD, 0xE6, 0xF0)
p2 = tf.add_paragraph(); r = p2.add_run()
r.text = "equation → discretise on our mesh → rearrange for the unknown → code → what we see"
r.font.size = Pt(18); r.font.italic = True; r.font.color.rgb = RGBColor(0xDD, 0xE6, 0xF0)
add_pic(s, ART / "ns_fine_shedding_frame.png", Inches(7.3), Inches(3.9), height=Inches(3.3))
tb = s.shapes.add_textbox(Inches(0.8), Inches(6.3), Inches(6), Inches(0.8))
tf = tb.text_frame; r = tf.paragraphs[0].add_run()
r.text = "Time stepping / projection after Griebel, Dornseifer & Neunhoeffer (1998)\nas in github.com/CaptainProton42/NavierStokes; finite-volume operators on the multiblock mesh"
r.font.size = Pt(12); r.font.color.rgb = RGBColor(0xBB, 0xC8, 0xD8)

# 1 problem & unknowns
s = prs.slides.add_slide(BLANK); n += 1
add_title(s, "STEP 1", "What we solve, and where the unknowns sit")
add_eq(s, r"\nabla\cdot\mathbf{u}=0,\qquad \frac{\partial\mathbf{u}}{\partial t}+\nabla\cdot(\mathbf{u}\otimes\mathbf{u})=-\nabla p+\nu\nabla^2\mathbf{u},\qquad \nu=\frac{1}{Re},\ Re=150", Inches(0.6), Inches(1.3), height=Inches(0.6), max_width=Inches(6.4))
add_bullets(s, [
    "Mesh from mesh_from_known_sizes.py: O-ring + transition + Cartesian blocks, 26,444 cells (d_min = 0.015)",
    "No i, j — a cell knows its neighbours only through its faces (owner / neighbour arrays)",
    "Staggered (MAC): p, u, v at cell centres P;  U_n = u·n on face centres f",
    "The pressure difference across a face and the velocity through it sit in the same place → no checkerboard",
], Inches(0.6), Inches(2.1), Inches(6.4), Inches(2.6), size=15)
add_code(s, """P   = S["cell_centers"]   # p, u, v live here
Fc  = S["face_centers"]   # Un lives here
n   = S["face_normals"]   # owner -> neighbour
L   = S["face_lengths"]
own = S["face_owner"]
nei = S["face_neigh"]     # -1 on a boundary""", Inches(0.6), Inches(4.7), Inches(6.4), Inches(2.1), size=12)
add_pic(s, ROOT / "mesh_output" / "staggered_fine.png", Inches(7.2), Inches(1.4), width=Inches(5.8))
add_note(s, "left: cell centres P (pressure)   right: face centres (velocity U_n) in the cylinder boundary layer", Inches(7.2), Inches(4.3), Inches(5.8), Inches(0.6), size=11, color=GREY, bold=False)
add_pic(s, sketch_cell(), Inches(7.6), Inches(4.8), height=Inches(2.1))
footer(s, n)

# 2 one tool
s = prs.slides.add_slide(BLANK); n += 1
add_title(s, "STEP 1b", "One tool for every term: cell integral → face sum")
add_eq(s, r"\frac{1}{V_P}\int_{V_P}\nabla\cdot\mathbf{F}\,dV\;=\;\frac{1}{V_P}\sum_{f\in P}s_{P,f}\,(\mathbf{F}\cdot\mathbf{n})_f\,L_f,\qquad s_{P,f}=+1\ (P\ \mathrm{owns}\ f),\ \ -1\ (P\ \mathrm{is\ the\ neighbour})", Inches(0.6), Inches(1.35), height=Inches(1.1))
add_code(s, """def accumulate(flux, own, nei, Nc):   # sum_f s_{P,f} * flux_f, for every cell at once
    out = np.zeros(Nc)
    np.add.at(out, own, flux)          # + for the owner cell
    m = nei >= 0
    np.add.at(out, nei[m], -flux[m])   # - for the neighbour cell
    return out""", Inches(0.6), Inches(2.7), Inches(7.6), Inches(1.9), size=13)
add_bullets(s, [
    "Convection, diffusion, divergence and pressure gradient are all this function with a different flux",
    "On a Cartesian cell the face sum collapses to the usual (i+1) − 2(i) + (i−1) stencils; here the faces may point anywhere",
    "Cell area V_P is itself a face sum:  V_P = ½ Σ (x_f · n_f) L_f",
    "d_PN = |x_N − x_P| plays the role of Δx between the two cells; L_f plays the role of Δy of the face",
], Inches(0.6), Inches(4.8), Inches(7.6), Inches(2.2), size=14)
add_pic(s, sketch_cell(), Inches(8.5), Inches(2.6), width=Inches(4.4))
footer(s, n)

# 3 time derivative
s = prs.slides.add_slide(BLANK); n += 1
add_title(s, "STEP 2", "The time derivative: forward (explicit) Euler")
add_eq(s, r"\frac{\partial\phi}{\partial t}\approx\frac{\phi_P^{\,n+1}-\phi_P^{\,n}}{\Delta t}\quad\Longrightarrow\quad \phi_P^{\,n+1}=\phi_P^{\,n}+\Delta t\cdot(\mathrm{RHS\ evaluated\ with\ level}\ n\ \mathrm{fields})", Inches(0.6), Inches(1.5), height=Inches(0.9))
add_bullets(s, [
    "Everything on the right-hand side is a known number → the new field is a one-line numpy expression, no equation to solve for the velocity",
    "Same update for every cell, for u and for v",
    "The price: Δt is limited by the smallest cell (Step 9) — the whole domain marches at the step dictated by the cylinder boundary layer",
    "First order in time; this is one reason Cd and St come out a few % above reference values",
], Inches(0.6), Inches(2.8), Inches(12), Inches(3.5), size=17)
footer(s, n)

# 4 convection
s = prs.slides.add_slide(BLANK); n += 1
add_title(s, "STEP 3", "The convective term — three ways to pick φ at the face")
add_eq(s, r"[\nabla\cdot(\mathbf{u}\,\phi)]_P=\frac{1}{V_P}\sum_{f\in P}s_{P,f}\;U_{n,f}\,L_f\;\phi_f\qquad(U_{n,f}L_f:\ \mathrm{flux\ we\ have};\ \ \phi_f:\ \mathrm{value\ at\ the\ face?})", Inches(0.6), Inches(1.3), height=Inches(1.0))
add_eq(s, r"\phi_f=\frac{1}{2}(\phi_P+\phi_N)\;-\;\frac{1}{2}\,\gamma\,\mathrm{sgn}(U_{n,f})\,(\phi_N-\phi_P)", Inches(0.6), Inches(2.45), height=Inches(0.55))
add_bullets(s, [
    "γ = 0  central: the average, 2nd order, no built-in damping  (--scheme central)",
    "γ = 1  upwind / donor cell: the upstream cell, 1st order, numerical viscosity ≈ ½|U_n| d_PN  (--scheme upwind)",
    "γ = CFL = max |U_n| Δt / h_min: Griebel's adaptive blend  (--scheme donor, ≈ 0.36 here)",
], Inches(0.6), Inches(3.15), Inches(7.4), Inches(1.6), size=14)
add_code(s, """phi_f[interior] = (1-gamma)*0.5*(phi[o]+phi[nn]) \\
                + gamma*np.where(Un[interior] >= 0, phi[o], phi[nn])
ru = accumulate(Un * uf * L, own, nei, Nc)     # for u
rv = accumulate(Un * vf * L, own, nei, Nc)     # for v
conv_u, conv_v = -ru / vol, -rv / vol""", Inches(0.6), Inches(4.8), Inches(7.4), Inches(1.5), size=12)
add_pic(s, sketch_face_value(), Inches(8.3), Inches(1.4), width=Inches(4.6))
add_note(s, "What we see: same 37k-cell mesh to tU/D = 150 — donor keeps Cl ≡ 0 (its numerical viscosity holds the symmetric wake), central sheds vortices. The scheme is not cosmetic.", Inches(8.3), Inches(4.2), Inches(4.6), Inches(1.6), size=13)
add_note(s, "Boundary faces: φ_f = (U∞, 0) at the inlet, 0 on walls and cylinder, owner value at the outlet", Inches(0.6), Inches(6.4), Inches(12), Inches(0.5), size=12, color=GREY, bold=False)
footer(s, n)

# 5 viscous
s = prs.slides.add_slide(BLANK); n += 1
add_title(s, "STEP 4", "The viscous term — central gradient across each face")
add_eq(s, r"[\nu\nabla^2\phi]_P=\frac{\nu}{V_P}\sum_{f\in P}s_{P,f}(\frac{\partial\phi}{\partial n})_f L_f,\qquad (\frac{\partial\phi}{\partial n})_f\approx\frac{\phi_N-\phi_P}{d_{PN}}\quad(\mathrm{boundary:}\ \frac{\phi_{bc}-\phi_P}{d/2})", Inches(0.6), Inches(1.35), height=Inches(1.0))
add_code(s, """g = np.empty(Nf)
g[interior] = (phi[nei[interior]] - phi[own[interior]]) / d_pn[interior]
g[bnd]      = (phi_bc[bnd] - phi[own[bnd]]) / (0.5 * d_pn[bnd])
lap = accumulate(nu * g * L, own, nei, Nc) / vol""", Inches(0.6), Inches(2.6), Inches(7.6), Inches(1.5), size=13)
add_bullets(s, [
    "On a Cartesian cell this is exactly ν(φ_{i+1} − 2φ_i + φ_{i−1})/Δx² — written as a flux balance so skewed ring cells work too",
    "No non-orthogonal correction: first-order on strongly skewed cells, one reason the Δt safety factor τ is 0.4 rather than 0.5",
    "u_bc = U∞ at the inlet, 0 on walls / cylinder;  v_bc = 0 on all of them;  outlet: zero gradient",
], Inches(0.6), Inches(4.3), Inches(12), Inches(2.2), size=15)
footer(s, n)

# 6 predictor
s = prs.slides.add_slide(BLANK); n += 1
add_title(s, "STEP 5", "The predictor F, G — momentum without pressure")
add_eq(s, r"F_P=u_P^{\,n}+\Delta t[\nu\nabla^2u-\nabla\cdot(\mathbf{u}u)]_P^{\,n},\qquad G_P=v_P^{\,n}+\Delta t[\nu\nabla^2v-\nabla\cdot(\mathbf{u}v)]_P^{\,n}", Inches(0.6), Inches(1.35), height=Inches(0.75))
add_code(s, """cu, cv = _convective(u, v, Un, gamma)        # Step 3
F = u + dt * (_laplacian(u, u_bc) + cu)      # Step 2 + 4
G = v + dt * (_laplacian(v, v_bc) + cv)

uf, vf  = face_values(F, G, gamma=0.0)       # average to faces
Un_star = uf * n[:, 0] + vf * n[:, 1]        # provisional flux""", Inches(0.6), Inches(2.3), Inches(7.2), Inches(2.1), size=13)
add_eq(s, r"U^{*}_{n,f}=F_f\,n_{x,f}+G_f\,n_{y,f},\qquad F_f=\frac{1}{2}(F_P+F_N)", Inches(0.6), Inches(4.6), Inches(7.2), None, fontsize=20)
add_bullets(s, [
    "F, G carry the right momentum but nobody has enforced continuity: ∇·(F,G) ≠ 0",
    "The pressure is left out on purpose — its only job (Steps 6–8) is to remove that divergence",
    "Because F, G contain no old pressure, the p we solve for next is the pressure itself, not an increment",
], Inches(0.6), Inches(5.3), Inches(7.4), Inches(1.7), size=14)
add_pic(s, sketch_projection(), Inches(8.0), Inches(2.3), width=Inches(5.0))
footer(s, n)

# 7 divergence -> Poisson
s = prs.slides.add_slide(BLANK); n += 1
add_title(s, "STEP 6", "Divergence of a cell ⇒ the equation for the pressure")
add_eq(s, r"[\nabla\cdot\mathbf{u}]_P=\frac{1}{V_P}\sum_{f\in P}s_{P,f}\,U_{n,f}\,L_f", Inches(0.6), Inches(1.3), height=Inches(0.75))
add_note(s, "We want the end-of-step flux to be corrected by the pressure difference across the face  AND  zero divergence:", Inches(0.6), Inches(2.15), Inches(12), Inches(0.4), size=14, color=GREY, bold=False)
add_eq(s, r"U^{\,n+1}_{n,f}=U^{*}_{n,f}-\Delta t\,\frac{p_N-p_P}{d_{PN}},\qquad \frac{1}{V_P}\sum_{f}s_{P,f}\,U^{\,n+1}_{n,f}\,L_f=0", Inches(0.6), Inches(2.6), height=Inches(0.8))
add_note(s, "Put the first into the second — one linear equation per cell, unknowns p in the cell and its neighbours:", Inches(0.6), Inches(3.5), Inches(12), Inches(0.4), size=14, color=GREY, bold=False)
add_eq(s, r"\sum_{f\in P}s_{P,f}\,\frac{L_f}{d_{PN}}\,(p_N-p_P)\;=\;\frac{V_P}{\Delta t}[\nabla\cdot\mathbf{u}^{*}]_P", Inches(0.6), Inches(3.95), height=Inches(0.9), fontsize=24)
add_code(s, """o, c = own[interior], nei[interior];  g = L[interior] / d_pn[interior]
rows = r_[o, o, c, c]; cols = r_[c, o, o, c]; data = r_[g, -g, g, -g]
A = sparse.coo_matrix((data, (rows, cols)), shape=(Nc, Nc)).tocsr()   # built once""", Inches(0.6), Inches(5.0), Inches(7.8), Inches(1.1), size=12)
add_bullets(s, [
    "A is symmetric and never changes (the mesh does not move)",
    "All boundaries ∂p/∂n = 0 → rows sum to zero: p defined up to a constant, and solvable only if rhs.sum() == 0 (inflow = outflow)",
], Inches(8.6), Inches(4.9), Inches(4.4), Inches(2.0), size=13)
footer(s, n)

# 8 BCs & mass balance
s = prs.slides.add_slide(BLANK); n += 1
add_title(s, "STEP 7", "Boundary conditions on the flux, and the mass balance")
add_code(s, """def _enforce_Un_bc(Un):
    Un[inlet] = U_inf * n[inlet, 0] + v_in(t) * n[inlet, 1]   # prescribed
    Un[wall_top] = Un[wall_bottom] = Un[cylinder] = 0.0        # no-slip
    return Un                       # outlet: left as is (zero gradient)

def _balance_outlet(Un):            # make rhs.sum() == 0
    Q_other = np.sum(Un[~outlet & bnd] * L[~outlet & bnd])
    Q_out   = np.sum(Un[outlet] * L[outlet])
    Un[outlet] += -(Q_other + Q_out) / np.sum(L[outlet])
    return Un""", Inches(0.6), Inches(1.35), Inches(7.6), Inches(3.3), size=12)
add_bullets(s, [
    "inlet x = 0: u = U∞ (+ a short transverse pulse, Step 10);  outlet x = 24 D: ∂u/∂n = 0;  walls y = 0, 12 D and cylinder (centre 6 D, 6 D): u = 0",
    "A zero-gradient outlet does not automatically let out what came in",
], Inches(0.6), Inches(4.9), Inches(7.6), Inches(1.6), size=14)
add_note(s, "What we saw without _balance_outlet: the solver ran, but max|div| sat at 1e-2, largest in the tiny cells on the cylinder — the unbalanced mass was spread over the domain as a constant divergence.\n\nWith it: max|div| = 1e-12.", Inches(8.5), Inches(1.5), Inches(4.4), Inches(3.0), size=14)
add_code(s, """step 10220  t=48.09  max|div|=1.571e-02   # before
step 31819  t=150.0  max|div|=7.857e-13   # after""", Inches(8.5), Inches(4.6), Inches(4.4), Inches(0.9), size=11)
footer(s, n)

# 9 solve & correct
s = prs.slides.add_slide(BLANK); n += 1
add_title(s, "STEP 8", "Solve for p, then correct — faces first, cells second")
add_code(s, """rhs = vol / dt * _div(Un_star);  rhs -= rhs.mean()   # compatibility
p = solve(A, rhs);               p -= p.mean()       # free constant

Un = Un_star.copy()                                  # faces first
Un[interior] -= dt * (p[nei[interior]] - p[own[interior]]) / d_pn[interior]
Un = _enforce_Un_bc(Un)                              # _div(Un) ~ 1e-12

pf = 0.5 * (p[own] + p[nei]);  pf[bnd] = p[own[bnd]] # Green-Gauss
gx = accumulate(pf * n[:, 0] * L, own, nei, Nc) / vol
gy = accumulate(pf * n[:, 1] * L, own, nei, Nc) / vol
u = F - dt * gx;   v = G - dt * gy                   # cells second""", Inches(0.6), Inches(1.35), Inches(7.6), Inches(3.5), size=11)
add_eq(s, r"U^{\,n+1}_{n,f}=U^{*}_{n,f}-\Delta t\,\frac{p_N-p_P}{d_{PN}},\qquad u_P^{\,n+1}=F_P-\Delta t\,[\partial_x p]_P,\quad v_P^{\,n+1}=G_P-\Delta t\,[\partial_y p]_P", Inches(0.6), Inches(5.1), height=Inches(0.7), fontsize=20, max_width=Inches(7.6))
tb = add_bullets(s, [
    "--poisson sor: Gauss–Seidel sweeps, ω = 1.7 (Griebel's original) — thousands of sweeps",
    "--poisson cg: Jacobi-PCG on the mean-free operator — ≈1000 iterations, 390 ms/step",
    "--poisson lu (default): pin one cell, splu(A) once, two triangular solves per step — 2 ms",
    "Same (p_N − p_P)/d_PN that built A is used to correct the faces → divergence is zero to solver precision",
], Inches(8.5), Inches(1.4), Inches(4.5), Inches(4.0), size=13)
add_note(s, "max|div| = 7.9e-13 in the log, every 20 steps", Inches(8.5), Inches(5.6), Inches(4.5), Inches(0.5), size=13)
footer(s, n)

# 10 dt
s = prs.slides.add_slide(BLANK); n += 1
add_title(s, "STEP 9", "Choosing Δt — the smallest cell decides")
add_eq(s, r"\Delta t=\tau\,\min\left(\frac{h_{\min}}{|u|_{\max}},\;\frac{Re\,h_{\min}^2}{4}\right),\qquad h_{\min}=\min_P\sqrt{V_P},\qquad \tau=0.4", Inches(0.6), Inches(1.35), height=Inches(0.9))
add_code(s, """def stable_dt():
    umax = max(np.abs(Un).max(), U_inf)
    return tau * min(hmin / umax, Re * hmin**2 / 4)""", Inches(0.6), Inches(2.5), Inches(6.5), Inches(1.1), size=13)
add_bullets(s, [
    "h_min / |u|_max — CFL: fluid may not cross more than one cell per step",
    "Re h_min² / 4 — diffusive limit of explicit Euler: halving d_min quarters Δt",
    "τ is the safety factor (Griebel use 0.5; 0.4 here for the skewed ring cells)",
    "Fine mesh: h_min = 0.0177 (first ring on the cylinder), |u|_max ≈ 1.58 while shedding → Δt ≈ 0.0045",
    "≈ 1,100 steps per shedding period, 26,600 steps to tU/D = 120 — about 12 ms each with the LU solver",
    "Adaptive in name only once the start-up transient is over: h_min is fixed and |u|_max barely moves",
    "Far-field cells are 0.12 wide and could take a 7× larger step; one global explicit Δt does not allow it",
], Inches(0.6), Inches(3.8), Inches(12), Inches(3.2), size=14)
footer(s, n)

# 11 IC & symmetry
s = prs.slides.add_slide(BLANK); n += 1
add_title(s, "STEP 10", "Initial condition — why a symmetric start never sheds")
add_bullets(s, [
    "u = U∞, v = 0 everywhere (impulsive start), projected once with Steps 6–8 so step 1 begins divergence-free",
    "The O–H mesh is exactly mirror-symmetric about y = 0, and so is every operator above → a symmetric start stays symmetric",
    "Cl printed as −0.000 for 120 time units while the wake grew into the long, steady, unstable symmetric bubble",
    "Only at tU/D ≈ 120 had round-off grown enough (×10¹³) to start the oscillation — the left plot",
    "A small seed one diameter behind the cylinder brings the onset to tU/D ≈ 20 — the right plot; both end on the same limit cycle (earlier 40 D × 10 D domain; Cl ± 0.75)",
], Inches(0.6), Inches(1.3), Inches(12), Inches(2.2), size=14)
add_code(s, """r2 = (P[:, 0] - 1.0)**2 + P[:, 1]**2
v += perturb * U_inf * np.exp(-r2 / 0.5**2)    # perturb = 0.1  (--perturb)""", Inches(0.6), Inches(3.45), Inches(7.0), Inches(0.75), size=12)
add_pic(s, ART / "forces_noseed_40x10.png", Inches(0.6), Inches(4.3), height=Inches(2.65))
add_pic(s, ART / "forces_seed_40x10.png", Inches(6.9), Inches(4.3), height=Inches(2.65))
footer(s, n)

# 12 loop + results
s = prs.slides.add_slide(BLANK); n += 1
add_title(s, "STEP 11", "The time loop — and what comes out")
add_code(s, """while t < t_end:
    dt    = stable_dt()                                   # Step 9
    gamma = scheme_gamma(dt)                              # Step 3
    Un    = _enforce_Un_bc(Un)                            # Step 7
    cu, cv = _convective(u, v, Un, gamma)                 # Step 3
    F = u + dt * (_laplacian(u, u_bc) + cu)               # Steps 4-5
    G = v + dt * (_laplacian(v, v_bc) + cv)
    Un_star = face_normal(F, G)                           # Step 5
    Un_star = _balance_outlet(_enforce_Un_bc(Un_star))    # Step 7
    rhs = vol / dt * _div(Un_star); rhs -= rhs.mean()     # Step 6
    p = solve(A, rhs); p -= p.mean()                      # Step 8
    Un = Un_star - dt * grad_face(p); Un = _enforce_Un_bc(Un)
    u, v = F - dt * gx(p), G - dt * gy(p)
    t += dt
    Cd, Cl = forces_on_cylinder(p, u, v)                  # diagnostics
    if step % 100 == 0: save_frame(u, v, t)               # Step 12""", Inches(0.6), Inches(1.35), Inches(7.2), Inches(5.2), size=11)
add_bullets(s, [
    "Forces: Σ_cyl p n L (pressure) + ν u_t/(d/2) (wall shear), over ½ρU∞²D",
    "Fine mesh, central: after the impulsive transient Cl grows exponentially and saturates at ± 0.71 by tU/D ≈ 30",
    "Period 1/St ≈ 4.8 → St ≈ 0.21;  Cd ≈ 1.56 oscillating at twice the lift frequency",
    "Unbounded-cylinder references: St ≈ 0.18, Cd ≈ 1.33 — the 8.3 % blockage of the 12 D-high channel and first-order Euler push both up",
    "Cost: 12 ms/step, 37k cells, 26,600 steps → ≈ 6 min to tU/D = 120",
], Inches(8.0), Inches(1.4), Inches(5.0), Inches(4.5), size=13)
add_code(s, """forces   : Cd_mean=1.564  Cl_amp=0.712  St=0.207""", Inches(8.0), Inches(5.9), Inches(5.0), Inches(0.5), size=11)
footer(s, n)

# 13 pictures
s = prs.slides.add_slide(BLANK); n += 1
add_title(s, "STEP 12", "The pictures — snapshots, one instant per frame")
add_code(s, """coll = PolyCollection(S["quads"], array=field, cmap="RdBu_r")  # one colour per cell
ax.add_collection(coll)
idx = np.arange(0, Nc, Nc // 400)                              # ~400 sampled cells
ax.quiver(P[idx, 0], P[idx, 1], u[idx], v[idx])                 # velocity vector there""", Inches(0.6), Inches(1.35), Inches(6.6), Inches(1.4), size=12)
add_bullets(s, [
    "Top: vorticity ω_z = ∂v/∂x − ∂u/∂y from the Green–Gauss gradient of Step 8 — red / blue = the two senses of rotation",
    "Bottom, 'Velocity plot': the speed |u| — free stream ≈ 1, wake (slow) dark, shoulders (fast) bright; arrows = (u_P, v_P) in the sampled cells",
    "One frame every 100 steps (≈ 0.45 time units, ~11 per shedding period); GIF loops the last ~4 periods, MP4 is the whole history",
    "Nothing is traced between frames: the next frame is simply the next saved state, coloured and arrowed again",
    "No streamlines. An arrow is the local velocity vector at that instant, not a path",
], Inches(0.6), Inches(2.95), Inches(6.6), Inches(3.8), size=14)
add_pic(s, ART / "ns_fine_shedding_frame.png", Inches(7.4), Inches(1.35), height=Inches(5.5))
footer(s, n)

# 14 summary
s = prs.slides.add_slide(BLANK); n += 1
add_title(s, "SUMMARY", "The twelve steps on one slide")
rows = [
    ("1", "Mesh arrays, staggered unknowns, accumulate()", "p, u, v at P; U_n on faces"),
    ("2", "Explicit Euler", "φⁿ⁺¹ = φⁿ + Δt·RHS(n)"),
    ("3", "Convective flux, γ-blend", "central / upwind / donor"),
    ("4", "Viscous flux", "ν (φ_N − φ_P)/d_PN · L_f"),
    ("5", "Predictor F, G", "momentum without pressure"),
    ("6", "Divergence ⇒ Poisson", "Σ L/d (p_N − p_P) = V/Δt ∇·u*"),
    ("7", "BCs + outlet mass balance", "rhs.sum() == 0 → div 1e-12"),
    ("8", "Solve (LU) and correct", "faces first, then cells"),
    ("9", "Δt = τ·min(h/u, Re h²/4)", "0.0045 on the fine mesh"),
    ("10", "Symmetric start + seed", "shedding by tU/D ≈ 20–30"),
    ("11", "Loop, Cd / Cl", "Cl ±0.71, St 0.21, Cd 1.56"),
    ("12", "Snapshots", "cell colour + arrows, no streamlines"),
]
tbl = s.shapes.add_table(len(rows) + 1, 3, Inches(0.6), Inches(1.3), Inches(12.1), Inches(5.4)).table
tbl.columns[0].width = Inches(0.8); tbl.columns[1].width = Inches(5.6); tbl.columns[2].width = Inches(5.7)
for j, h in enumerate(["step", "what", "in our solver"]):
    c = tbl.cell(0, j); c.text = h
    c.text_frame.paragraphs[0].runs[0].font.bold = True
    c.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
for i, row in enumerate(rows, start=1):
    for j, val in enumerate(row):
        c = tbl.cell(i, j); c.text = val
        c.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
footer(s, n)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("wrote", OUT, OUT.stat().st_size / 1e6, "MB", n, "slides")
