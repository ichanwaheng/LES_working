#!/usr/bin/env python3
"""
mesh_from_known_sizes.py
========================
2D structured multiblock mesh of a cylinder in a rectangular fluid domain,
plus staggered (MAC-style) locations for finite-volume / collocated-face
calculations.

KNOWN (the inputs)
    1. the fluid domain and the position of the cylinder
    2. d_min : smallest cell size, next to every wall (cylinder included)
    3. d_max : largest cell size, kept everywhere outside the boundary layer

AND THE GROWTH RATIO
    growth_ratio : the ratio r by which cells grow inside the boundary
                   layer (used exactly; 1.1 to 1.2 is the usual range)

EVERYTHING ELSE IS DERIVED.

Staggered outputs (saved under mesh_output/)
    cell_centers   — pressure / scalar locations (one per quad)
    face_centers   — velocity locations (one per unique face)
    face_normals   — unit normals (owner → neighbour; outward on boundary)
    face_lengths   — face length (2-D "area")
    face_owner / face_neigh — cell connectivity (−1 = boundary)
    face_kind      — 0 = u-like (|n_x|≥|n_y|), 1 = v-like
    face_bc        — boundary name index (−1 interior); see bc_names

    python mesh_from_known_sizes.py

Needs numpy and matplotlib.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# =====================================================================
# THE INPUTS  (edit these)
# =====================================================================
INPUT = dict(
    # 1. fluid domain and cylinder position
    x_min=-10.0,
    x_max=30.0,
    y_min=-5.0,
    y_max=5.0,
    xc=0.0,
    yc=0.0,
    D=1.0,  # cylinder centre and diameter
    walls=("top", "bottom"),  # which domain sides are walls
    # 2. smallest cell (wall-normal size of the first cell at every wall)
    d_min=0.006,
    # 3. largest cell (outside the boundary layer)
    d_max=0.06,
    # the growth ratio, used exactly
    growth_ratio=1.15,
    # the curved block around the cylinder: None = find the smallest square
    # whose size jump (see below) stays within max_jump
    a=None,
    # how many transition cells per radial line (N_u):
    #   "middle": N_u = ceil(L_middle / d_max)
    #   "corner": N_u = ceil(L_corner / ((1 + corner_margin) * d_max))
    transition_count="middle",
    corner_margin=0.0,  # used only by "corner"
    max_jump=1.8,  # allowed size jump at O-interfaces
)


# =====================================================================
# STEP 1: boundary layer from d_min, d_max, growth
# =====================================================================
def boundary_layer(d_min, d_max, r):
    """Cells start at d_min and grow by exactly r. Layer stops at last cell
    still <= d_max. Beyond the layer: uniform ~ d_max."""
    if d_min >= d_max:
        raise ValueError("d_min must be smaller than d_max")
    if r <= 1.0:
        raise ValueError("growth_ratio must be larger than 1")
    n = int(np.floor(np.log(d_max / d_min) / np.log(r) + 1e-9)) + 1
    k = np.arange(n + 1)
    t = d_min * (r**k - 1.0) / (r - 1.0)
    return dict(n=n, r=r, t=t, T=t[-1], d_last=d_min * r ** (n - 1))


# =====================================================================
# STEP 2: grid lines of the rectangular blocks
# =====================================================================
def line_1d(p0, p1, bl0, bl1, bl, d_max):
    """Grid-line positions from p0 to p1 with optional BL at either end."""
    t, T = bl["t"], bl["T"]
    a0 = p0 + (T if bl0 else 0.0)
    a1 = p1 - (T if bl1 else 0.0)
    L = a1 - a0
    if L <= 1e-9:
        raise ValueError("no room for the uniform region: reduce a or d_max")
    nu = max(1, int(np.ceil(L / d_max - 1e-9)))
    mid = a0 + L * np.arange(nu + 1) / nu
    mid[0], mid[-1] = a0, a1
    parts = []
    if bl0:
        parts.append(p0 + t[:-1])
    parts.append(mid)
    if bl1:
        parts.append(p1 - t[::-1][1:])
    return np.concatenate(parts)


def build_grid_lines(I, bl):
    a, d_max, w = I["a"], I["d_max"], I["walls"]
    n_half = int(np.ceil(a / d_max - 1e-9))
    edge = a / n_half
    k = np.arange(-n_half, n_half + 1) / n_half
    mx, my = I["xc"] + a * k, I["yc"] + a * k

    xl = line_1d(I["x_min"], mx[0], "left" in w, False, bl, d_max)
    xr = line_1d(mx[-1], I["x_max"], False, "right" in w, bl, d_max)
    yb = line_1d(I["y_min"], my[0], "bottom" in w, False, bl, d_max)
    yt = line_1d(my[-1], I["y_max"], False, "top" in w, bl, d_max)
    xs = np.concatenate([xl[:-1], mx[:-1], xr])
    ys = np.concatenate([yb[:-1], my[:-1], yt])
    return dict(xs=xs, ys=ys, mx=mx, my=my, n_half=n_half, edge=edge)


# =====================================================================
# STEP 3: curved block (O-block) around the cylinder
# =====================================================================
def square_point(idx, n, mx, my):
    """Point idx (0..8n-1) on the square, CCW from mid-right."""
    if idx <= n:
        return (mx[-1], my[n + idx])
    if idx <= 3 * n:
        return (mx[3 * n - idx], my[-1])
    if idx <= 5 * n:
        return (mx[0], my[5 * n - idx])
    if idx <= 7 * n:
        return (mx[idx - 5 * n], my[0])
    return (mx[-1], my[idx - 7 * n])


def n_transition(I, L):
    d_max = I["d_max"]
    if I["transition_count"] == "middle":
        N = np.ceil(L.min() / d_max - 1e-9)
    else:
        N = np.ceil(L.max() / ((1 + I["corner_margin"]) * d_max) - 1e-9)
    return max(1, int(N))


def jump_for_a(I, bl, a):
    R, d_min, d_max = I["D"] / 2, I["d_min"], I["d_max"]
    nh = int(np.ceil(a / d_max - 1e-9))
    edge = a / nh
    Ro = R + bl["T"]
    th = (np.pi / 4) * np.arange(nh + 1) / nh
    C = Ro * np.stack([np.cos(th), np.sin(th)], axis=1)
    S = np.stack([np.full(nh + 1, a), a * np.arange(nh + 1) / nh], axis=1)
    L = np.linalg.norm(S - C, axis=1)
    if L.min() <= 0:
        return np.inf
    N_u = n_transition(I, L)
    u = L / N_u
    if u.min() < d_min:
        return np.inf
    j1 = np.maximum(u / bl["d_last"], bl["d_last"] / u)
    j2 = np.maximum(u / edge, edge / u)
    return float(max(j1.max(), j2.max()))


def choose_a(I, bl):
    R, T, d_max, w = I["D"] / 2, bl["T"], I["d_max"], I["walls"]
    room = (
        min(
            I["xc"] - I["x_min"] - (T if "left" in w else 0),
            I["x_max"] - I["xc"] - (T if "right" in w else 0),
            I["yc"] - I["y_min"] - (T if "bottom" in w else 0),
            I["y_max"] - I["yc"] - (T if "top" in w else 0),
        )
        - 2 * d_max
    )
    a = R + T + 2 * d_max
    best = (np.inf, a)
    while a <= room:
        j = jump_for_a(I, bl, a)
        if j < best[0]:
            best = (j, a)
        if j <= I["max_jump"]:
            return a
        a += 0.05
    print(
        f"warning: no square up to {room:.2f} reaches max_jump; "
        f"using a = {best[1]:.2f} (jump {best[0]:.2f})"
    )
    return best[1]


def region_report(I, bl, g):
    R, T, a = I["D"] / 2, bl["T"], I["a"]
    w = I["walls"]
    print("STEP 1: SIZE OF EACH REGION")
    print("-" * 78)
    print(
        f"boundary layer : {bl['n']} cells, ratio {bl['r']} (input), "
        f"last cell {bl['d_last']:.5f} (d_max {I['d_max']}), thickness T = {T:.4f}"
    )
    print(f"cylinder       : wall radius {R:.4f}; O-grid ring ends at {R + T:.4f}")
    print(f"curved block   : square half-side a = {a:.4f}, {g['n_half'] * 8} cells around")
    Nu = n_transition(I, np.array([a - R - T, np.sqrt(2) * a - R - T]))
    print(
        f"                 transition cells: {Nu} per line; size "
        f"{(a - R - T) / Nu:.4f} (side middle) to "
        f"{(np.sqrt(2) * a - R - T) / Nu:.4f} (corner)"
    )
    for s, lo, hi, bl_flag in (
        ("left", I["x_min"], I["xc"] - a, "left" in w),
        ("right", I["xc"] + a, I["x_max"], "right" in w),
        ("bottom", I["y_min"], I["yc"] - a, "bottom" in w),
        ("top", I["yc"] + a, I["y_max"], "top" in w),
    ):
        L = hi - lo
        txt = (
            f"{L:.4f} = boundary layer {T:.4f} + uniform {L - T:.4f}"
            if bl_flag
            else f"{L:.4f} (all uniform)"
        )
        print(f"{s:<7} block  : {txt}")
    print("-" * 78)


def build_o_block(I, bl, g):
    R, d_min, n = I["D"] / 2, I["d_min"], bl["n"]
    nh = g["n_half"]
    NT = 8 * nh
    centre = np.array([I["xc"], I["yc"]])
    th = 2 * np.pi * np.arange(NT) / NT
    ring = np.stack([np.cos(th), np.sin(th)], axis=1)
    S = np.array([square_point(i, nh, g["mx"], g["my"]) for i in range(NT)])
    C_out = centre + (R + bl["T"]) * ring
    L = np.linalg.norm(S - C_out, axis=1)
    if L.min() <= 0:
        raise ValueError("square around the cylinder is too small: increase a")
    N_u = n_transition(I, L)
    u = L / N_u
    if u.min() < d_min:
        raise ValueError("square around the cylinder is too small: increase a")
    NR = n + N_u

    nodes = np.zeros((NT + 1, NR + 1, 2))
    nodes[:NT, : n + 1] = centre + (R + bl["t"])[None, :, None] * ring[:, None, :]
    frac = np.arange(1, N_u + 1) / N_u
    nodes[:NT, n + 1 :] = C_out[:, None, :] + frac[None, :, None] * (S - C_out)[:, None, :]
    nodes[:NT, -1] = S
    nodes[NT] = nodes[0]

    A, B = nodes[:-1, :-1], nodes[:-1, 1:]
    Cc, Dd = nodes[1:, 1:], nodes[1:, :-1]
    quads = np.stack([A, B, Cc, Dd], axis=2).reshape(-1, 4, 2)
    zone = np.tile(np.where(np.arange(NR) < n, 0, 3), NT)
    return dict(nodes=nodes, quads=quads, zone=zone, NT=NT, NR=NR, N_u=N_u, u=u)


# =====================================================================
# STEP 4: rectangular blocks
# =====================================================================
def build_rect_blocks(I, bl, g):
    a, xs, ys = I["a"], g["xs"], g["ys"]
    X0, Y0 = np.meshgrid(xs[:-1], ys[:-1], indexing="ij")
    X1, Y1 = np.meshgrid(xs[1:], ys[1:], indexing="ij")
    cx, cy = (X0 + X1) / 2, (Y0 + Y1) / 2
    keep = ~((np.abs(cx - I["xc"]) < a) & (np.abs(cy - I["yc"]) < a))
    X0, X1, Y0, Y1, cx, cy = (v[keep] for v in (X0, X1, Y0, Y1, cx, cy))
    quads = np.stack(
        [
            np.stack([X0, Y0], -1),
            np.stack([X1, Y0], -1),
            np.stack([X1, Y1], -1),
            np.stack([X0, Y1], -1),
        ],
        axis=1,
    )
    dist = np.full(len(cx), np.inf)
    side = {
        "left": cx - I["x_min"],
        "right": I["x_max"] - cx,
        "bottom": cy - I["y_min"],
        "top": I["y_max"] - cy,
    }
    for s in I["walls"]:
        dist = np.minimum(dist, side[s])
    zone = np.where(dist < bl["T"], 1, 2)
    return quads, zone


# =====================================================================
# STEP 5: assemble
# =====================================================================
ZONES = {
    0: "cylinder boundary-layer ring",
    1: "wall boundary layer",
    2: "uniform region (rectangular blocks)",
    3: "transition cells (curved block)",
}


def assemble(I, ob, rq, rzone):
    quads = np.concatenate([ob["quads"], rq])
    zone = np.concatenate([ob["zone"], rzone])
    uniq, inv = np.unique(quads.reshape(-1, 2), axis=0, return_inverse=True)
    conn = inv.reshape(-1, 4)

    e = np.concatenate([conn[:, [k, (k + 1) % 4]] for k in range(4)])
    e.sort(axis=1)
    nn = len(uniq)
    code = e[:, 0].astype(np.int64) * nn + e[:, 1]
    u, cnt = np.unique(code, return_counts=True)
    bnd = u[cnt == 1]
    ba, bb = bnd // nn, bnd % nn

    tol, R = 1e-9, I["D"] / 2
    xa, ya, xb, yb = uniq[ba, 0], uniq[ba, 1], uniq[bb, 0], uniq[bb, 1]
    names = np.full(len(bnd), "", dtype=object)
    sides = {
        "left": ("inlet", xa, xb, I["x_min"]),
        "right": ("outlet", xa, xb, I["x_max"]),
        "bottom": ("bottom", ya, yb, I["y_min"]),
        "top": ("top", ya, yb, I["y_max"]),
    }
    for s, (nm, p, q, val) in sides.items():
        if s in I["walls"]:
            nm = "wall_" + s
        names[(np.abs(p - val) < tol) & (np.abs(q - val) < tol)] = nm
    ra = np.hypot(xa - I["xc"], ya - I["yc"])
    rb = np.hypot(xb - I["xc"], yb - I["yc"])
    names[(np.abs(ra - R) < tol) & (np.abs(rb - R) < tol)] = "cylinder"
    return dict(
        nodes=uniq,
        conn=conn,
        quads=quads,
        zone=zone,
        bnd=np.stack([ba, bb], 1),
        bnd_names=names,
    )


# =====================================================================
# STEP 5b: staggered (MAC-style) locations
# =====================================================================
def extract_staggered(mesh):
    """
    Build staggered locations from the primal quad mesh.

    Pressure / scalars live at cell centres.
    Velocities live at face centres (unique faces shared by ≤2 cells).

    Face kind (for a Cartesian-like split of the flux):
        0 = u-like  (|n_x| ≥ |n_y|)  — primarily vertical face
        1 = v-like  (|n_y| >  |n_x|) — primarily horizontal face

    On the O-block the same rule still assigns each face to a dominant
    Cartesian direction; the stored normal is the true geometric normal.
    """
    nodes = mesh["nodes"]
    conn = mesh["conn"]  # (Nc, 4) CCW
    Nc = len(conn)

    # --- cell centres (P) ---
    cell_centers = nodes[conn].mean(axis=1)  # (Nc, 2)

    # --- gather all half-faces, then unique ---
    # edge key: sorted node pair
    n_node = len(nodes)
    edges_local = []  # (c, local_face, a, b) with a < b
    for c in range(Nc):
        for k in range(4):
            a = int(conn[c, k])
            b = int(conn[c, (k + 1) % 4])
            if a > b:
                a, b = b, a
            edges_local.append((c, k, a, b))
    edges_local = np.asarray(edges_local, dtype=np.int64)
    codes = edges_local[:, 2] * n_node + edges_local[:, 3]

    order = np.argsort(codes)
    codes_s = codes[order]
    edges_s = edges_local[order]

    # group identical codes
    face_owner = []
    face_neigh = []
    face_nodes = []  # (a, b) with orientation from owner (CCW edge)
    face_owner_local = []

    i = 0
    Nhalf = len(codes_s)
    while i < Nhalf:
        j = i + 1
        while j < Nhalf and codes_s[j] == codes_s[i]:
            j += 1
        group = edges_s[i:j]
        # owner = first cell in group
        c0, k0, a0, b0 = (int(x) for x in group[0])
        # restore owner CCW orientation
        ca, cb = int(conn[c0, k0]), int(conn[c0, (k0 + 1) % 4])
        face_owner.append(c0)
        face_owner_local.append(k0)
        face_nodes.append((ca, cb))
        if j - i == 1:
            face_neigh.append(-1)
        else:
            # neighbour = the other cell
            c1 = int(group[1, 0])
            face_neigh.append(c1)
            if j - i > 2:
                raise RuntimeError("non-manifold face: more than 2 cells share an edge")
        i = j

    face_owner = np.asarray(face_owner, dtype=np.int64)
    face_neigh = np.asarray(face_neigh, dtype=np.int64)
    face_nodes = np.asarray(face_nodes, dtype=np.int64)
    Nf = len(face_owner)

    # geometry
    pa = nodes[face_nodes[:, 0]]
    pb = nodes[face_nodes[:, 1]]
    face_centers = 0.5 * (pa + pb)
    tang = pb - pa
    face_lengths = np.linalg.norm(tang, axis=1)
    # outward-ish normal from owner: rotate tangent (CCW edge → outward for CCW cell)
    # For CCW quad, edge a→b has outward normal (by, -bx) / |t|? 
    # Inward normal of CCW polygon is rotate tangent left: (-ty, tx).
    # Outward is (ty, -tx).
    normals = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
    # flip so normal points from owner centre toward face (and beyond)
    to_face = face_centers - cell_centers[face_owner]
    flip = np.sum(normals * to_face, axis=1) < 0
    normals[flip] *= -1
    # also swap node order when flipped so (a→b) × normal stays consistent
    face_nodes[flip] = face_nodes[flip][:, ::-1]
    nrm = np.linalg.norm(normals, axis=1)
    nrm = np.maximum(nrm, 1e-30)
    face_normals = normals / nrm[:, None]

    # u-like vs v-like
    face_kind = (np.abs(face_normals[:, 1]) > np.abs(face_normals[:, 0])).astype(np.int8)

    # boundary name per face
    # map boundary edge (sorted node pair) → name
    bnd = mesh["bnd"]
    bnd_names = mesh["bnd_names"]
    bc_lookup = {}
    for (a, b), nm in zip(bnd, bnd_names):
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        bc_lookup[(a, b)] = str(nm)

    bc_names = sorted({nm for nm in bnd_names if nm})
    bc_index = {nm: i for i, nm in enumerate(bc_names)}
    face_bc = np.full(Nf, -1, dtype=np.int32)
    for f in range(Nf):
        if face_neigh[f] >= 0:
            continue
        a, b = int(face_nodes[f, 0]), int(face_nodes[f, 1])
        key = (a, b) if a < b else (b, a)
        nm = bc_lookup.get(key, "")
        if nm in bc_index:
            face_bc[f] = bc_index[nm]

    # convenience masks
    u_faces = np.where(face_kind == 0)[0]
    v_faces = np.where(face_kind == 1)[0]

    stagger = dict(
        cell_centers=cell_centers,
        cell_zone=mesh["zone"].astype(np.int32),
        face_centers=face_centers,
        face_normals=face_normals,
        face_lengths=face_lengths,
        face_owner=face_owner,
        face_neigh=face_neigh,
        face_nodes=face_nodes,
        face_kind=face_kind,
        face_bc=face_bc,
        bc_names=np.array(bc_names, dtype=object),
        u_faces=u_faces,
        v_faces=v_faces,
        n_cells=Nc,
        n_faces=Nf,
    )
    return stagger


def write_staggered(path, stagger, I, bl):
    """Save staggered arrays as a single .npz for calculations."""
    # object array bc_names → save as list of strings via allow_pickle
    np.savez_compressed(
        path,
        cell_centers=stagger["cell_centers"],
        cell_zone=stagger["cell_zone"],
        face_centers=stagger["face_centers"],
        face_normals=stagger["face_normals"],
        face_lengths=stagger["face_lengths"],
        face_owner=stagger["face_owner"],
        face_neigh=stagger["face_neigh"],
        face_nodes=stagger["face_nodes"],
        face_kind=stagger["face_kind"],
        face_bc=stagger["face_bc"],
        bc_names=np.array(list(stagger["bc_names"]), dtype="U64"),
        u_faces=stagger["u_faces"],
        v_faces=stagger["v_faces"],
        # meta
        d_min=I["d_min"],
        d_max=I["d_max"],
        growth_ratio=I["growth_ratio"],
        D=I["D"],
        xc=I["xc"],
        yc=I["yc"],
        a=I["a"],
        T_bl=bl["T"],
        n_bl=bl["n"],
    )


def report_staggered(stagger):
    Nc, Nf = stagger["n_cells"], stagger["n_faces"]
    n_bnd = int(np.sum(stagger["face_neigh"] < 0))
    n_u = len(stagger["u_faces"])
    n_v = len(stagger["v_faces"])
    print("\nSTAGGERED GRID")
    print("-" * 78)
    print(f"cells (P locations)     : {Nc:,}")
    print(f"faces (velocity locs)   : {Nf:,}  ({n_bnd:,} boundary, {Nf - n_bnd:,} interior)")
    print(f"  u-like faces (|nx|≥|ny|): {n_u:,}")
    print(f"  v-like faces (|ny|>|nx|): {n_v:,}")
    print(f"boundary patches        : {list(stagger['bc_names'])}")
    for i, nm in enumerate(stagger["bc_names"]):
        count = int(np.sum(stagger["face_bc"] == i))
        print(f"    {nm:<20} {count:>8} faces")
    print("-" * 78)


def plot_staggered(I, mesh, stagger, path):
    """Zoom near the cylinder: P centres + u/v face centres."""
    a, xc, yc = I["a"], I["xc"], I["yc"]
    Q = mesh["quads"]
    P = stagger["cell_centers"]
    Fc = stagger["face_centers"]
    kind = stagger["face_kind"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # left: full near-field with P
    ax = axes[0]
    ax.add_collection(
        PolyCollection(Q, facecolors="none", edgecolors="#94a3b8", linewidths=0.25)
    )
    ax.plot(P[:, 0], P[:, 1], ".", ms=1.5, color="#1d4ed8", label="P (cell centre)")
    ax.add_patch(plt.Circle((xc, yc), I["D"] / 2, fc="#e5e7eb", ec="k", lw=1, zorder=3))
    ax.set_xlim(xc - a - 0.5, xc + a + 0.5)
    ax.set_ylim(yc - a - 0.5, yc + a + 0.5)
    ax.set_aspect("equal")
    ax.set_title("Staggered P locations (near field)", loc="left", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, markerscale=3)

    # right: zoom BL ring with u/v faces
    ax = axes[1]
    R = I["D"] / 2
    mask_cell = np.hypot(P[:, 0] - xc, P[:, 1] - yc) < R + 0.8
    ax.add_collection(
        PolyCollection(Q[mask_cell], facecolors="none", edgecolors="#cbd5e1", linewidths=0.4)
    )
    # faces in the zoom window
    m = (
        (Fc[:, 0] > xc - 1.2)
        & (Fc[:, 0] < xc + 1.2)
        & (Fc[:, 1] > yc - 1.2)
        & (Fc[:, 1] < yc + 1.2)
    )
    ax.plot(
        Fc[m & (kind == 0), 0],
        Fc[m & (kind == 0), 1],
        ".",
        ms=3,
        color="#dc2626",
        label="u-like face",
    )
    ax.plot(
        Fc[m & (kind == 1), 0],
        Fc[m & (kind == 1), 1],
        ".",
        ms=3,
        color="#16a34a",
        label="v-like face",
    )
    ax.plot(
        P[mask_cell, 0],
        P[mask_cell, 1],
        ".",
        ms=2,
        color="#1d4ed8",
        label="P",
    )
    ax.add_patch(plt.Circle((xc, yc), R, fc="#e5e7eb", ec="k", lw=1, zorder=3))
    ax.set_xlim(xc - 1.2, xc + 1.2)
    ax.set_ylim(yc - 1.2, yc + 1.2)
    ax.set_aspect("equal")
    ax.set_title("Staggered u/v face centres (cylinder BL)", loc="left", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# STEP 6: verification
# =====================================================================
def quad_metrics(Q):
    x, y = Q[:, :, 0], Q[:, :, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1)
    v = np.roll(Q, -1, 1) - Q
    ln = np.linalg.norm(v, axis=2)
    a_in, a_out = -np.roll(v, 1, 1), v
    cosang = np.sum(a_in * a_out, 2) / (np.linalg.norm(a_in, axis=2) * ln)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    return area, ln, np.abs(ang - 90).max(1)


def max_ratio(d):
    d = np.asarray(d)
    r = d[1:] / d[:-1]
    return float(np.max(np.maximum(r, 1 / r)))


def verify(I, bl, g, ob, mesh):
    Q, zone = mesh["quads"], mesh["zone"]
    area, ln, skew = quad_metrics(Q)
    xs, ys, w, n = g["xs"], g["ys"], I["walls"], bl["n"]

    first = {}
    if "top" in w:
        first["top"] = ys[-1] - ys[-2]
    if "bottom" in w:
        first["bottom"] = ys[1] - ys[0]
    if "left" in w:
        first["left"] = xs[1] - xs[0]
    if "right" in w:
        first["right"] = xs[-1] - xs[-2]
    seg = np.linalg.norm(np.diff(ob["nodes"], axis=1), axis=2)
    first["cylinder"] = seg[:, 0].max()
    ok1 = all(abs(v - I["d_min"]) < 1e-9 for v in first.values())

    rect_u = zone == 2
    size_u = np.concatenate(
        [np.ptp(Q[rect_u][:, :, 0], axis=1), np.ptp(Q[rect_u][:, :, 1], axis=1)]
    )
    growth = max(max_ratio(np.diff(xs)), max_ratio(np.diff(ys)))
    if n > 1:
        ring_r = seg[:, 1:n] / seg[:, : n - 1]
        growth = max(growth, float(ring_r.max()), float(1 / ring_r.min()))
    j1 = np.maximum(seg[:, n] / seg[:, n - 1], seg[:, n - 1] / seg[:, n])
    j2 = np.maximum(ob["u"] / g["edge"], g["edge"] / ob["u"])
    jump = float(max(j1.max(), j2.max()))
    unnamed = int(np.sum(mesh["bnd_names"] == ""))

    rules = [
        (
            "first cell = d_min at every wall",
            ok1,
            ", ".join(f"{k} {v:.5f}" for k, v in first.items()),
        ),
        (
            "largest edge, rectangular blocks <= d_max",
            ln[(zone == 1) | (zone == 2)].max() <= I["d_max"] * 1.01,
            f"largest edge {ln[(zone == 1) | (zone == 2)].max():.5f} (d_max {I['d_max']})",
        ),
        (
            "largest edge, curved block",
            (
                ln[(zone == 0) | (zone == 3)].max()
                <= I["d_max"] * (1 + I["corner_margin"]) * 1.01
            )
            if I["transition_count"] == "corner"
            else None,
            f"largest edge {ln[(zone == 0) | (zone == 3)].max():.5f} = "
            f"{ln[(zone == 0) | (zone == 3)].max() / I['d_max']:.2f} x d_max",
        ),
        (
            "uniform region close to d_max",
            size_u.min() >= 0.9 * I["d_max"],
            f"rectangular cells {size_u.min():.5f} to {size_u.max():.5f}",
        ),
        (
            "growth ratio = input ratio",
            growth <= I["growth_ratio"] + 1e-6,
            f"max {growth:.4f} (input {I['growth_ratio']})",
        ),
        (
            "size jumps at block interfaces",
            jump <= I["max_jump"],
            f"ring->transition {j1.max():.3f}, transition->rectangles {j2.max():.3f} "
            f"(limit {I['max_jump']})",
        ),
        ("no inverted cells", area.min() > 0, f"min area {area.min():.3e}"),
        (
            "conforming (no hanging nodes)",
            unnamed == 0,
            f"{unnamed} unclassified boundary edges",
        ),
    ]
    print("\nDERIVED FROM THE INPUTS")
    print("-" * 78)
    print(
        f"boundary layer: {bl['n']} cells, ratio {bl['r']}, "
        f"last cell {bl['d_last']:.5f}, thickness {bl['T']:.4f}"
    )
    print(
        f"around the cylinder: {ob['NT']} cells, {ob['NR']} radial cells "
        f"({bl['n']} in the ring + {ob['N_u']} in the transition)"
    )
    print("\nMESH REPORT")
    print("-" * 78)
    print(f"cells {len(Q):,}   nodes {len(mesh['nodes']):,}")
    for k, v in ZONES.items():
        print(f"   {v:<38}{int(np.sum(zone == k)):>9,} cells")
    print(
        f"max aspect ratio {(ln.max(1) / ln.min(1)).max():.1f}   "
        f"max non-orthogonality {skew.max():.1f} deg"
    )
    print("-" * 78)
    for name, ok, info in rules:
        tag = "INFO" if ok is None else ("PASS" if ok else "FAIL")
        print(f"[{tag}] {name:<38} {info}")
    print("-" * 78)
    return rules


# =====================================================================
# STEP 7: export and plot
# =====================================================================
def write_vtk(path, mesh):
    nodes, conn, zone = mesh["nodes"], mesh["conn"], mesh["zone"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\ncylinder mesh\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(nodes)} float\n")
        np.savetxt(f, np.column_stack([nodes, np.zeros(len(nodes))]), fmt="%.9g")
        f.write(f"CELLS {len(conn)} {5 * len(conn)}\n")
        np.savetxt(f, np.column_stack([np.full(len(conn), 4), conn]), fmt="%d")
        f.write(f"CELL_TYPES {len(conn)}\n")
        np.savetxt(f, np.full(len(conn), 9), fmt="%d")
        f.write(f"CELL_DATA {len(conn)}\nSCALARS zone int 1\nLOOKUP_TABLE default\n")
        np.savetxt(f, zone, fmt="%d")


def write_gmsh(path, mesh):
    nodes, conn = mesh["nodes"], mesh["conn"]
    bn = sorted(set(mesh["bnd_names"]) - {""})
    fluid = len(bn) + 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n{fluid}\n")
        for k, nme in enumerate(bn, 1):
            f.write(f'1 {k} "{nme}"\n')
        f.write(f'2 {fluid} "fluid"\n$EndPhysicalNames\n$Nodes\n{len(nodes)}\n')
        for i, (x, y) in enumerate(nodes, 1):
            f.write(f"{i} {x:.12g} {y:.12g} 0\n")
        f.write("$EndNodes\n")
        lines = [
            (bn.index(nm) + 1, a, b)
            for (a, b), nm in zip(mesh["bnd"], mesh["bnd_names"])
            if nm
        ]
        f.write(f"$Elements\n{len(lines) + len(conn)}\n")
        eid = 1
        for tag, a, b in lines:
            f.write(f"{eid} 1 2 {tag} {tag} {a + 1} {b + 1}\n")
            eid += 1
        for c in conn:
            f.write(
                f"{eid} 3 2 {fluid} {fluid} "
                f"{c[0]+1} {c[1]+1} {c[2]+1} {c[3]+1}\n"
            )
            eid += 1
        f.write("$EndElements\n")


def plot_mesh(I, bl, mesh, path):
    Q, zone = mesh["quads"], mesh["zone"]
    col = {0: "#1d4ed8", 1: "#ea580c", 2: "#6b7280", 3: "#0891b2"}
    cen = Q.mean(axis=1)
    a, xc, yc = I["a"], I["xc"], I["yc"]

    def draw(ax, xlim, ylim, lw):
        inside = (
            (cen[:, 0] > xlim[0] - 1)
            & (cen[:, 0] < xlim[1] + 1)
            & (cen[:, 1] > ylim[0] - 1)
            & (cen[:, 1] < ylim[1] + 1)
        )
        for z in range(4):
            sel = inside & (zone == z)
            ax.add_collection(
                PolyCollection(Q[sel], facecolors="none", edgecolors=col[z], linewidths=lw)
            )
        ax.add_patch(
            plt.Circle((xc, yc), I["D"] / 2, fc="#e5e7eb", ec="k", lw=1, zorder=3)
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)
        for s, val, horiz in (
            ("top", I["y_max"], True),
            ("bottom", I["y_min"], True),
            ("left", I["x_min"], False),
            ("right", I["x_max"], False),
        ):
            if s in I["walls"]:
                (ax.axhline if horiz else ax.axvline)(val, color="k", lw=2.5)

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.28, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    draw(ax, (I["x_min"], I["x_max"]), (I["y_min"], I["y_max"]), 0.12)
    ax.set_title(f"Full domain ({len(Q):,} cells)", loc="left", fontsize=11, fontweight="bold")
    ax1 = fig.add_subplot(gs[1, 0])
    draw(ax1, (xc - a - 0.6, xc + a + 0.6), (yc - a - 0.6, yc + a + 0.6), 0.4)
    ax1.set_title("Zoom 1: boundary layer on the cylinder", loc="left", fontsize=11, fontweight="bold")
    if "top" in I["walls"]:
        ax2 = fig.add_subplot(gs[1, 1])
        draw(
            ax2,
            (xc - 1, xc + 1),
            (I["y_max"] - bl["T"] - 0.5, I["y_max"] + 0.1),
            0.4,
        )
        ax2.set_title("Zoom 2: top-wall boundary layer", loc="left", fontsize=11, fontweight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# main
# =====================================================================
def main(I=INPUT, tag="result"):
    I = dict(I)
    bl = boundary_layer(I["d_min"], I["d_max"], I["growth_ratio"])
    if I["a"] is None:
        I["a"] = choose_a(I, bl)
    g = build_grid_lines(I, bl)
    region_report(I, bl, g)
    ob = build_o_block(I, bl, g)
    rq, rzone = build_rect_blocks(I, bl, g)
    mesh = assemble(I, ob, rq, rzone)
    verify(I, bl, g, ob, mesh)

    # staggered locations for calculations
    stagger = extract_staggered(mesh)
    report_staggered(stagger)

    out = "mesh_output"
    os.makedirs(out, exist_ok=True)
    write_vtk(f"{out}/mesh_{tag}.vtk", mesh)
    write_gmsh(f"{out}/mesh_{tag}.msh", mesh)
    plot_mesh(I, bl, mesh, f"{out}/mesh_{tag}.png")
    write_staggered(f"{out}/staggered_{tag}.npz", stagger, I, bl)
    plot_staggered(I, mesh, stagger, f"{out}/staggered_{tag}.png")
    print(f"written to ./{out}/mesh_{tag}.(vtk|msh|png)")
    print(f"written to ./{out}/staggered_{tag}.(npz|png)")
    return mesh, stagger


if __name__ == "__main__":
    main()
