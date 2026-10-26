#!/usr/bin/env python3
"""
mesh_from_known_sizes_3d.py
===========================
Spanwise extrusion of the 2-D O–H staggered mesh from
``mesh_from_known_sizes.py`` (that file is imported, never edited).

The xy section is the known-sizes cylinder mesh. It is copied into ``nz``
layers of thickness ``dz = Lz / nz``. Spanwise faces are periodic in z by
default (the usual cylinder-LES condition).

Cell index
    cid = k * Nc2 + c     layer k = 0 … nz−1,  2-D cell c

Face kinds
    0  u-like lateral   (extruded 2-D |n_x| ≥ |n_y| faces), area = L_2d · dz
    1  v-like lateral   (extruded 2-D |n_y| >  |n_x| faces), area = L_2d · dz
    2  w-like spanwise  (between layers, periodic),           area = A_2d

    python mesh_from_known_sizes_3d.py
    python mesh_from_known_sizes_3d.py --quick --nz 4 --Lz 3.1416

Needs numpy and matplotlib. Requires ``nz >= 2`` (a single periodic layer
would make a spanwise face connect a cell to itself).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mesh_from_known_sizes as m2  # noqa: E402  (2-D mesher, read-only)


PRESETS = {
    "smoke": dict(d_min=0.05, d_max=0.30, growth_ratio=1.20, max_jump=2.5, a=2.0),
    "fine": dict(d_min=0.015, d_max=0.12, growth_ratio=1.15, max_jump=2.0, a=None),
}


def volumes_2d(S: dict) -> np.ndarray:
    """2-D cell areas from the staggered face geometry (same as the 2-D solver)."""
    Fc, n, L = S["face_centers"], S["face_normals"], S["face_lengths"]
    own, nei = S["face_owner"], S["face_neigh"]
    Nc = len(S["cell_centers"])
    contrib = 0.5 * (Fc[:, 0] * n[:, 0] + Fc[:, 1] * n[:, 1]) * L
    vol = np.zeros(Nc)
    np.add.at(vol, own, contrib)
    interior = nei >= 0
    np.add.at(vol, nei[interior], -contrib[interior])
    return np.maximum(vol, 1e-30)


def load_staggered_2d(path: Path) -> dict:
    raw = np.load(path, allow_pickle=False)
    out = {k: raw[k] for k in raw.files}
    out["bc_names"] = [str(x) for x in raw["bc_names"].tolist()]
    return out


def ensure_staggered_2d(preset: str | None, build: bool) -> Path:
    """Reuse or build mesh_output/staggered_{tag}.npz via the 2-D mesher."""
    tag = preset if preset is not None else "result"
    path = HERE / "mesh_output" / f"staggered_{tag}.npz"
    if path.is_file() and not build:
        print(f"[mesh-3d] reusing 2-D mesh {path}")
        return path
    I = dict(m2.INPUT)
    if preset is not None:
        I.update(PRESETS[preset])
    print(f"[mesh-3d] building 2-D known-sizes mesh → {path}")
    print(
        f"[mesh-3d] d_min={I['d_min']}  d_max={I['d_max']}  "
        f"growth={I['growth_ratio']}"
    )
    cwd = os.getcwd()
    try:
        os.chdir(HERE)
        m2.main(I, tag=tag)
    finally:
        os.chdir(cwd)
    return HERE / "mesh_output" / f"staggered_{tag}.npz"


def extrude_staggered(
    S2: dict, nz: int, Lz: float, periodic: bool = True
) -> dict:
    """
    Extrude a 2-D staggered mesh into ``nz`` layers of height ``Lz``.

    Periodic z (default) adds ``nz`` spanwise faces per 2-D cell, wrapping
    layer ``nz-1`` onto layer ``0``. Non-periodic adds wall patches
    ``zmin`` / ``zmax`` instead.
    """
    if nz < 2:
        raise ValueError("nz must be >= 2 (periodic self-faces are not allowed)")
    if Lz <= 0:
        raise ValueError("Lz must be positive")

    P2 = np.asarray(S2["cell_centers"], dtype=np.float64)
    Fc2 = np.asarray(S2["face_centers"], dtype=np.float64)
    n2 = np.asarray(S2["face_normals"], dtype=np.float64)
    L2 = np.asarray(S2["face_lengths"], dtype=np.float64)
    own2 = np.asarray(S2["face_owner"], dtype=np.int64)
    nei2 = np.asarray(S2["face_neigh"], dtype=np.int64)
    kind2 = np.asarray(S2["face_kind"], dtype=np.int8)
    bc2 = np.asarray(S2["face_bc"], dtype=np.int32)
    zone2 = np.asarray(S2["cell_zone"], dtype=np.int32)
    Nc2, Nf2 = len(P2), len(Fc2)
    dz = float(Lz) / int(nz)
    vol2 = volumes_2d(S2)

    # ---- cells ----
    k_cell = np.arange(nz)
    zc = (k_cell + 0.5) * dz
    P = np.empty((nz * Nc2, 3), dtype=np.float64)
    P[:, 0] = np.tile(P2[:, 0], nz)
    P[:, 1] = np.tile(P2[:, 1], nz)
    P[:, 2] = np.repeat(zc, Nc2)
    zone = np.tile(zone2, nz)
    vol = np.tile(vol2 * dz, nz)

    # ---- lateral faces (extruded 2-D edges) ----
    Nf_lat = nz * Nf2
    Fc_lat = np.empty((Nf_lat, 3), dtype=np.float64)
    Fc_lat[:, 0] = np.tile(Fc2[:, 0], nz)
    Fc_lat[:, 1] = np.tile(Fc2[:, 1], nz)
    Fc_lat[:, 2] = np.repeat(zc, Nf2)
    n_lat = np.zeros((Nf_lat, 3), dtype=np.float64)
    n_lat[:, 0] = np.tile(n2[:, 0], nz)
    n_lat[:, 1] = np.tile(n2[:, 1], nz)
    area_lat = np.tile(L2 * dz, nz)
    layer_f = np.repeat(k_cell, Nf2)
    own_lat = np.tile(own2, nz) + layer_f * Nc2
    nei_lat = np.tile(nei2, nz)
    interior2 = nei_lat >= 0
    nei_lat[interior2] += layer_f[interior2] * Nc2
    kind_lat = np.tile(kind2, nz)
    bc_lat = np.tile(bc2, nz)

    bc_names = [str(x) for x in S2["bc_names"]]

    # ---- spanwise faces ----
    if periodic:
        Nf_z = nz * Nc2
        Fc_z = np.empty((Nf_z, 3), dtype=np.float64)
        Fc_z[:, 0] = np.tile(P2[:, 0], nz)
        Fc_z[:, 1] = np.tile(P2[:, 1], nz)
        Fc_z[:, 2] = np.repeat(k_cell * dz, Nc2)  # face k at z = k dz
        n_z = np.zeros((Nf_z, 3), dtype=np.float64)
        n_z[:, 2] = 1.0
        area_z = np.tile(vol2, nz)
        layer_k = np.repeat(k_cell, Nc2)
        c_tile = np.tile(np.arange(Nc2), nz)
        own_z = ((layer_k - 1) % nz) * Nc2 + c_tile
        nei_z = layer_k * Nc2 + c_tile
        kind_z = np.full(Nf_z, 2, dtype=np.int8)
        bc_z = np.full(Nf_z, -1, dtype=np.int32)
    else:
        # nz+1 faces per 2-D cell: interior at k=1…nz−1, walls at 0 and nz
        k_face = np.arange(nz + 1)
        Nf_z = (nz + 1) * Nc2
        Fc_z = np.empty((Nf_z, 3), dtype=np.float64)
        Fc_z[:, 0] = np.tile(P2[:, 0], nz + 1)
        Fc_z[:, 1] = np.tile(P2[:, 1], nz + 1)
        Fc_z[:, 2] = np.repeat(k_face * dz, Nc2)
        n_z = np.zeros((Nf_z, 3), dtype=np.float64)
        n_z[:, 2] = 1.0
        area_z = np.tile(vol2, nz + 1)
        layer_k = np.repeat(k_face, Nc2)
        c_tile = np.tile(np.arange(Nc2), nz + 1)
        own_z = np.empty(Nf_z, dtype=np.int64)
        nei_z = np.empty(Nf_z, dtype=np.int64)
        bc_z = np.full(Nf_z, -1, dtype=np.int32)
        lo = layer_k == 0
        hi = layer_k == nz
        mid = ~lo & ~hi
        own_z[lo] = c_tile[lo]
        nei_z[lo] = -1
        n_z[lo, 2] = -1.0
        own_z[hi] = (nz - 1) * Nc2 + c_tile[hi]
        nei_z[hi] = -1
        own_z[mid] = (layer_k[mid] - 1) * Nc2 + c_tile[mid]
        nei_z[mid] = layer_k[mid] * Nc2 + c_tile[mid]
        kind_z = np.full(Nf_z, 2, dtype=np.int8)
        bc_names = bc_names + ["zmin", "zmax"]
        izmin, izmax = bc_names.index("zmin"), bc_names.index("zmax")
        bc_z[lo] = izmin
        bc_z[hi] = izmax

    face_centers = np.concatenate([Fc_lat, Fc_z], axis=0)
    face_normals = np.concatenate([n_lat, n_z], axis=0)
    face_areas = np.concatenate([area_lat, area_z], axis=0)
    face_owner = np.concatenate([own_lat, own_z], axis=0)
    face_neigh = np.concatenate([nei_lat, nei_z], axis=0)
    face_kind = np.concatenate([kind_lat, kind_z], axis=0)
    face_bc = np.concatenate([bc_lat, bc_z], axis=0)

    return dict(
        cell_centers=P,
        cell_zone=zone,
        cell_volumes=vol,
        face_centers=face_centers,
        face_normals=face_normals,
        face_areas=face_areas,
        face_owner=face_owner,
        face_neigh=face_neigh,
        face_kind=face_kind,
        face_bc=face_bc,
        bc_names=np.array(bc_names, dtype="U64"),
        u_faces=np.where(face_kind == 0)[0],
        v_faces=np.where(face_kind == 1)[0],
        w_faces=np.where(face_kind == 2)[0],
        n_cells=int(nz * Nc2),
        n_faces=int(len(face_owner)),
        n_cells_2d=int(Nc2),
        n_faces_2d=int(Nf2),
        nz=int(nz),
        Lz=float(Lz),
        dz=float(dz),
        periodic_z=bool(periodic),
        D=float(S2["D"]),
        xc=float(S2["xc"]),
        yc=float(S2["yc"]),
        d_min=float(S2["d_min"]),
        d_max=float(S2["d_max"]),
        a=float(S2["a"]),
        quads=np.asarray(S2["quads"], dtype=np.float64) if "quads" in S2 else None,
    )


def write_staggered_3d(path: Path, S: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in S.items() if v is not None}
    np.savez_compressed(path, **payload)


def report(S: dict) -> None:
    Nc, Nf, nz = S["n_cells"], S["n_faces"], S["nz"]
    n_bnd = int(np.sum(S["face_neigh"] < 0))
    print("\nSTAGGERED GRID 3-D")
    print("-" * 78)
    print(f"cells (P locations)     : {Nc:,}  = {S['n_cells_2d']:,} (xy) × {nz} (z)")
    print(f"faces (velocity locs)   : {Nf:,}  ({n_bnd:,} boundary, {Nf - n_bnd:,} interior)")
    print(f"  u-like (lateral)      : {len(S['u_faces']):,}")
    print(f"  v-like (lateral)      : {len(S['v_faces']):,}")
    print(f"  w-like (spanwise)     : {len(S['w_faces']):,}")
    print(f"Lz / nz / dz            : {S['Lz']:g} / {nz} / {S['dz']:.5g}")
    print(f"periodic z              : {S['periodic_z']}")
    print(f"boundary patches        : {list(S['bc_names'])}")
    for i, nm in enumerate(S["bc_names"]):
        count = int(np.sum(S["face_bc"] == i))
        print(f"    {nm:<20} {count:>8} faces")
    print("-" * 78)


def plot_mesh_3d(S: dict, path: Path) -> None:
    """xy slice of the 2-D section plus an xz sketch of the layers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    quads = S.get("quads")
    xc, yc, R = S["xc"], S["yc"], S["D"] / 2
    if quads is not None:
        ax0.add_collection(
            PolyCollection(
                quads, facecolors="none", edgecolors="#4b5563", linewidths=0.15
            )
        )
    ax0.add_patch(plt.Circle((xc, yc), R, fc="#e5e7eb", ec="k", lw=1.0, zorder=3))
    ax0.set_aspect("equal")
    P2 = S["cell_centers"][: S["n_cells_2d"]]
    ax0.set_xlim(P2[:, 0].min() - 0.3, P2[:, 0].max() + 0.3)
    ax0.set_ylim(P2[:, 1].min() - 0.3, P2[:, 1].max() + 0.3)
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_title(
        f"xy section  ({S['n_cells_2d']:,} quads, extruded × {S['nz']})",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )

    # xz: a few cells near the centreline y ≈ yc
    P = S["cell_centers"]
    mid = np.abs(P[:, 1] - yc) < 0.15
    ax1.scatter(P[mid, 0], P[mid, 2], s=4, c=P[mid, 2], cmap="viridis", linewidths=0)
    ax1.axvline(xc - R, color="k", lw=1.0)
    ax1.axvline(xc + R, color="k", lw=1.0)
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    ax1.set_title(
        f"xz cut (cells with |y−yc|<0.15), Lz={S['Lz']:g}, nz={S['nz']}",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    ax1.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Extrude the known-sizes O–H mesh in z (3-D staggered)."
    )
    p.add_argument("--quick", action="store_true", help="smoke 2-D preset")
    p.add_argument("--fine", action="store_true", help="fine 2-D preset")
    p.add_argument("--build-mesh", action="store_true", help="rebuild the 2-D section")
    p.add_argument("--nz", type=int, default=4, help="spanwise cells (default 4)")
    p.add_argument(
        "--Lz",
        type=float,
        default=float(np.pi),
        help="spanwise length (default π D, D=1)",
    )
    p.add_argument(
        "--no-periodic",
        action="store_true",
        help="zmin/zmax walls instead of periodic z",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="output npz (default mesh_output/staggered_3d_<tag>_nz<nz>.npz)",
    )
    return p.parse_args()


def main(
    preset: str | None = None,
    nz: int = 4,
    Lz: float = float(np.pi),
    periodic: bool = True,
    build: bool = False,
    out: Path | None = None,
) -> dict:
    path2 = ensure_staggered_2d(preset, build)
    S2 = load_staggered_2d(path2)
    S = extrude_staggered(S2, nz=nz, Lz=Lz, periodic=periodic)
    report(S)
    tag = preset if preset is not None else "result"
    if out is None:
        out = HERE / "mesh_output" / f"staggered_3d_{tag}_nz{nz}.npz"
    write_staggered_3d(out, S)
    png = out.with_suffix(".png")
    plot_mesh_3d(S, png)
    print(f"written {out}")
    print(f"written {png}")
    return S


if __name__ == "__main__":
    args = parse_args()
    preset = "fine" if args.fine else ("smoke" if args.quick else None)
    main(
        preset=preset,
        nz=args.nz,
        Lz=args.Lz,
        periodic=not args.no_periodic,
        build=args.build_mesh,
        out=Path(args.out) if args.out else None,
    )
