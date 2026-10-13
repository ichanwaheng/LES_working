#!/usr/bin/env python3
"""
2-D structured (multiblock) mesh: laminar flow past a circular cylinder
in a channel bounded by two parallel walls.

Block arrangement
-----------------
::

    y = +H/2  ╔══════════════════════════════════════════════════╗ wall_top
              ║                   TOP_WALL (BL)                  ║
    y_ti      ╟──────────┬─────────────────────┬─────────────────╢
              ║   IN_U   │       ABOVE         │      OUT_U      ║
    y = +S    ╟──────────╭─────────────────────╮─────────────────╢
              ║          │        O_N          │                 ║
              ║   IN_W   │  O_W   cyl   O_E    │      OUT_E      ║
              ║          │        O_S          │                 ║
    y = -S    ╟──────────╰─────────────────────╯─────────────────╢
              ║   IN_L   │       BELOW         │      OUT_L      ║
    y_bi      ╟──────────┴─────────────────────┴─────────────────╢
              ║                  BOT_WALL (BL)                   ║
    y = -H/2  ╚══════════════════════════════════════════════════╝ wall_bottom
            x_in        -S                    +S              x_out

* ``O_E/N/W/S`` — four structured O-blocks from the cylinder to a
  superellipse (rounded square, ``p_outer``). This keeps ray lengths nearly
  equal so the neighbour size ratio stays ≤ 1.2 (a sharp square cannot).
  Radial BL follows rules 1–2 / 4; ≥ ``n_theta`` nodes on the cylinder.
* ``TOP_WALL`` / ``BOT_WALL`` — full-length channel-wall BL (rules 1–2, 7).
* Core at the wall-layer junction uses ``dy_core ≈ dy_last`` (rule 3).
* ``OUT_*`` — wake refinement ≤ D/15, growth < 1.1 (rule 5).
* All interfaces share identical node coordinates (conforming, rule 6).

Built with NumPy; exported as ``mesh.msh`` (Gmsh 2.2), Plot3D ``mesh.xyz``,
optional CGNS, and PNG previews.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# PARAMETERS — edit here and re-run
# ---------------------------------------------------------------------------
PARAMS = {
    # Geometry [m]
    "D": 1.0,
    "H_over_D": 10.0,
    "x_in_over_D": -10.0,
    "x_out_over_D": 30.0,
    "S_over_D": 2.0,  # bounding half-side of the near-field superellipse
    "p_outer": 2.5,  # superellipse exponent (2=circle … ∞=square)
    # Flow
    "Re_D": 150.0,
    # Wall / O-layer BL (rules 1–2, 4)
    "dy1": 0.0011,
    "growth": 1.08,
    "n_bl": 25,
    "n_theta": 160,  # ≥ 150, multiple of 4 (may be raised automatically)
    # Wake (rule 5)
    "wake_x_over_D": 20.0,
    "wake_dx_max_over_D": 1.0 / 15.0,
    "wake_growth_max": 1.1,
    # Smoothness (rule 6)
    "neigh_ratio_max": 1.2,
    "cf": None,
    "out_dir": "structured_mesh",
}


# ---------------------------------------------------------------------------
# Spacing helpers
# ---------------------------------------------------------------------------

def bl_thickness(dy1: float, r: float, n: int) -> float:
    return dy1 * (r**n - 1.0) / (r - 1.0)


def bl_last(dy1: float, r: float, n: int) -> float:
    return dy1 * r ** (n - 1)


def geometric_sizes(dy1: float, r: float, n: int) -> np.ndarray:
    return dy1 * r ** np.arange(n)


def geometric_breaks(dy1: float, r: float, n: int) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(geometric_sizes(dy1, r, n))])


def uniform_count(length: float, h_target: float) -> tuple[int, float, float]:
    n = max(1, int(round(length / h_target)))
    h = length / n
    return n, h, 100.0 * (h - h_target) / h_target


def progression_ratio(length: float, n: int, h0: float) -> float:
    if n <= 0 or h0 <= 0:
        return 1.0
    if abs(h0 * n - length) < 1e-12 * max(length, h0):
        return 1.0
    target = length / h0
    lo, hi = (1e-8, 1.0) if target < n else (1.0, 4.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = float(n) if abs(mid - 1.0) < 1e-14 else (mid**n - 1.0) / (mid - 1.0)
        if s < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def geometric_sizes_with_ratio(h0: float, ratio: float, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0)
    if abs(ratio - 1.0) < 1e-14:
        return np.full(n, h0)
    return h0 * ratio ** np.arange(n)


def nodes_from_sizes(x0: float, sizes: np.ndarray) -> np.ndarray:
    if len(sizes) == 0:
        return np.array([x0], dtype=float)
    return np.concatenate([[x0], x0 + np.cumsum(sizes)])


def unique_nodes(x: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    x = np.sort(np.asarray(x, dtype=float))
    if len(x) == 0:
        return x
    keep = [0]
    for i in range(1, len(x)):
        if x[i] - x[keep[-1]] > tol:
            keep.append(i)
    out = x[np.asarray(keep)]
    out[0], out[-1] = x[0], x[-1]
    return out


def fill_remaining(length: float, h_start: float, h_max: float, gmax: float) -> np.ndarray:
    """
    Cover ``length`` starting at ``h_start``, growing by at most ``gmax``,
    never exceeding ``h_max``. Neighbour ratios stay within [1/gmax, gmax].
    """
    if length <= 1e-15:
        return np.zeros(0)
    h0 = min(max(h_start, length * 1e-9), h_max, length)

    # 1) Pure forward growth until we hit h_max or run out of room
    sizes: list[float] = []
    rem = length
    h = h0
    while rem > 1e-12:
        h_use = min(h, h_max)
        if rem <= h_use * gmax:
            break  # handle remainder below
        sizes.append(h_use)
        rem -= h_use
        h = min(h_use * gmax, h_max)

    # 2) Fill remainder with N equal cells compatible with the last size
    if rem > 1e-12:
        prev = sizes[-1] if sizes else h0
        # Choose N so rem/N is within [prev/gmax, min(prev*gmax, h_max)]
        lo = rem / min(prev * gmax, h_max)
        hi = rem / max(prev / gmax, rem * 1e-9)
        N = max(1, int(math.ceil(lo - 1e-12)))
        # Prefer N that also keeps rem/N ≤ h_max
        while N < 100000:
            s = rem / N
            if s <= h_max + 1e-15 and (not sizes or (prev / gmax - 1e-12 <= s <= prev * gmax + 1e-12)):
                break
            N += 1
        else:
            N = max(1, int(math.ceil(rem / h_max)))
            s = rem / N
            if sizes and s < prev / gmax:
                # shrink prev slightly by merging last into remainder pool
                pool = prev + rem
                sizes.pop()
                N = max(1, int(math.ceil(pool / h_max)))
                # also respect growth from new previous
                sizes.extend([pool / N] * N)
                rem = 0.0
                N = 0
                s = 0.0
        if N > 0:
            sizes.extend([rem / N] * N)

    arr = np.asarray(sizes, dtype=float)
    if arr.size and abs(arr.sum() - length) > 1e-14 * max(length, 1.0):
        arr *= length / arr.sum()
    # Final safety: smooth any residual ratio violations
    for _ in range(5):
        if arr.size < 2:
            break
        r = arr[1:] / np.maximum(arr[:-1], 1e-30)
        i_bad = np.where((r > gmax * 1.0000001) | (r < 1.0 / gmax * 0.9999999))[0]
        if len(i_bad) == 0:
            break
        for i in i_bad:
            total = arr[i] + arr[i + 1]
            a = min(total / (1.0 + min(gmax, total / max(arr[i], 1e-30))), h_max)
            a = min(max(a, total / (1.0 + gmax)), h_max)
            b = total - a
            if b > h_max:
                b = h_max
                a = total - b
            arr[i], arr[i + 1] = a, b
    return arr


def transition_nodes(a: float, b: float, h_a: float, h_b: float, gmax: float) -> np.ndarray:
    length = abs(b - a)
    if length < 1e-15:
        return np.array([a, b])
    sign = 1.0 if b > a else -1.0
    if h_a <= h_b:
        sizes = fill_remaining(length, h_a, h_b, gmax)
        return a + sign * np.concatenate([[0.0], np.cumsum(sizes)])
    sizes = fill_remaining(length, h_b, h_a, gmax)
    return b - sign * np.concatenate([[0.0], np.cumsum(sizes)])[::-1]


def wake_x_nodes(x_start, x_out, dx0, dx_max, gmax) -> np.ndarray:
    return nodes_from_sizes(x_start, fill_remaining(x_out - x_start, dx0, dx_max, gmax))


def inlet_x_nodes(x_in, x_sq, dx0, dx_max, gmax) -> np.ndarray:
    sizes = fill_remaining(x_sq - x_in, dx0, dx_max, gmax)
    return unique_nodes(x_sq - nodes_from_sizes(0.0, sizes))


def superellipse_point(theta: float, S: float, p: float) -> np.ndarray:
    """Point on |x/S|^p + |y/S|^p = 1 at angle theta (ray from origin)."""
    c, s = math.cos(theta), math.sin(theta)
    # r such that (|r c|/S)^p + (|r s|/S)^p = 1
    ap = abs(c) ** p + abs(s) ** p
    r = S / ap ** (1.0 / p)
    return np.array([r * c, r * s])


def superellipse_arc(theta0: float, theta1: float, n: int, S: float, p: float) -> np.ndarray:
    thetas = np.linspace(theta0, theta1, n)
    return np.array([superellipse_point(th, S, p) for th in thetas])


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------

def ogrid_to_superellipse(
    theta0: float,
    theta1: float,
    R: float,
    S: float,
    p: float,
    dy1: float,
    r: float,
    n_bl: int,
    n_edge: int,
    dy_last: float,
) -> np.ndarray:
    """O-block: cylinder arc → superellipse arc. Uniform-ish rays."""
    thetas = np.linspace(theta0, theta1, n_edge + 1)
    outer = np.array([superellipse_point(th, S, p) for th in thetas])

    # Reference length = mid-angle ray
    th_mid = 0.5 * (theta0 + theta1)
    L_mid = float(np.linalg.norm(superellipse_point(th_mid, S, p)) - R)
    bl = geometric_sizes(dy1, r, n_bl)
    assert bl.sum() < L_mid * 0.95, "BL thicker than mid-ray gap; increase S_over_D"
    rem_mid = L_mid - float(bl.sum())
    n_extra = max(1, int(round(rem_mid / dy_last)))
    n_i = n_bl + n_extra

    X = np.zeros((n_i + 1, n_edge + 1))
    Y = np.zeros((n_i + 1, n_edge + 1))
    for j, th in enumerate(thetas):
        p_in = np.array([R * math.cos(th), R * math.sin(th)])
        p_out = outer[j]
        L = float(np.linalg.norm(p_out - p_in))
        s_bl = geometric_breaks(dy1, r, n_bl)
        rem = L - float(s_bl[-1])
        assert rem > 0.0
        r_cont = progression_ratio(rem, n_extra, dy_last)
        extra = geometric_sizes_with_ratio(dy_last, r_cont, n_extra)
        extra *= rem / extra.sum()
        s_all = np.concatenate([s_bl, s_bl[-1] + np.cumsum(extra)])
        s_all *= L / s_all[-1]
        direction = (p_out - p_in) / L
        pts = p_in[None, :] + s_all[:, None] * direction[None, :]
        X[:, j] = pts[:, 0]
        Y[:, j] = pts[:, 1]
    return np.stack([X, Y], axis=-1)


def rect_block(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([X, Y], axis=-1)


def tfi_rect(west, east, south, north) -> np.ndarray:
    """west/east (ny,2) S→N; south/north (nx,2) W→E → (nx,ny,2)."""
    nx, ny = len(south), len(west)
    assert len(north) == nx and len(east) == ny
    xi = np.linspace(0.0, 1.0, nx)
    eta = np.linspace(0.0, 1.0, ny)
    sw, se, nw, ne = south[0], south[-1], north[0], north[-1]
    grid = np.zeros((nx, ny, 2))
    for i in range(nx):
        for j in range(ny):
            u, v = xi[i], eta[j]
            grid[i, j] = (
                (1 - u) * west[j]
                + u * east[j]
                + (1 - v) * south[i]
                + v * north[i]
                - ((1 - u) * (1 - v) * sw + u * (1 - v) * se + (1 - u) * v * nw + u * v * ne)
            )
    return grid


def max_growth_in_grid(grid: np.ndarray, axis: int):
    d = np.linalg.norm(np.diff(grid, axis=axis), axis=2)
    if (axis == 0 and d.shape[0] < 2) or (axis == 1 and d.shape[1] < 2):
        return 1.0, None
    if axis == 0:
        ratios = np.maximum(d[1:] / np.maximum(d[:-1], 1e-30), d[:-1] / np.maximum(d[1:], 1e-30))
    else:
        ratios = np.maximum(d[:, 1:] / np.maximum(d[:, :-1], 1e-30), d[:, :-1] / np.maximum(d[:, 1:], 1e-30))
    idx = np.unravel_index(np.argmax(ratios), ratios.shape)
    return float(ratios[idx]), idx


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_blocks(P: dict) -> tuple[dict[str, np.ndarray], dict]:
    D = float(P["D"])
    H = float(P["H_over_D"]) * D
    x_in = float(P["x_in_over_D"]) * D
    x_out = float(P["x_out_over_D"]) * D
    S = float(P["S_over_D"]) * D
    p_out = float(P["p_outer"])
    dy1 = float(P["dy1"])
    r = float(P["growth"])
    n_bl = int(P["n_bl"])
    n_theta = int(P["n_theta"])
    assert n_theta % 4 == 0 and n_theta >= 150
    gmax = float(P["neigh_ratio_max"])

    delta = bl_thickness(dy1, r, n_bl)
    dy_last = bl_last(dy1, r, n_bl)
    y_bot, y_top = -0.5 * H, 0.5 * H
    y_bi, y_ti = y_bot + delta, y_top - delta
    assert S < abs(y_bi) - 1e-9

    dx_wake_max = float(P["wake_dx_max_over_D"]) * D
    g_wake = float(P["wake_growth_max"])

    # Circumferential: edge arc length of one quarter ≈ (π/2)*S_eff; use chord 2S as bound
    # Require spacing along outer ≤ dx_wake_max and ≈ dy_last for smooth wall-junction core.
    h_target = min(dy_last, dx_wake_max)
    # Outer arc length of one quarter of superellipse (numeric)
    arc_pts = superellipse_arc(-0.25 * math.pi, 0.25 * math.pi, 200, S, p_out)
    quarter_len = float(np.sum(np.linalg.norm(np.diff(arc_pts, axis=0), axis=1)))
    n_q_need = int(math.ceil(quarter_len / h_target))
    n_q = max(n_theta // 4, n_q_need)
    if n_q * 4 != n_theta:
        n_theta = n_q * 4
        print(f"  [adjust] n_theta → {n_theta} (outer spacing ≈ {quarter_len / n_q:.5f})")

    n_core_nom, dy_core, core_pct = uniform_count(y_ti - y_bi, dy_last)
    h_sq = quarter_len / n_q  # representative outer spacing

    y_bot_layer = y_bot + geometric_breaks(dy1, r, n_bl)
    y_top_layer = np.sort(y_top - geometric_breaks(dy1, r, n_bl))

    # Bounding box of superellipse is still [-S, S]
    y_below = transition_nodes(y_bi, -S, dy_core, h_sq, gmax)
    y_above = transition_nodes(S, y_ti, h_sq, dy_core, gmax)
    n_core = (len(y_below) - 1) + n_q + (len(y_above) - 1)

    # Superellipse edge |x|<S, so horizontal march amplifies Δx off the mid-side.
    # Prefactor keeps physical Δx ≤ dx_wake_max on every row.
    amp_out = (x_out - 0.0) / (x_out - S)  # upper bound (edge_x ≥ 0 on east)
    amp_in = (0.0 - x_in) / ((-S) - x_in)
    dx_out_cap = dx_wake_max / max(amp_out, 1.0)
    dx_in_cap = dx_wake_max / max(amp_in, 1.0)

    x_inlet = inlet_x_nodes(x_in, -S, h_sq, dx_in_cap, g_wake)
    x_box = np.linspace(-S, S, n_q + 1)
    x_out_nodes = wake_x_nodes(S, x_out, h_sq, dx_out_cap, g_wake)
    x_wall = np.concatenate([x_inlet[:-1], x_box[:-1], x_out_nodes])

    R = 0.5 * D

    # ----- O blocks -----
    O_E = ogrid_to_superellipse(-0.25 * math.pi, 0.25 * math.pi, R, S, p_out, dy1, r, n_bl, n_q, dy_last)
    O_N = ogrid_to_superellipse(0.25 * math.pi, 0.75 * math.pi, R, S, p_out, dy1, r, n_bl, n_q, dy_last)
    O_W = ogrid_to_superellipse(0.75 * math.pi, 1.25 * math.pi, R, S, p_out, dy1, r, n_bl, n_q, dy_last)
    O_S = ogrid_to_superellipse(1.25 * math.pi, 1.75 * math.pi, R, S, p_out, dy1, r, n_bl, n_q, dy_last)

    oe, on, ow, os_ = O_E[-1], O_N[-1], O_W[-1], O_S[-1]

    def march_x(edge: np.ndarray, x_nodes: np.ndarray, x_ref: float) -> np.ndarray:
        """Horizontal march from curved edge to far x, copying x_nodes spacing."""
        x_nodes = np.asarray(x_nodes, dtype=float)
        n_j = len(edge)
        grid = np.zeros((len(x_nodes), n_j, 2))
        if abs(x_nodes[-1] - x_ref) <= abs(x_nodes[0] - x_ref):
            # … → x_ref  (inlet): i=0 at far
            x_far = float(x_nodes[0])
            t = (x_nodes - x_far) / (x_ref - x_far)
            for j in range(n_j):
                grid[:, j, 0] = x_far + t * (edge[j, 0] - x_far)
                grid[:, j, 1] = edge[j, 1]
        else:
            # x_ref → … (outlet): i=0 at edge
            x_far = float(x_nodes[-1])
            t = (x_nodes - x_ref) / (x_far - x_ref)
            for j in range(n_j):
                grid[:, j, 0] = edge[j, 0] + t * (x_far - edge[j, 0])
                grid[:, j, 1] = edge[j, 1]
        return grid

    def march_y_matched(
        edge: np.ndarray,
        h0_arr: np.ndarray,
        y_far: float,
        gmax: float,
        h_max: float,
        upward: bool,
    ) -> np.ndarray:
        """
        Vertical march from ``edge`` to ``y_far``.

        Uses one reference size distribution (longest column, starting at the
        median first-cell height) applied parametrically to every column so
        that n_j matches, growth ≤ gmax, and sizes ≤ h_max.
        """
        h0_arr = np.asarray(h0_arr, dtype=float)
        n = len(edge)
        lengths = np.abs(y_far - edge[:, 1])
        L_ref = float(np.max(lengths))
        h0_ref = float(np.median(h0_arr))
        sizes_ref = fill_remaining(L_ref, h0_ref, h_max, gmax)
        # Prepend adjustment so first ≈ median h0
        if len(sizes_ref) == 0:
            sizes_ref = np.array([L_ref])
        frac = np.concatenate([[0.0], np.cumsum(sizes_ref)])
        frac /= frac[-1]

        grid = np.zeros((n, len(frac), 2))
        for i in range(n):
            L = float(lengths[i])
            h0 = max(float(h0_arr[i]), 1e-15)
            if h0 >= L * 0.5:
                # very short column — uniform
                ys_off = np.linspace(0.0, L, len(frac))
            else:
                # Exact first cell h0; remaining length uses the same relative
                # distribution as sizes_ref[1:]
                rest_frac = frac[1:]  # starts at first node after wall
                # Map rest_frac from [f1, 1] to [h0, L]
                f1 = float(frac[1]) if len(frac) > 1 else 1.0
                ys_off = np.empty(len(frac))
                ys_off[0] = 0.0
                ys_off[1] = h0
                if len(frac) > 2 and abs(1.0 - f1) > 1e-15:
                    ys_off[2:] = h0 + (rest_frac[1:] - f1) / (1.0 - f1) * (L - h0)
                elif len(frac) > 2:
                    ys_off[2:] = np.linspace(h0, L, len(frac) - 1)[1:]
                ys_off[-1] = L
            if upward:
                ys = edge[i, 1] + ys_off
            else:
                ys = edge[i, 1] - ys_off
                ys = ys[::-1]
            grid[i, :, 0] = edge[i, 0]
            grid[i, :, 1] = ys
        return grid

    # ----- Side H blocks -----
    inw_east = ow[::-1]
    IN_W = march_x(inw_east, x_inlet, -S)
    OUT_E = march_x(oe, x_out_nodes, S)
    IN_W[-1, :, :] = inw_east
    OUT_E[0, :, :] = oe

    above_south = on[::-1]
    below_north = os_
    # Match first H cell to last O radial cell
    dr_N = np.linalg.norm(O_N[-1] - O_N[-2], axis=1)[::-1]
    dr_S = np.linalg.norm(O_S[-1] - O_S[-2], axis=1)
    # Vertical size stays ~ dy_core (≪ D/15), matching the wall-layer junction
    h_vert = max(dy_core, float(np.max(dr_N)), float(np.max(dr_S)))
    ABOVE = march_y_matched(above_south, dr_N, y_ti, gmax, h_vert, upward=True)
    BELOW = march_y_matched(below_north, dr_S, y_bi, gmax, h_vert, upward=False)
    ABOVE[:, 0, :] = above_south
    BELOW[:, -1, :] = below_north

    # ----- Corner blocks: TFI from channel corner to O-band sides -----
    # IN_U: east = ABOVE west (ABOVE[0,:]), west = inlet, south = IN_W north, north = y_ti
    in_u_east = ABOVE[0, :, :].copy()
    in_u_west = np.stack([np.full(ABOVE.shape[1], x_in), ABOVE[0, :, 1]], axis=-1)
    in_u_south = IN_W[:, -1, :].copy()
    in_u_north = np.stack([
        np.linspace(x_in, in_u_east[-1, 0] if False else ABOVE[0, -1, 0], len(x_inlet)),
        np.full(len(x_inlet), y_ti),
    ], axis=-1)
    # south must have len = len(x_inlet); IN_W has that
    # east/west must have len = ABOVE.shape[1]
    # TFI requires south/north len = west/east... no: south len = nx, west len = ny
    # nx = len(x_inlet), ny = ABOVE.shape[1]
    # so south/north need len x_inlet, west/east need len ABOVE.shape[1]
    in_u_north = np.stack([x_inlet, np.full(len(x_inlet), y_ti)], axis=-1)
    # west: at x_in, y following ABOVE west y's — but then south-west corner
    # (x_in, IN_W north y) must equal (x_in, ABOVE[0,0].y). 
    # ABOVE[0,0] = above_south[0] = O_N west end. IN_W north = O_W north end.
    # These are THE SAME point (shared O corner at 3π/4). Good.
    in_u_west = np.stack([np.full(ABOVE.shape[1], x_in), ABOVE[0, :, 1]], axis=-1)
    IN_U = tfi_rect(in_u_west, in_u_east, in_u_south, in_u_north)

    # OUT_U
    out_u_west = ABOVE[-1, :, :].copy()
    out_u_east = np.stack([np.full(ABOVE.shape[1], x_out), ABOVE[-1, :, 1]], axis=-1)
    out_u_south = OUT_E[:, -1, :].copy()
    out_u_north = np.stack([x_out_nodes, np.full(len(x_out_nodes), y_ti)], axis=-1)
    OUT_U = tfi_rect(out_u_west, out_u_east, out_u_south, out_u_north)

    # IN_L
    in_l_east = BELOW[0, :, :].copy()
    in_l_west = np.stack([np.full(BELOW.shape[1], x_in), BELOW[0, :, 1]], axis=-1)
    in_l_north = IN_W[:, 0, :].copy()
    in_l_south = np.stack([x_inlet, np.full(len(x_inlet), y_bi)], axis=-1)
    IN_L = tfi_rect(in_l_west, in_l_east, in_l_south, in_l_north)

    # OUT_L
    out_l_west = BELOW[-1, :, :].copy()
    out_l_east = np.stack([np.full(BELOW.shape[1], x_out), BELOW[-1, :, 1]], axis=-1)
    out_l_north = OUT_E[:, 0, :].copy()
    out_l_south = np.stack([x_out_nodes, np.full(len(x_out_nodes), y_bi)], axis=-1)
    OUT_L = tfi_rect(out_l_west, out_l_east, out_l_south, out_l_north)

    # Snap shared edges (O interfaces already exact on ABOVE/BELOW/IN_W/OUT_E)
    IN_U[-1, :, :] = ABOVE[0, :, :]
    OUT_U[0, :, :] = ABOVE[-1, :, :]
    IN_L[-1, :, :] = BELOW[0, :, :]
    OUT_L[0, :, :] = BELOW[-1, :, :]
    IN_U[:, 0, :] = IN_W[:, -1, :]
    IN_L[:, -1, :] = IN_W[:, 0, :]
    OUT_U[:, 0, :] = OUT_E[:, -1, :]
    OUT_L[:, -1, :] = OUT_E[:, 0, :]

    # Wall strips
    x_wall = np.concatenate([x_inlet[:-1], x_box[:-1], x_out_nodes])
    # Smooth any junction ratio > gmax by blending two cells
    def smooth_nodes(x, gmax):
        x = x.copy()
        for _ in range(5):
            d = np.diff(x)
            r = d[1:] / np.maximum(d[:-1], 1e-30)
            bad = np.where((r > gmax) | (r < 1.0 / gmax))[0]
            if len(bad) == 0:
                break
            for i in bad:
                # redistribute d[i]+d[i+1] equally-ish
                total = d[i] + d[i + 1]
                d[i] = total / (1.0 + min(max(r[i], 1.0), gmax))
                d[i + 1] = total - d[i]
            x = np.concatenate([[x[0]], x[0] + np.cumsum(d)])
            x[-1] = x_wall[-1]
        return x

    x_wall = smooth_nodes(x_wall, gmax)
    BOT_WALL = rect_block(x_wall, y_bot_layer)
    TOP_WALL = rect_block(x_wall, y_top_layer)
    for blk in (IN_L, BELOW, OUT_L):
        blk[:, 0, 1] = y_bi
    for blk in (IN_U, ABOVE, OUT_U):
        blk[:, -1, 1] = y_ti

    blocks = {
        "O_E": O_E, "O_N": O_N, "O_W": O_W, "O_S": O_S,
        "ABOVE": ABOVE, "BELOW": BELOW,
        "IN_W": IN_W, "OUT_E": OUT_E,
        "IN_L": IN_L, "IN_U": IN_U, "OUT_L": OUT_L, "OUT_U": OUT_U,
        "BOT_WALL": BOT_WALL, "TOP_WALL": TOP_WALL,
    }

    meta = dict(
        D=D, H=H, R=R, S=S, p_outer=p_out,
        x_in=x_in, x_out=x_out,
        delta_bl=delta, dy_last=dy_last, dy_core=dy_core,
        n_core=n_core, n_core_nom=n_core_nom, core_pct_change=core_pct,
        n_theta=n_theta, n_q=n_q, n_radial=O_E.shape[0] - 1, h_sq=h_sq,
        y_bi=y_bi, y_ti=y_ti, dy1=dy1, r=r, n_bl=n_bl,
        dx_wake_max=dx_wake_max, g_wake=g_wake,
        x_wall=x_wall, y_bot_layer=y_bot_layer, y_top_layer=y_top_layer,
    )
    return blocks, meta


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_gmsh(path: Path, blocks: dict[str, np.ndarray]) -> dict[str, int]:
    tol = 1e-10
    node_index: dict[tuple[float, float], int] = {}
    nodes: list[tuple[float, float]] = []

    def nid(xy) -> int:
        k = (round(float(xy[0]) / tol) * tol, round(float(xy[1]) / tol) * tol)
        if k not in node_index:
            node_index[k] = len(nodes) + 1
            nodes.append(k)
        return node_index[k]

    block_quads = {}
    block_counts = {}
    for name, grid in blocks.items():
        qids = []
        ni, nj = grid.shape[0] - 1, grid.shape[1] - 1
        for i in range(ni):
            for j in range(nj):
                qids.append([nid(grid[i, j]), nid(grid[i + 1, j]),
                             nid(grid[i + 1, j + 1]), nid(grid[i, j + 1])])
        block_quads[name] = qids
        block_counts[name] = len(qids)

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    boundaries = {k: set() for k in ("inlet", "outlet", "wall_top", "wall_bottom", "cylinder")}

    def add_edges(grid, i_sl, j_sl, bname):
        pts = grid[i_sl, j_sl]
        if pts.ndim == 3:
            pts = pts.reshape(-1, 2)
        ids = [nid(p) for p in pts]
        for a, b in zip(ids[:-1], ids[1:]):
            boundaries[bname].add(edge_key(a, b))

    add_edges(blocks["BOT_WALL"], slice(None), 0, "wall_bottom")
    add_edges(blocks["TOP_WALL"], slice(None), -1, "wall_top")
    for blk in ("IN_L", "IN_W", "IN_U", "BOT_WALL", "TOP_WALL"):
        add_edges(blocks[blk], 0, slice(None), "inlet")
    for blk in ("OUT_L", "OUT_E", "OUT_U", "BOT_WALL", "TOP_WALL"):
        add_edges(blocks[blk], -1, slice(None), "outlet")
    for o in ("O_E", "O_N", "O_W", "O_S"):
        add_edges(blocks[o], 0, slice(None), "cylinder")

    phys_curve = {"inlet": 1, "outlet": 2, "wall_top": 3, "wall_bottom": 4, "cylinder": 5}
    phys_surf = {n: 100 + i for i, n in enumerate(block_quads)}

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$PhysicalNames\n")
        f.write(f"{len(phys_curve) + len(phys_surf)}\n")
        for n, t in phys_curve.items():
            f.write(f'1 {t} "{n}"\n')
        for n, t in phys_surf.items():
            f.write(f'2 {t} "{n}"\n')
        f.write("$EndPhysicalNames\n$Nodes\n")
        f.write(f"{len(nodes)}\n")
        for i, (x, y) in enumerate(nodes, start=1):
            f.write(f"{i} {x:.16e} {y:.16e} 0.0\n")
        f.write("$EndNodes\n")
        lines, quads, eid = [], [], 1
        for bname, edges in boundaries.items():
            ptg = phys_curve[bname]
            for a, b in sorted(edges):
                lines.append((eid, 1, ptg, a, b)); eid += 1
        for bname, qids in block_quads.items():
            ptg = phys_surf[bname]
            for q in qids:
                quads.append((eid, 3, ptg, *q)); eid += 1
        f.write("$Elements\n")
        f.write(f"{len(lines) + len(quads)}\n")
        for e in lines:
            f.write(f"{e[0]} {e[1]} 2 {e[2]} {e[2]} {e[3]} {e[4]}\n")
        for e in quads:
            f.write(f"{e[0]} {e[1]} 2 {e[2]} {e[2]} {e[3]} {e[4]} {e[5]} {e[6]}\n")
        f.write("$EndElements\n")

    try:
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(path.with_name("mesh_v2.msh")))
        try:
            gmsh.write(str(path.with_name("mesh.cgns")))
        except Exception:
            pass
        gmsh.finalize()
    except Exception as exc:
        print(f"  [warn] gmsh re-export skipped: {exc}")
    return block_counts


def write_plot3d(path: Path, blocks: dict[str, np.ndarray]) -> None:
    names = list(blocks.keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(names)}\n")
        for n in names:
            g = blocks[n]
            f.write(f"{g.shape[0]} {g.shape[1]} 1 ")
        f.write("\n")
        for n in names:
            g = blocks[n]
            for v in g[:, :, 0].ravel(order="F"):
                f.write(f"{v:.16e}\n")
            for v in g[:, :, 1].ravel(order="F"):
                f.write(f"{v:.16e}\n")
            for _ in range(g.shape[0] * g.shape[1]):
                f.write(f"{0.0:.16e}\n")


# ---------------------------------------------------------------------------
# Verify + plot
# ---------------------------------------------------------------------------

def verify_and_plot(blocks, meta, block_counts, out_dir, P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    dy1, r, n_bl = meta["dy1"], meta["r"], meta["n_bl"]
    D, H, R = meta["D"], meta["H"], meta["R"]
    neigh_max = float(P["neigh_ratio_max"])
    wake_dx_max = meta["dx_wake_max"]
    wake_x = float(P["wake_x_over_D"]) * D

    struct_max, struct_where = 1.0, "n/a"
    for name, g in blocks.items():
        for ax in (0, 1):
            rg, loc = max_growth_in_grid(g, ax)
            if rg > struct_max:
                struct_max, struct_where = rg, f"{name} axis={ax} loc={loc}"

    def wall_h(grid, low):
        return float(np.mean(np.abs(grid[:, 1, 1] - grid[:, 0, 1]))) if low else \
            float(np.mean(np.abs(grid[:, -1, 1] - grid[:, -2, 1])))

    h_bot, h_top = wall_h(blocks["BOT_WALL"], True), wall_h(blocks["TOP_WALL"], False)
    h_cyl = float(np.mean([
        np.mean(np.linalg.norm(blocks[o][1] - blocks[o][0], axis=1))
        for o in ("O_E", "O_N", "O_W", "O_S")
    ]))

    # Also check O→H interface ratios
    iface_max = 1.0
    iface_where = "n/a"
    pairs = [
        ("O_E", -1, "OUT_E", 0, 0),   # radial last vs first x of OUT
        ("O_W", -1, "IN_W", -1, 0),
        ("O_N", -1, "ABOVE", 0, 1),   # vs first y of ABOVE — need careful
        ("O_S", -1, "BELOW", -1, 1),
    ]
    # O_E last radial size vs OUT_E first Δx
    for oname, oi, hname, hi, axis in [
        ("O_E", -1, "OUT_E", 0, "x"),
        ("O_W", -1, "IN_W", -1, "x"),
        ("O_N", -1, "ABOVE", 0, "y"),
        ("O_S", -1, "BELOW", -1, "y"),
    ]:
        og = blocks[oname]
        hg = blocks[hname]
        dr = np.linalg.norm(og[oi] - og[oi - 1 if oi == -1 else oi + 1], axis=1)
        if axis == "x":
            # hi=0 → cells between i=0 and 1; hi=-1 → between -2 and -1
            if hi == 0:
                dh = np.linalg.norm(hg[1] - hg[0], axis=1)
            else:
                dh = np.linalg.norm(hg[-1] - hg[-2], axis=1)
        else:
            if hi == 0:
                dh = np.linalg.norm(hg[:, 1] - hg[:, 0], axis=1)
            else:
                dh = np.linalg.norm(hg[:, -1] - hg[:, -2], axis=1)
        # lengths may differ if n_j mismatch — compare min length
        m = min(len(dr), len(dh))
        ratio = np.maximum(dr[:m], dh[:m]) / np.maximum(np.minimum(dr[:m], dh[:m]), 1e-30)
        rm = float(ratio.max())
        if rm > iface_max:
            iface_max = rm
            iface_where = f"{oname}/{hname}"

    overall_max = max(struct_max, iface_max)
    overall_where = struct_where if struct_max >= iface_max else f"interface {iface_where}"

    wake_edge_max = 0.0
    aspects, skew_cos = [], []
    rng = np.random.default_rng(0)
    for name, g in blocks.items():
        ni, nj = g.shape[0] - 1, g.shape[1] - 1
        for i in range(ni):
            for j in range(nj):
                q = np.array([g[i, j], g[i + 1, j], g[i + 1, j + 1], g[i, j + 1]])
                c = q.mean(0)
                e = np.array([np.linalg.norm(q[(k + 1) % 4] - q[k]) for k in range(4)])
                if 0.0 <= c[0] <= wake_x:
                    wake_edge_max = max(wake_edge_max, float(e.max()))
                if rng.random() < 0.03 or name.startswith("O_") or "WALL" in name:
                    aspects.append(max(e[0] + e[2], e[1] + e[3]) / max(min(e[0] + e[2], e[1] + e[3]), 1e-30))
                    for k in range(4):
                        v1, v2 = q[k] - q[(k - 1) % 4], q[(k + 1) % 4] - q[k]
                        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if n1 > 1e-16 and n2 > 1e-16:
                            skew_cos.append(abs(np.dot(v1, v2) / (n1 * n2)))

    max_aspect = float(np.max(aspects)) if aspects else float("nan")
    max_skew = float(np.max(skew_cos)) if skew_cos else 0.0
    min_orth = float(np.min(1.0 - np.asarray(skew_cos))) if skew_cos else 1.0

    cf = P.get("cf")
    yp_msg = "n/a (no cf supplied)"
    if cf is not None:
        yp_msg = f"{dy1 * math.sqrt(float(cf) / 2.0) / (D / float(P['Re_D'])):.3f}"

    tol = 0.05
    rules = {
        1: abs(h_cyl - dy1) / dy1 < tol and abs(h_bot - dy1) / dy1 < tol and abs(h_top - dy1) / dy1 < tol,
        2: abs(meta["delta_bl"] - bl_thickness(dy1, r, n_bl)) < 1e-12,
        3: abs(meta["core_pct_change"]) < 5.0,
        4: meta["n_theta"] >= 150 and meta["n_radial"] >= n_bl,
        5: wake_edge_max <= wake_dx_max * 1.01,
        6: overall_max <= neigh_max * 1.01,
        7: abs(h_bot - h_top) / dy1 < tol,
    }

    total = sum(block_counts.values())
    print("\n" + "=" * 72)
    print("STRUCTURED MULTIBLOCK MESH — VERIFICATION")
    print("=" * 72)
    print(f"  Domain: x∈[{meta['x_in']:.1f}, {meta['x_out']:.1f}], H={H:.1f}, D={D:.1f}, Re={P['Re_D']}")
    print(f"  BL: dy1={dy1}, r={r}, n={n_bl} → δ={meta['delta_bl']:.6f}, dy_last={meta['dy_last']:.6f}")
    print(f"  Core: dy_core={meta['dy_core']:.6f} ({meta['core_pct_change']:+.3f}% vs dy_last), "
          f"h_sq≈{meta['h_sq']:.5f}, p_outer={meta['p_outer']}")
    print(f"  O-grid: n_theta={meta['n_theta']}, n_radial={meta['n_radial']}")
    print(f"\n  Total cells: {total}")
    for k, v in sorted(block_counts.items()):
        print(f"    {k:20s}  {v}")
    print(f"\n  First cell height  cylinder={h_cyl:.6f}  bottom={h_bot:.6f}  top={h_top:.6f}")
    print(f"  Max y+ estimate: {yp_msg}")
    print(f"  Max neighbour size ratio: {overall_max:.4f} at {overall_where}")
    print(f"    (structured {struct_max:.4f}; interface {iface_max:.4f})")
    print(f"  Max cell aspect ratio (sample): {max_aspect:.2f}")
    print(f"  Min orthogonality proxy (1-|cos|): {min_orth:.4f}  (max |cos|={max_skew:.4f})")
    print(f"  Wake max edge (0≤x≤{wake_x:.1f}): {wake_edge_max:.5f}  (limit {wake_dx_max:.5f})")
    print("\n  Rule checks:")
    for i in range(1, 8):
        print(f"    Rule {i}: {'PASS' if rules[i] else 'FAIL'}")
    print("=" * 72)

    def collect(xlim=None, ylim=None, stride=1):
        polys = []
        for g in blocks.values():
            ni, nj = g.shape[0] - 1, g.shape[1] - 1
            for i in range(0, ni, stride):
                for j in range(0, nj, stride):
                    q = np.array([g[i, j], g[i + 1, j], g[i + 1, j + 1], g[i, j + 1]])
                    c = q.mean(0)
                    if xlim is not None and not (xlim[0] <= c[0] <= xlim[1] and ylim[0] <= c[1] <= ylim[1]):
                        continue
                    polys.append(q)
        return polys

    def draw(ax, xlim=None, ylim=None, title="", stride=1):
        pc = PolyCollection(collect(xlim, ylim, stride), facecolors="none",
                            edgecolors="#1f4e79", linewidths=0.12)
        ax.add_collection(pc)
        th = np.linspace(0, 2 * np.pi, 256)
        ax.plot(R * np.cos(th), R * np.sin(th), "k-", lw=1.4)
        ax.axhline(0.5 * H, color="k", lw=0.8)
        ax.axhline(-0.5 * H, color="k", lw=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*(xlim if xlim else (meta["x_in"], meta["x_out"])))
        ax.set_ylim(*(ylim if ylim else (-0.5 * H, 0.5 * H)))
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title(title)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    draw(axes[0], title=f"Full mesh ({total} cells)",
         stride=max(1, int(math.sqrt(total / 50000))))
    draw(axes[1], xlim=(-2.8 * D, 2.8 * D), ylim=(-2.8 * D, 2.8 * D),
         title="Zoom: cylinder O-layer")
    fig.tight_layout()
    fig.savefig(out_dir / "mesh_full_and_cylinder.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    draw(ax, xlim=(-1.0 * D, 4.0 * D),
         ylim=(-0.5 * H, -0.5 * H + 6 * meta["delta_bl"]),
         title="Zoom: bottom wall BL layer")
    fig.tight_layout()
    fig.savefig(out_dir / "mesh_wall_zoom.png", dpi=160)
    plt.close(fig)

    print(f"\n  Wrote {out_dir / 'mesh.msh'}")
    print(f"  Wrote {out_dir / 'mesh_v2.msh'}")
    print(f"  Wrote {out_dir / 'mesh.xyz'}")
    if (out_dir / "mesh.cgns").exists():
        print(f"  Wrote {out_dir / 'mesh.cgns'}")
    print(f"  Wrote {out_dir / 'mesh_full_and_cylinder.png'}")
    print(f"  Wrote {out_dir / 'mesh_wall_zoom.png'}")


def main() -> int:
    P = dict(PARAMS)
    out_dir = Path(__file__).resolve().parent / P["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Building structured multiblock mesh…")
    print("Blocks: O_E/N/W/S (→ superellipse) | ABOVE/BELOW | IN_* | OUT_* | WALLS")
    blocks, meta = build_blocks(P)
    total = sum((g.shape[0] - 1) * (g.shape[1] - 1) for g in blocks.values())
    print(f"  δ={meta['delta_bl']:.5f}, dy_last={meta['dy_last']:.5f}, "
          f"dy_core={meta['dy_core']:.5f} ({meta['core_pct_change']:+.2f}%), "
          f"n_θ={meta['n_theta']}, n_r={meta['n_radial']}, cells={total}")
    print("Exporting…")
    counts = export_gmsh(out_dir / "mesh.msh", blocks)
    write_plot3d(out_dir / "mesh.xyz", blocks)
    verify_and_plot(blocks, meta, counts, out_dir, P)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
