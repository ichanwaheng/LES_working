from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

EdgeKey = Tuple[int, int]

@dataclass
class UWMResult:
    nodes: np.ndarray
    n_weight_updates: int
    residual: float
    objective: float

def _undirected(a: int, b: int) -> EdgeKey:
    return (a, b) if a < b else (b, a)

def triangle_area(nodes: np.ndarray, conn: np.ndarray) -> float:
    """Calculate the 3D area of a single triangular element."""
    p0, p1, p2 = nodes[conn[0]], nodes[conn[1]], nodes[conn[2]]
    return float(0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0)))

def uwm_objective(nodes: np.ndarray, weights: Dict[EdgeKey, float]) -> float:
    """Objective helper function measuring current form network energy."""
    obj = 0.0
    for (i, j), w in weights.items():
        L = np.linalg.norm(nodes[i] - nodes[j])
        obj += w * (L ** 2)
    return float(obj)

def edge_side_force(N_pre: float, area: float, length: float) -> float:
    return N_pre * area / max(length, 1e-14)

def compute_edge_weights(
    nodes: np.ndarray,
    elements: np.ndarray,
    N_pre: float,
) -> Dict[EdgeKey, float]: 
    weights: Dict[EdgeKey, float] = {}
    for conn in elements:
        area = triangle_area(nodes, conn)
        if area < 1e-16:
            continue
        for a, b in ((conn[0], conn[1]), (conn[1], conn[2]), (conn[2], conn[0])):
            key = _undirected(int(a), int(b))
            L = float(np.linalg.norm(nodes[b] - nodes[a]))
            t = edge_side_force(N_pre, area, L)
            w = t / (2.0 * max(L, 1e-14))
            weights[key] = weights.get(key, 0.0) + w
    return weights  # Corrected Indentation: returns after loop processing completely

def assemble_force_density_system(
    nodes: np.ndarray,
    fixed: np.ndarray,
    weights: Dict[EdgeKey, float], # Fixed case-sensitivity type annotation
    f_ext: np.ndarray,
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    free = np.where(~fixed)[0]
    n_free = free.size
    if n_free == 0:
        return sparse.csr_matrix((0, 0)), np.zeros((0, 3)), free

    # Corrected Indentation: Shifted left so this executes properly
    idx = -np.ones(nodes.shape[0], dtype=int)
    idx[free] = np.arange(n_free)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = f_ext[free].copy()
    diag = np.zeros(n_free)

    for (i, j), W in weights.items(): # Fixed comma typo to dot operator
        coef = 2.0 * float(W)
        ia, ib = int(idx[i]), int(idx[j])
        if ia >= 0 and ib >= 0:
            diag[ia] += coef
            diag[ib] += coef
            rows.extend([ia, ib, ia, ib])
            cols.extend([ia, ib, ib, ia])
            data.extend([0.0, 0.0, -coef, -coef]) # Assembling standard off-diagonals
        elif ia >= 0 and ib < 0:
            diag[ia] += coef
            rhs[ia] += coef * nodes[j]
        elif ib >= 0 and ia < 0:
            diag[ib] += coef
            rhs[ib] += coef * nodes[i]

    for ia in range(n_free):
        rows.append(ia)
        cols.append(ia)
        if diag[ia] > 0.0:
            data.append(diag[ia])
        else:
            data.append(1.0)
            rhs[ia] = nodes[free[ia]]
            
    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_free, n_free)).tocsr()
    return A, rhs, free

def solve_force_density(
    nodes: np.ndarray,
    fixed: np.ndarray,
    weights: Dict[EdgeKey, float],
    f_ext: Optional[np.ndarray] = None,
) -> np.ndarray:
    f = np.zeros_like(nodes) if f_ext is None else np.asarray(f_ext, dtype=float)
    A, rhs, free = assemble_force_density_system(nodes, fixed, weights, f)
    x = nodes.copy()
    if free.size == 0:
        return x
    for d in range(3):
        x[free, d] = spsolve(A, rhs[:, d])
    x[fixed] = nodes[fixed]
    return x

def updated_weight_form_find(
    nodes: np.ndarray,
    elements: np.ndarray,
    fixed: np.ndarray,
    N_pre: float,
    f_ext: Optional[np.ndarray] = None,
    support_nodes: Optional[np.ndarray] = None,
    max_weight_updates: int = 25,
    tol: float = 1e-7,
    under_relaxation: float = 1.0,
) -> UWMResult:
    x = np.asarray(nodes, dtype=float).copy()
    x_support = (
        np.asarray(nodes, dtype=float).copy()
        if support_nodes is None
        else np.asarray(support_nodes, dtype=float).copy()
    )
    elements = np.asarray(elements, dtype=int)
    fixed = np.asarray(fixed, dtype=bool)
    alpha = float(np.clip(under_relaxation, 1e-3, 1.0))
    residual = np.inf
    obj = 0.0
    it = 0

    for it in range(1, max_weight_updates + 1):
        weights = compute_edge_weights(x, elements, N_pre)
        x_new = solve_force_density(x, fixed, weights, f_ext)
        x_new[fixed] = x_support[fixed]
        dx = x_new - x
        residual = float(np.linalg.norm(dx) / (np.linalg.norm(x) + 1e-12))
        x = (1.0 - alpha) * x + alpha * x_new
        x[fixed] = x_support[fixed]
        obj = uwm_objective(x, weights)
        if residual < tol:
            break

    return UWMResult(
        nodes=x, n_weight_updates=it, residual=residual, objective=obj
    )
