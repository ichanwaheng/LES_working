


"""I/O helpers: config loading, NPZ / VTK export."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

_SCI = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def _coerce_numbers(obj: Any) -> Any:
    """Recursively cast numeric-looking strings (incl. scientific) to float/int."""
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        if _SCI.match(s) or re.match(r"^[+-]?\d+\.\d+$", s):
            return float(s)
        if re.match(r"^[+-]?\d+$", s):
            return int(s)
    return obj


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _coerce_numbers(raw)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_snapshot(
    out_dir: str | Path,
    step: int,
    time: float,
    membrane_nodes: np.ndarray,
    membrane_elements: np.ndarray,
    fluid_u: Optional[np.ndarray] = None,
    fluid_v: Optional[np.ndarray] = None,
    fluid_w: Optional[np.ndarray] = None,
    fluid_p: Optional[np.ndarray] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a compressed NPZ snapshot."""
    out = ensure_dir(out_dir)
    path = out / f"snapshot_{step:06d}.npz"
    payload: Dict[str, Any] = {
        "time": np.array(time),
        "membrane_nodes": membrane_nodes,
        "membrane_elements": membrane_elements,
    }
    if fluid_u is not None:
        payload["u"] = fluid_u
    if fluid_v is not None:
        payload["v"] = fluid_v
    if fluid_w is not None:
        payload["w"] = fluid_w
    if fluid_p is not None:
        payload["p"] = fluid_p
    if meta:
        for k, v in meta.items():
            payload[f"meta_{k}"] = np.asarray(v)
    np.savez_compressed(path, **payload)
    return path


def write_membrane_vtk(
    path: str | Path,
    nodes: np.ndarray,
    elements: np.ndarray,
    point_data: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write a simple legacy VTK polydata for the membrane surface."""
    path = Path(path)
    n_pts = nodes.shape[0]
    n_cells = elements.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("tensile membrane\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n_pts} float\n")
        for p in nodes:
            f.write(f"{p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")
        f.write(f"POLYGONS {n_cells} {n_cells * 4}\n")
        for e in elements:
            f.write(f"3 {e[0]} {e[1]} {e[2]}\n")
        if point_data:
            first = True
            for name, arr in point_data.items():
                arr = np.asarray(arr)
                if first:
                    f.write(f"POINT_DATA {n_pts}\n")
                    first = False
                if arr.ndim == 1:
                    f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
                    for v in arr:
                        f.write(f"{float(v):.6e}\n")
                else:
                    f.write(f"VECTORS {name} float\n")
                    for v in arr:
                        f.write(f"{v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")


def save_history_csv(path: str | Path, history) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("time,max_disp,kinetic,cfl,residual\n")
        for i in range(len(history.time)):
            f.write(
                f"{history.time[i]:.6e},"
                f"{history.max_disp[i]:.6e},"
                f"{history.kinetic[i]:.6e},"
                f"{history.cfl[i]:.6e},"
                f"{history.residual[i]:.6e}\n"
            )

