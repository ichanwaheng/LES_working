"""Validation metrics for laminar cylinder at Re=150 (NASA GRC Study #1).

Reference targets (experimental / literature cited by the NASA case):
- Mean drag coefficient  Cd ≈ 1.78  (Schlichting / NASA table)
- Strouhal number        St ≈ 0.179–0.182  (Roshko)

NASA Wind-US Study #1 also reports Cd ≈ 1.68–1.80 and St ≈ 0.15–0.18
depending on time-marching options:
https://www.grc.nasa.gov/www/wind/valid/lamcyl/Study1_files/Study1.html
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# Literature / NASA Study #1 reference band
CD_EXP = 1.78
ST_EXP_LO = 0.179
ST_EXP_HI = 0.182
ST_EXP_MID = 0.5 * (ST_EXP_LO + ST_EXP_HI)


@dataclass
class ValidationReport:
    Cd_mean: float
    Cd_exp: float
    Cd_error_pct: float
    St: float
    St_exp_lo: float
    St_exp_hi: float
    St_method: str
    Cl_amp: float
    n_periods: int
    t_start: float
    t_end: float
    within_Cd_tol: bool
    within_St_band: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _find_peaks(y: np.ndarray, min_prominence: float) -> np.ndarray:
    """Simple peak finder (no SciPy signal dependency)."""
    if y.size < 3:
        return np.array([], dtype=int)
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] > y[i + 1]:
            # local baseline ~ neighbours
            prom = y[i] - 0.5 * (y[i - 1] + y[i + 1])
            if prom >= min_prominence or y[i] > min_prominence:
                peaks.append(i)
    # enforce separation: keep strongest in sliding window of nearby peaks
    if not peaks:
        return np.array([], dtype=int)
    peaks = np.asarray(peaks, dtype=int)
    keep = [peaks[0]]
    for p in peaks[1:]:
        if p - keep[-1] < 3:
            if y[p] > y[keep[-1]]:
                keep[-1] = p
        else:
            keep.append(p)
    return np.asarray(keep, dtype=int)


def strouhal_from_lift(
    tUD: np.ndarray,
    Cl: np.ndarray,
    t_start: Optional[float] = None,
) -> tuple[float, int, str]:
    """Strouhal number from lift zero-crossing / peak period.

    ``St = f D / U`` with nondimensional time ``t* = t U / D``, so
    ``St = 1 / T*`` where ``T*`` is the mean period between Cl peaks.
    """
    tUD = np.asarray(tUD, dtype=float)
    Cl = np.asarray(Cl, dtype=float)
    if t_start is None:
        t_start = 0.6 * float(tUD[-1]) if tUD.size else 0.0
    mask = tUD >= t_start
    t = tUD[mask]
    cl = Cl[mask]
    if t.size < 20:
        return float("nan"), 0, "insufficient_data"

    amp = 0.5 * (float(np.max(cl)) - float(np.min(cl)))
    prom = max(0.05 * amp, 1e-4)
    peaks = _find_peaks(cl, min_prominence=prom)
    if peaks.size >= 2:
        periods = np.diff(t[peaks])
        # discard outliers (> 2× median)
        med = float(np.median(periods))
        good = periods[(periods > 0.4 * med) & (periods < 2.0 * med)]
        if good.size >= 1:
            T = float(np.mean(good))
            return 1.0 / T, int(good.size), "cl_peak_period"

    # FFT fallback on the analysis window
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return float("nan"), 0, "bad_dt"
    cl0 = cl - np.mean(cl)
    n = len(cl0)
    freqs = np.fft.rfftfreq(n, d=dt)
    amp_spec = np.abs(np.fft.rfft(cl0))
    # ignore zero frequency
    if freqs.size < 2:
        return float("nan"), 0, "fft_failed"
    amp_spec[0] = 0.0
    k = int(np.argmax(amp_spec))
    f = float(freqs[k])
    if f <= 0:
        return float("nan"), 0, "fft_failed"
    return f, 1, "cl_fft"


def mean_drag(tUD: np.ndarray, Cd: np.ndarray, t_start: Optional[float] = None) -> float:
    tUD = np.asarray(tUD, dtype=float)
    Cd = np.asarray(Cd, dtype=float)
    if t_start is None:
        t_start = 0.6 * float(tUD[-1]) if tUD.size else 0.0
    mask = tUD >= t_start
    if not np.any(mask):
        return float(np.mean(Cd)) if Cd.size else float("nan")
    return float(np.mean(Cd[mask]))


def validate_history(
    tUD: np.ndarray,
    Cd: np.ndarray,
    Cl: np.ndarray,
    t_start: Optional[float] = None,
    cd_tol_pct: float = 25.0,
) -> ValidationReport:
    """Compute Cd / St and compare to NASA Study #1 experimental targets."""
    tUD = np.asarray(tUD, dtype=float)
    Cd = np.asarray(Cd, dtype=float)
    Cl = np.asarray(Cl, dtype=float)
    if t_start is None:
        t_start = 0.6 * float(tUD[-1]) if tUD.size else 0.0

    Cd_m = mean_drag(tUD, Cd, t_start=t_start)
    St, n_per, method = strouhal_from_lift(tUD, Cl, t_start=t_start)
    mask = tUD >= t_start
    cl_w = Cl[mask] if np.any(mask) else Cl
    cl_amp = 0.5 * (float(np.max(cl_w)) - float(np.min(cl_w))) if cl_w.size else 0.0

    cd_err = 100.0 * (Cd_m - CD_EXP) / CD_EXP if np.isfinite(Cd_m) else float("nan")
    within_cd = bool(np.isfinite(cd_err) and abs(cd_err) <= cd_tol_pct)
    within_st = bool(np.isfinite(St) and (ST_EXP_LO * 0.7 <= St <= ST_EXP_HI * 1.3))

    return ValidationReport(
        Cd_mean=Cd_m,
        Cd_exp=CD_EXP,
        Cd_error_pct=cd_err,
        St=St,
        St_exp_lo=ST_EXP_LO,
        St_exp_hi=ST_EXP_HI,
        St_method=method,
        Cl_amp=cl_amp,
        n_periods=n_per,
        t_start=float(t_start),
        t_end=float(tUD[-1]) if tUD.size else 0.0,
        within_Cd_tol=within_cd,
        within_St_band=within_st,
    )


def write_report(report: ValidationReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


def format_report(report: ValidationReport) -> str:
    lines = [
        "=== Laminar cylinder Re=150 validation (NASA Study #1) ===",
        f"  analysis window: tU/D ∈ [{report.t_start:.2f}, {report.t_end:.2f}]",
        f"  Cd_mean = {report.Cd_mean:.4f}   exp = {report.Cd_exp:.3f}   "
        f"error = {report.Cd_error_pct:+.1f}%   "
        f"{'OK' if report.within_Cd_tol else 'CHECK'}",
        f"  St      = {report.St:.4f}   exp = [{report.St_exp_lo:.3f}, {report.St_exp_hi:.3f}]   "
        f"method={report.St_method}  n_periods={report.n_periods}   "
        f"{'OK' if report.within_St_band else 'CHECK'}",
        f"  Cl amp  = {report.Cl_amp:.4f}",
        "  refs: Roshko NACA 1191; Schlichting; "
        "https://www.grc.nasa.gov/www/wind/valid/lamcyl/Study1_files/Study1.html",
    ]
    return "\n".join(lines)
