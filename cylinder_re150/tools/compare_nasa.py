"""
Compare the staggered Griebel solver results with NASA GRC Wind-US Study #1
(https://www.grc.nasa.gov/www/wind/valid/lamcyl/Study1_files/Study1.html).

    python tools/compare_nasa.py [artifact_dir]

Reads the fields_*.npz histories listed in CASES (produced by
ns_staggered_euler.py), computes Cd, Cl, St (peak spacing and FFT, as on the
NASA page) and the pressure/viscous drag split, and writes
demo/nasa_comparison.png, demo/nasa_histories.png, demo/nasa_blockage.png
and nasa_comparison.json.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ns_staggered_euler as m

ROOT = HERE.parent
ART = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo"
ART.mkdir(parents=True, exist_ok=True)

# ---- NASA Study #1 (https://www.grc.nasa.gov/www/wind/valid/lamcyl/Study1_files/Study1.html)
NASA_EXP_CD = 1.78                 # Schlichting, cited by NASA
NASA_EXP_ST = (0.179, 0.182)       # Roshko, NACA 1191
NASA_RUNS = {                      # run: (Cd, St1 peaks, St2 FFT 2000, St3 FFT 3000)
    "A (1st order)":          (1.767, 0.161, 0.150, 0.167),
    "B (2nd order)":          (1.804, 0.164, 0.175, 0.167),
    "C (1st order + Newton)": (1.679, 0.177, 0.175, 0.183),
    "D (2nd order + Newton)": (1.715, 0.180, 0.175, 0.183),
}
# other incompressible, unbounded references (for context)
WILLIAMSON_ST = -3.3265 / 150 + 0.1816 + 1.6e-4 * 150   # Williamson (1988/1996) universal St–Re curve
HENDERSON_CD = 1.33                                      # Henderson (1995) 2-D DNS, Re = 150

CASES = [
    ("40 D × 10 D, 10 % blockage",  ROOT / "output_ns/fine_default/fields_central_fine.npz", ROOT / "mesh_output/staggered_fine_40x10.npz"),
    ("24 D × 12 D, 8.3 % blockage", ROOT / "output_ns/fine_24x12/fields_central_fine.npz",  ROOT / "mesh_output/staggered_fine.npz"),
    ("24 D × 24 D, 4.2 % blockage", ROOT / "output_ns/tall24/fields_central_run.npz",      ROOT / "mesh_output/staggered_tall24.npz"),
]


def peaks(t, y):
    i = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    return t[i], y[i]


def strouhal(t, Cl, frac):
    """St from the last `frac` of the Cl history: peak spacing and FFT."""
    sel = t >= t[-1] - frac * (t[-1] - t[0])
    tt, cc = t[sel], Cl[sel] - Cl[sel].mean()
    tp, _ = peaks(tt, cc)
    st_peaks = 1.0 / np.mean(np.diff(tp[-5:])) if len(tp) >= 5 else np.nan   # last four periods, like NASA St1
    # FFT on a uniform resample
    tu = np.linspace(tt[0], tt[-1], 1 << 14)
    cu = np.interp(tu, tt, cc) * np.hanning(tu.size)
    nfft = tu.size * 32                                     # zero-pad: interpolates the spectral peak
    f = np.fft.rfftfreq(nfft, tu[1] - tu[0]); a = np.abs(np.fft.rfft(cu, n=nfft))
    st_fft = f[np.argmax(a[1:]) + 1]
    return st_peaks, st_fft, tp


def analyse(label, fields, mesh):
    d = np.load(fields, allow_pickle=True)
    t, Cd, Cl = d["hist_t"], d["hist_Cd"], d["hist_Cl"]
    late = t >= 0.6 * t[-1]
    st_peaks, st_fft40, tp = strouhal(t, Cl, 0.4)
    _, st_fft60, _ = strouhal(t, Cl, 0.6)
    tpk, cpk = peaks(t[late], Cl[late])
    out = dict(
        case=label,
        t_end=float(t[-1]),
        Cd_mean=float(Cd[late].mean()),
        Cd_osc=float(0.5 * np.ptp(Cd[late][-int(0.5 * late.sum()):])),
        Cl_amp=float(0.5 * np.ptp(Cl[late])),
        St_peaks=float(st_peaks), St_fft40=float(st_fft40), St_fft60=float(st_fft60),
    )
    # pressure / viscous split at the final instant
    if mesh.exists():
        S = m.load_staggered(mesh)
        s = m.GriebelStaggeredNS(S, Re=150, scheme="central")
        s.load_state(fields)
        topo = s.topo; mk = topo["masks"]
        f = np.where(mk["cylinder"])[0]; n = S["face_normals"][f]; L = S["face_lengths"][f]; o = S["face_owner"][f]
        dd = 0.5 * topo["d_pn"][f]
        q = 0.5 * s.rho * s.U_inf ** 2 * S["D"]
        Fp_x = np.sum(s.p[o] * n[:, 0] * L)
        tx, ty = -n[:, 1], n[:, 0]
        ut = s.u[o] * tx + s.v[o] * ty
        Fs_x = np.sum(s.rho * s.nu * ut / dd * tx * L)
        out["Cd_p_final"] = float(Fp_x / q); out["Cd_v_final"] = float(Fs_x / q); out["Cd_final"] = float((Fp_x + Fs_x) / q)
        out["cells"] = int(topo["Nc"])
    return out, (t, Cd, Cl)


results, hists = [], []
for label, fields, mesh in CASES:
    if not fields.exists():
        print("skip", label); continue
    r, h = analyse(label, fields, mesh)
    results.append(r); hists.append((label, h))
    print(json.dumps(r, indent=1))

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
ax.axhline(NASA_EXP_CD, color="k", lw=1.5, ls="--", label="experiment cited by NASA (Schlichting) 1.78")
ax.axhline(HENDERSON_CD, color="grey", lw=1.2, ls=":", label="2-D DNS, unbounded (Henderson 1995) ≈ 1.33")
xs, ys, labels = [], [], []
for k, (run, (cd, *_)) in enumerate(NASA_RUNS.items()):
    xs.append(k); ys.append(cd); labels.append("Wind-US " + run.split(" ")[0])
ax.plot(xs, ys, "s", ms=9, color="#1F3A5F", label="NASA Wind-US runs A–D (M = 0.2, far field 200 D)")
for j, r in enumerate(results):
    x = len(xs) + j
    ax.plot(x, r["Cd_mean"], "o", ms=10, color="#B23A2A", label="this solver (incompressible, channel)" if j == 0 else None)
    ax.annotate(f'{r["Cd_mean"]:.2f}', (x, r["Cd_mean"]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color="#B23A2A")
    labels.append(r["case"].split(",")[0].replace(" D × ", "×").replace(" D", ""))
for k, y in zip(xs, ys):
    ax.annotate(f"{y:.3f}", (k, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color="#1F3A5F")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("mean Cd"); ax.set_ylim(1.2, 1.9); ax.grid(alpha=0.3); ax.set_title("Mean drag coefficient", loc="left")
ax.legend(fontsize=8, loc="lower left")

ax = axes[1]
ax.axhspan(*NASA_EXP_ST, color="k", alpha=0.15, label="experiment cited by NASA (Roshko) 0.179–0.182")
ax.axhline(WILLIAMSON_ST, color="grey", lw=1.2, ls=":", label=f"Williamson St–Re curve, unbounded ≈ {WILLIAMSON_ST:.3f}")
labels = []
for k, (run, (cd, st1, st2, st3)) in enumerate(NASA_RUNS.items()):
    ax.plot([k, k, k], [st1, st2, st3], "s", ms=7, color="#1F3A5F", alpha=0.8, label="NASA Wind-US A–D (St1 peaks, St2/St3 FFT)" if k == 0 else None)
    labels.append("Wind-US " + run.split(" ")[0])
for j, r in enumerate(results):
    x = len(NASA_RUNS) + j
    ax.plot([x, x, x], [r["St_peaks"], r["St_fft40"], r["St_fft60"]], "o", ms=8, color="#B23A2A", label="this solver (peaks, FFT 40 %, FFT 60 %)" if j == 0 else None)
    ax.annotate(f'{r["St_fft40"]:.3f}', (x, r["St_fft40"]), textcoords="offset points", xytext=(10, 0), ha="left", fontsize=9, color="#B23A2A")
    labels.append(r["case"].split(",")[0].replace(" D × ", "×").replace(" D", ""))
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Strouhal number"); ax.set_ylim(0.14, 0.23); ax.grid(alpha=0.3); ax.set_title("Strouhal number", loc="left")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
for p in {ART / "nasa_comparison.png", ROOT / "demo" / "nasa_comparison.png"}:
    fig.savefig(p, dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ Cl / Cd histories overlay
fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
for (label, (t, Cd, Cl)), col in zip(hists, ["#777777", "#B23A2A", "#1F3A5F"]):
    axes[0].plot(t, Cd, color=col, lw=1, label=label)
    axes[1].plot(t, Cl, color=col, lw=1, label=label)
axes[0].axhline(NASA_EXP_CD, color="k", ls="--", lw=1, label="NASA exp. 1.78"); axes[0].axhspan(1.679, 1.804, color="#1F3A5F", alpha=0.12, label="Wind-US A–D 1.68–1.80")
axes[0].set_ylabel("Cd"); axes[0].set_ylim(1.1, 1.9); axes[0].legend(fontsize=8, ncol=3, loc="lower right"); axes[0].grid(alpha=0.3)
axes[1].set_ylabel("Cl"); axes[1].set_xlabel("t U / D"); axes[1].grid(alpha=0.3); axes[1].set_xlim(0, 120); axes[1].set_ylim(-0.9, 0.9)
fig.tight_layout()
for p in {ART / "nasa_histories.png", ROOT / "demo" / "nasa_histories.png"}:
    fig.savefig(p, dpi=150)
json.dump(dict(nasa=dict(exp_Cd=NASA_EXP_CD, exp_St=NASA_EXP_ST, runs=NASA_RUNS), ours=results), open(ART / "nasa_comparison.json", "w"), indent=1)

# ------------------------------------------------------------------ blockage trend (same 24 D length: β = 1/12 and 1/24)
beta = {"40 D × 10 D, 10 % blockage": 0.10, "24 D × 12 D, 8.3 % blockage": 1 / 12, "24 D × 24 D, 4.2 % blockage": 1 / 24}
same_len = [r for r in results if r["case"].startswith("24 D")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ext = {}
for ax, key, ylab, ref, refl in [
    (axes[0], "Cd_mean", "mean Cd", NASA_EXP_CD, "NASA exp. 1.78"),
    (axes[1], "St_fft40", "St", np.mean(NASA_EXP_ST), "NASA exp. 0.179–0.182"),
]:
    b = np.array([beta[r["case"]] for r in results]); y = np.array([r[key] for r in results])
    ax.plot(b, y, "o", ms=9, color="#B23A2A", label="this solver")
    bb = np.array([beta[r["case"]] for r in same_len]); yy = np.array([r[key] for r in same_len])
    slope, icpt = np.polyfit(bb, yy, 1); ext[key] = icpt
    xs = np.linspace(0, 0.11, 10); ax.plot(xs, slope * xs + icpt, "--", color="#B23A2A", lw=1, label=f"linear in β (24 D-long cases) → β = 0: {icpt:.3f}")
    ax.plot(0, icpt, "x", ms=10, color="#B23A2A", mew=2)
    if key == "Cd_mean":
        ax.axhline(ref, color="k", ls="--", lw=1.2, label=refl); ax.axhspan(1.679, 1.804, color="#1F3A5F", alpha=0.12, label="Wind-US A–D")
        ax.axhline(HENDERSON_CD, color="grey", ls=":", lw=1.2, label="unbounded 2-D DNS ≈ 1.33")
    else:
        ax.axhspan(*NASA_EXP_ST, color="k", alpha=0.15, label=refl); ax.axhspan(0.150, 0.183, color="#1F3A5F", alpha=0.12, label="Wind-US A–D (all St estimates)")
        ax.axhline(WILLIAMSON_ST, color="grey", ls=":", lw=1.2, label="Williamson curve ≈ 0.183")
    ax.set_xlabel("blockage β = D / H"); ax.set_ylabel(ylab); ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_xlim(-0.005, 0.11)
axes[0].set_title("Drag vs channel blockage", loc="left"); axes[1].set_title("Strouhal vs channel blockage", loc="left")
fig.tight_layout()
for p in {ART / "nasa_blockage.png", ROOT / "demo" / "nasa_blockage.png"}:
    fig.savefig(p, dpi=150)
print("extrapolated to beta=0:", ext)
