#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evaluate.py

Evaluation (example) script you can use with:
  - generate.py (dataset)
  - train_cffs_8cff.py (replicas)

It reports:
  * per-CFF ensemble mean/std and bias vs truth
  * histograms of (fit - truth) for each CFF component
  * pseudodata vs phi plots for:
        XS, BSA, BCA, TSA, DSA
    including:
      - "ensemble mean curve" (mean of curves across replicas)
      - optional 1sig band across replicas
      - truth curve
      - a diagnostic "curve from mean(CFFs)" (forward model on mean CFF vector)

Important diagnostic:
---------------------
If the problem is under-constrained, replicas may produce *very different CFFs*
while still fitting all observables nearly perfectly. In that case:
  - the ensemble mean curve will match the pseudodata,
  - but the curve computed from mean(CFFs) may NOT,
because the forward map CFF->observable is nonlinear.
All the stuff you need should be in the same dir.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from cross_section_script import compute_observables


# ============================================================
# CONFIG (edit here)
# ============================================================

TAG = "v_1"
DATA_NPZ = f"output/data/dataset_{TAG}.npz"
TRUTH_JSON = f"output/data/truth_{TAG}.json"

OUT_ROOT = "output_torch_8cff"
REPLICA_GLOB = os.path.join(OUT_ROOT, "replicas", "replica_*", "extracted_cffs_per_point.npz")

OUT_DIR = os.path.join(OUT_ROOT, "eval")

# Forward settings (must match generator/training)
K_BEAM = 5.75
USING_WW = True

PLOT_ENSEMBLE_BAND = True
PLOT_TRUTH_CURVE = True
PLOT_MEAN_CFF_CURVE = True

HIST_BINS = "auto"

# ============================================================
# END CONFIG
# ============================================================

CFF_NAMES = ["ReH", "ReE", "ReHtilde", "ReEtilde", "ImH", "ImE", "ImHtilde", "ImEtilde"]
OBS_NAMES = ["XS", "BSA", "BCA", "TSA", "DSA"]


def safe_mkdir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return json.load(open(path, "r"))


def load_replica_npz(glob_pat: str) -> Tuple[np.ndarray, List[str]]:
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise FileNotFoundError(f"No replicas matched: {glob_pat}")
    cffs_list = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        c = d["cffs"].astype(np.float64)
        if c.ndim != 2 or c.shape[1] != 8:
            raise ValueError(f"Unexpected cffs shape in {p}: {c.shape}")
        cffs_list.append(c)
    # Stack: (R,P,8)
    return np.stack(cffs_list, axis=0), paths


def extract_truth8(truth: Dict) -> np.ndarray:
    c = truth.get("km15_truth_cffs", {})
    keys = [
        "cff_real_h_km15", "cff_real_e_km15", "cff_real_ht_km15", "cff_real_et_km15",
        "cff_imag_h_km15", "cff_imag_e_km15", "cff_imag_ht_km15", "cff_imag_et_km15",
    ]
    return np.array([float(c[k]) for k in keys], dtype=np.float64)


def plot_hist(values: np.ndarray, truth: float, title: str, xlabel: str, outpath: str) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    mu = float(np.mean(v))
    sig = float(np.std(v, ddof=0))
    plt.figure()
    plt.hist(v, bins=HIST_BINS, edgecolor="black", alpha=0.75)
    plt.axvline(mu, label="mean")
    plt.axvline(mu - sig, linestyle=":", label=r"$\pm 1\sigma$")
    plt.axvline(mu + sig, linestyle=":")
    plt.axvline(float(truth), color="red", linestyle="--", linewidth=2, label="truth")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Replica count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return mu, sig


def _forward_one_point(phi_rad: np.ndarray, kin3: np.ndarray, cff8: np.ndarray) -> np.ndarray:
    t, xb, q2 = [float(x) for x in kin3.reshape(-1)]
    cff8 = np.asarray(cff8, dtype=float).reshape(8)
    cffs = {
        "re_h": float(cff8[0]), "re_e": float(cff8[1]), "re_ht": float(cff8[2]), "re_et": float(cff8[3]),
        "im_h": float(cff8[4]), "im_e": float(cff8[5]), "im_ht": float(cff8[6]), "im_et": float(cff8[7]),
    }
    obs = compute_observables(
        phi_rad=phi_rad,
        k_beam=float(K_BEAM),
        q_squared=float(q2),
        xb=float(xb),
        t=float(t),
        cffs=cffs,
        using_ww=bool(USING_WW),
    )
    return np.column_stack([obs["xs"], obs["bsa"], obs["bca"], obs["tsa"], obs["dsa"]]).astype(np.float64)


def plot_pseudodata(phi_deg: np.ndarray, y_obs: np.ndarray, y_err: np.ndarray,
                   y_curve_mean: np.ndarray, title: str, ylabel: str, outpath: str,
                   band_std: Optional[np.ndarray] = None, truth_curve: Optional[np.ndarray] = None,
                   mean_cff_curve: Optional[np.ndarray] = None) -> None:
    idx = np.argsort(phi_deg)
    x = phi_deg[idx]
    y = y_obs[idx]
    ye = y_err[idx]
    yc = y_curve_mean[idx]

    plt.figure()
    if np.any(ye > 0):
        plt.errorbar(x, y, yerr=ye, fmt="o", capsize=2, label="pseudodata")
    else:
        plt.plot(x, y, "o", label="pseudodata")

    plt.plot(x, yc, label="inferred (ensemble mean curve)")

    if band_std is not None:
        ys = band_std[idx]
        plt.fill_between(x, yc - ys, yc + ys, alpha=0.25, label=r"ensemble $\pm 1\sigma$")

    if truth_curve is not None:
        yt = truth_curve[idx]
        plt.plot(x, yt, linestyle="--", label="truth")

    if mean_cff_curve is not None:
        ym = mean_cff_curve[idx]
        plt.plot(x, ym, linestyle=":", linewidth=2, label="curve from mean(CFFs)")

    plt.xlabel(r"$\phi$ (deg)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    safe_mkdir(OUT_DIR)

    d = load_npz(DATA_NPZ)
    X = d["x"].astype(np.float64)
    y = d["y_central"].astype(np.float64)
    ysig = d["y_sigma"].astype(np.float64)

    if y.shape[1] != 5:
        raise ValueError(f"Expected y_central (N,5) for XS/BSA/BCA/TSA/DSA; got {y.shape}")

    # Currently we assume fixed-kinematics (P=1) for plotting
    kin3 = X[0, :3].astype(np.float64)
    phi = X[:, 3].astype(np.float64)
    phi_deg = np.degrees(phi)

    truth = load_json(TRUTH_JSON)
    truth8 = extract_truth8(truth)

    # Load replicas
    cffs_all, paths = load_replica_npz(REPLICA_GLOB)  # (R,P,8)
    R, P, _ = cffs_all.shape
    if P != 1:
        print(f"NOTE: Found P={P} points in extracted CFFs. This evaluator will plot point000 only.")

    cffs_rep = cffs_all[:, 0, :]  # (R,8)

    # CFF residual histograms
    print(f"Loaded {R} replicas from: {os.path.dirname(REPLICA_GLOB)}")

    summary = {}
    for j, name in enumerate(CFF_NAMES):
        diff = cffs_rep[:, j] - truth8[j]
        mu = float(np.mean(cffs_rep[:, j]))
        sig = float(np.std(cffs_rep[:, j], ddof=0))
        bias = mu - float(truth8[j])
        summary[name] = {"mean": mu, "std": sig, "truth": float(truth8[j]), "bias": bias}

        plot_hist(
            diff,
            0.0,
            title=f"CFF residuals fit-truth ({name})",
            xlabel="fit - truth",
            outpath=os.path.join(OUT_DIR, f"hist_resid_{name}.png"),
        )

    print("\nCFF summary (P=1):")
    for name in CFF_NAMES:
        s = summary[name]
        print(f"  {name:8s}: mean={s['mean']:.6g} std={s['std']:.6g} truth={s['truth']:.6g} bias={s['bias']:.6g}")

    # Forward all replica curves
    curves = []
    for r in range(R):
        curves.append(_forward_one_point(phi, kin3, cffs_rep[r]))
    curves = np.stack(curves, axis=0)  # (R,N,5)

    curve_mean = np.mean(curves, axis=0)  # (N,5) mean of curves
    curve_std = np.std(curves, axis=0, ddof=0) if PLOT_ENSEMBLE_BAND else None

    # Truth curve
    truth_curve = _forward_one_point(phi, kin3, truth8) if PLOT_TRUTH_CURVE else None

    # Curve from mean(CFFs)
    mean_cffs = np.mean(cffs_rep, axis=0)
    mean_cff_curve = _forward_one_point(phi, kin3, mean_cffs) if PLOT_MEAN_CFF_CURVE else None

    # Plots per observable
    ylabels = ["XS", "BSA", "BCA", "TSA", "DSA"]
    titles = [
        "Pseudodata XS vs $\phi$",
        "Pseudodata BSA vs $\phi$",
        "Pseudodata BCA vs $\phi$",
        "Pseudodata TSA vs $\phi$",
        "Pseudodata DSA vs $\phi$",
    ]

    for k in range(5):
        plot_pseudodata(
            phi_deg=phi_deg,
            y_obs=y[:, k],
            y_err=ysig[:, k],
            y_curve_mean=curve_mean[:, k],
            title=titles[k],
            ylabel=ylabels[k],
            outpath=os.path.join(OUT_DIR, f"fit_{OBS_NAMES[k]}.png"),
            band_std=None if curve_std is None else curve_std[:, k],
            truth_curve=None if truth_curve is None else truth_curve[:, k],
            mean_cff_curve=None if mean_cff_curve is None else mean_cff_curve[:, k],
        )

    print("\nWrote outputs to:", OUT_DIR)
    print("  - hist_resid_<CFF>.png")
    print("  - fit_XS.png, fit_BSA.png, fit_BCA.png, fit_TSA.png, fit_DSA.png")


if __name__ == "__main__":
    main()
