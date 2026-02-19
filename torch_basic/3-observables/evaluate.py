#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
After training you can evaluate the results by running this script.

Evaluation for the torch closure pipeline with 8 CFFs and 3 observables:
  - XS(phi)  : unpolarized cross section
  - BSA(phi) : beam spin asymmetry
  - BCA(phi) : beam charge asymmetry

It reads:
  <OUT_ROOT>/replicas/replica_*/extracted_cffs_per_point.npz
and (optionally) a truth JSON produced by the generator:
  <OUTPUT_DIR>/data/truth_<TAG>.json
and the dataset itself:
  <OUTPUT_DIR>/data/dataset_<TAG>.npz

It produces in:
  <OUT_ROOT>/eval/
    - cff_residuals_pointXXX.png (histograms: fit - truth)
    - cff_summary_pointXXX.json, cff_residuals_pointXXX.csv
    - xs_fit_pointXXX.png, bsa_fit_pointXXX.png, bca_fit_pointXXX.png

This script is intentionally no-CLI: edit the CONFIG block.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

from bkm10_torch_forward import BKM10Config, BKM10Forward


# ============================================================
# CONFIG (edit here)
# ============================================================

OUT_ROOT = "output_torch_8cff"

OUTPUT_DIR = "output"   # where dataset/truth live
TAG = "v_1"

DATA_NPZ = os.path.join(OUTPUT_DIR, "data", f"dataset_{TAG}.npz")
TRUTH_JSON = os.path.join(OUTPUT_DIR, "data", f"truth_{TAG}.json")

OUT_EVAL_DIR = os.path.join(OUT_ROOT, "eval")

# Evaluate all points (multi-kin) or a single point index
EVAL_ALL_POINTS = True
POINT_INDEX = 0

# Match tolerance for (t,xB,Q2)
KIN_ATOL = 1e-8

# Plot options
PLOT_ENSEMBLE_BAND = True   # propagate all replicas to show +/-1σ band
PLOT_TRUTH_CURVE = True

# If you want to see the *replica-sampled pseudodata*, set this to an integer
# replica index (0-based) and also set REPLICA_SEED to match training.
# By default we plot the dataset's y_central.
PLOT_REPLICA_PSEUDODATA = None
REPLICA_SEED = 2222

HIST_BINS = "auto"

# ============================================================
# END CONFIG
# ============================================================

CFF_NAMES = ["ReH","ReE","ReHtilde","ReEtilde","ImH","ImE","ImHtilde","ImEtilde"]
OBS_NAMES = ["XS","BSA","BCA"]


def safe_mkdir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ not found: {path}")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def km15_to_vec(d: Dict) -> np.ndarray:
    return np.array([
        float(d["cff_real_h_km15"]),
        float(d["cff_real_e_km15"]),
        float(d["cff_real_ht_km15"]),
        float(d["cff_real_et_km15"]),
        float(d["cff_imag_h_km15"]),
        float(d["cff_imag_e_km15"]),
        float(d["cff_imag_ht_km15"]),
        float(d["cff_imag_et_km15"]),
    ], dtype=float)


def build_truth_map(truth: Dict) -> Tuple[Optional[np.ndarray], Dict[Tuple[float,float,float], np.ndarray]]:
    """Return (single_truth, per_point_map[(t,xB,Q2)])."""
    single_truth = None
    if "km15_truth_cffs" in truth:
        single_truth = km15_to_vec(truth["km15_truth_cffs"])

    mp: Dict[Tuple[float,float,float], np.ndarray] = {}
    if "truth_cffs_per_point" in truth and isinstance(truth["truth_cffs_per_point"], list):
        for e in truth["truth_cffs_per_point"]:
            try:
                t = float(e["t"])
                xb = float(e["x_b"])
                q2 = float(e["q_squared"])
                v = km15_to_vec(e["km15_truth_cffs"])
                mp[(t, xb, q2)] = v
            except Exception:
                continue
    return single_truth, mp


def match_truth(kin: np.ndarray, single_truth: Optional[np.ndarray], mp: Dict[Tuple[float,float,float], np.ndarray]) -> np.ndarray:
    """For each row kin[i]=[t,xB,Q2], pick matching truth; fall back to single_truth."""
    kin = np.asarray(kin, dtype=float)
    out = np.zeros((kin.shape[0], 8), dtype=float)
    for i in range(kin.shape[0]):
        t, xb, q2 = kin[i].tolist()
        found = None
        if (t, xb, q2) in mp:
            found = mp[(t, xb, q2)]
        else:
            for (tt, xx, qq), v in mp.items():
                if abs(tt - t) <= KIN_ATOL and abs(xx - xb) <= KIN_ATOL and abs(qq - q2) <= KIN_ATOL:
                    found = v
                    break
        if found is None:
            if single_truth is None:
                raise KeyError("No matching per-point truth and no single km15_truth_cffs in truth json.")
            found = single_truth
        out[i] = found
    return out


def plot_residual_hist(resid: np.ndarray, outpath: str, title: str) -> None:
    resid = np.asarray(resid, dtype=float)
    plt.figure(figsize=(14, 6))
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        ax.hist(resid[:, i], bins=HIST_BINS, edgecolor="black", alpha=0.75)
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.set_title(CFF_NAMES[i])
        ax.set_xlabel("fit - truth")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def write_residual_csv(resid: np.ndarray, outpath: str, replica_paths: List[str]) -> None:
    resid = np.asarray(resid, dtype=float)
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["replica_npz"] + [f"d{name}" for name in CFF_NAMES])
        for rp, row in zip(replica_paths, resid):
            w.writerow([os.path.relpath(rp, start=os.path.dirname(outpath))] + [float(x) for x in row.tolist()])


def plot_pseudodata_with_curves(
    phi_deg: np.ndarray,
    y_obs: np.ndarray,
    y_err: np.ndarray,
    y_curve: np.ndarray,
    curve_label: str,
    title: str,
    ylabel: str,
    outpath: str,
    y_band_std: Optional[np.ndarray] = None,
    truth_curve: Optional[np.ndarray] = None,
) -> None:
    idx = np.argsort(phi_deg)
    x = phi_deg[idx]
    y = y_obs[idx]
    ye = y_err[idx]
    yc = y_curve[idx]

    plt.figure(figsize=(10, 6))
    if np.any(np.asarray(ye) != 0.0):
        plt.errorbar(x, y, yerr=ye[idx], fmt="o", capsize=2, label="pseudodata")
    else:
        plt.plot(x, y, "o", label="pseudodata")

    plt.plot(x, yc, label=curve_label)

    if y_band_std is not None:
        ys = y_band_std[idx]
        plt.fill_between(x, yc - ys, yc + ys, alpha=0.25, label=r"ensemble $\pm 1\sigma$")

    if truth_curve is not None:
        yt = truth_curve[idx]
        plt.plot(x, yt, linestyle="--", label="truth")

    plt.xlabel(r"$\phi$ (deg)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def forward_curves(
    forward: BKM10Forward,
    kin: np.ndarray,
    phi_rad: np.ndarray,
    cffs: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (XS,BSA,BCA) for a single kinematic point over a phi array."""
    t = np.full_like(phi_rad, float(kin[0]), dtype=np.float32)
    xb = np.full_like(phi_rad, float(kin[1]), dtype=np.float32)
    q2 = np.full_like(phi_rad, float(kin[2]), dtype=np.float32)

    # Per-row cffs (N,8)
    cffs_row = np.repeat(np.asarray(cffs, dtype=np.float32).reshape(1, 8), repeats=phi_rad.shape[0], axis=0)

    with torch.no_grad():
        xs_t, bsa_t, bca_t = forward.forward_xs_bsa_bca(
            torch.tensor(t, device=device),
            torch.tensor(xb, device=device),
            torch.tensor(q2, device=device),
            torch.tensor(phi_rad.astype(np.float32), device=device),
            torch.tensor(cffs_row, device=device),
        )

    return (
        xs_t.detach().cpu().numpy().astype(float),
        bsa_t.detach().cpu().numpy().astype(float),
        bca_t.detach().cpu().numpy().astype(float),
    )


def main() -> None:
    safe_mkdir(OUT_EVAL_DIR)

    # Load dataset
    dset = load_npz(DATA_NPZ)
    X = np.asarray(dset["x"], dtype=float)          # (N,4)
    y_central = np.asarray(dset["y_central"], dtype=float)
    y_sigma = np.asarray(dset["y_sigma"], dtype=float)

    if y_central.shape[1] != 3 or y_sigma.shape[1] != 3:
        raise ValueError(
            f"This evaluator expects y_central/y_sigma with 3 columns [XS,BSA,BCA]. "
            f"Got y_central shape {y_central.shape}, y_sigma shape {y_sigma.shape}.\n"
            f"Did you generate the dataset with generate_closure_dataset_torch_xsbsa_bca.py?"
        )

    # Load truth
    truth = load_json(TRUTH_JSON)
    single_truth, truth_map = build_truth_map(truth)

    # Discover replicas
    rep_npz = sorted(glob.glob(os.path.join(OUT_ROOT, "replicas", "replica_*", "extracted_cffs_per_point.npz")))
    if not rep_npz:
        raise FileNotFoundError(
            f"No replicas found under {OUT_ROOT}/replicas/replica_*/extracted_cffs_per_point.npz\n"
            f"Did you run train_cffs_8cff_xsbsa_bca.py with OUT_ROOT={OUT_ROOT!r}?"
        )

    kin0 = None
    cffs_all = []
    used_paths = []
    for p in rep_npz:
        z = np.load(p, allow_pickle=True)
        kin = np.asarray(z["unique_kin"], dtype=float)
        cffs = np.asarray(z["cffs"], dtype=float)
        if kin0 is None:
            kin0 = kin
        else:
            if kin.shape != kin0.shape:
                continue
        if cffs.shape != (kin.shape[0], 8):
            continue
        cffs_all.append(cffs)
        used_paths.append(p)

    cffs_all = np.stack(cffs_all, axis=0)  # (R,P,8)
    R, P, _ = cffs_all.shape
    kin0 = np.asarray(kin0, dtype=float)  # (P,3)
    truth_per_point = match_truth(kin0, single_truth, truth_map)  # (P,8)

    print(f"Loaded {R} replicas with P={P} point(s) from OUT_ROOT={OUT_ROOT}")
    print(f"Dataset: {DATA_NPZ}")
    print(f"Truth:   {TRUTH_JSON}")

    # Build forward model (for curves)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bkm_settings = truth.get("bkm10_settings", {})
    cfg = BKM10Config(
        k_beam=float(truth.get("kinematics", {}).get("beam_energy", bkm_settings.get("k_beam", 5.75))),
        using_ww=bool(bkm_settings.get("using_ww", True)),
        target_polarization=float(bkm_settings.get("target_polarization", 0.0)),
        eps=float(bkm_settings.get("eps", 1e-12)),
    )
    forward = BKM10Forward(cfg=cfg, dtype=torch.float32).to(device)
    forward.eval()

    # Optional: build one replica pseudodata sample to visualize
    y_plot = y_central
    if PLOT_REPLICA_PSEUDODATA is not None:
        ridx = int(PLOT_REPLICA_PSEUDODATA)
        rng = np.random.default_rng(int(REPLICA_SEED) + 1000 * (ridx + 1))
        noise = rng.normal(0.0, 1.0, size=y_central.shape)
        y_plot = y_central + noise * y_sigma

    # Evaluate point(s)
    points = list(range(P)) if bool(EVAL_ALL_POINTS) else [int(POINT_INDEX)]

    for pidx in points:
        point_tag = f"point{pidx:03d}"

        fit = cffs_all[:, pidx, :]             # (R,8)
        truth_vec = truth_per_point[pidx, :]   # (8,)
        resid = fit - truth_vec[None, :]

        # --- CFF residual summary ---
        hist_path = os.path.join(OUT_EVAL_DIR, f"cff_residuals_{point_tag}.png")
        plot_residual_hist(resid, hist_path, title=f"CFF residuals fit-truth ({point_tag})")

        csv_path = os.path.join(OUT_EVAL_DIR, f"cff_residuals_{point_tag}.csv")
        write_residual_csv(resid, csv_path, used_paths)

        summary = {}
        for i, name in enumerate(CFF_NAMES):
            summary[name] = {
                "truth": float(truth_vec[i]),
                "fit_mean": float(np.mean(fit[:, i])),
                "fit_std": float(np.std(fit[:, i], ddof=0)),
                "bias": float(np.mean(fit[:, i]) - truth_vec[i]),
            }
        with open(os.path.join(OUT_EVAL_DIR, f"cff_summary_{point_tag}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        # --- Observable curves ---
        # Select dataset rows for this point
        t0, xb0, q20 = kin0[pidx].tolist()
        mask = (
            np.isclose(X[:, 0], t0, atol=KIN_ATOL) &
            np.isclose(X[:, 1], xb0, atol=KIN_ATOL) &
            np.isclose(X[:, 2], q20, atol=KIN_ATOL)
        )
        if not np.any(mask):
            print(f"[WARN] no dataset rows matched {point_tag} kinematics; skipping curves")
            continue

        phi_rad = X[mask, 3].astype(float)
        phi_deg = np.degrees(phi_rad).astype(float)

        y_obs = y_plot[mask, :]
        y_err = y_sigma[mask, :]

        # ensemble mean CFFs
        cffs_mean = np.mean(fit, axis=0)

        xs_mean, bsa_mean, bca_mean = forward_curves(forward, kin0[pidx], phi_rad, cffs_mean, device)

        # truth curves
        xs_truth = bsa_truth = bca_truth = None
        if PLOT_TRUTH_CURVE:
            xs_truth, bsa_truth, bca_truth = forward_curves(forward, kin0[pidx], phi_rad, truth_vec, device)

        # ensemble band
        xs_std = bsa_std = bca_std = None
        if PLOT_ENSEMBLE_BAND and R > 1:
            xs_all = []
            bsa_all = []
            bca_all = []
            for r in range(R):
                xs_r, bsa_r, bca_r = forward_curves(forward, kin0[pidx], phi_rad, fit[r], device)
                xs_all.append(xs_r)
                bsa_all.append(bsa_r)
                bca_all.append(bca_r)
            xs_std = np.std(np.asarray(xs_all), axis=0, ddof=0)
            bsa_std = np.std(np.asarray(bsa_all), axis=0, ddof=0)
            bca_std = np.std(np.asarray(bca_all), axis=0, ddof=0)

        plot_pseudodata_with_curves(
            phi_deg, y_obs[:, 0], y_err[:, 0], xs_mean,
            curve_label="inferred (ensemble mean CFFs)",
            title=f"XS vs phi ({point_tag})",
            ylabel="XS",
            outpath=os.path.join(OUT_EVAL_DIR, f"xs_fit_{point_tag}.png"),
            y_band_std=xs_std,
            truth_curve=xs_truth,
        )

        plot_pseudodata_with_curves(
            phi_deg, y_obs[:, 1], y_err[:, 1], bsa_mean,
            curve_label="inferred (ensemble mean CFFs)",
            title=f"BSA vs phi ({point_tag})",
            ylabel="BSA",
            outpath=os.path.join(OUT_EVAL_DIR, f"bsa_fit_{point_tag}.png"),
            y_band_std=bsa_std,
            truth_curve=bsa_truth,
        )

        plot_pseudodata_with_curves(
            phi_deg, y_obs[:, 2], y_err[:, 2], bca_mean,
            curve_label="inferred (ensemble mean CFFs)",
            title=f"BCA vs phi ({point_tag})",
            ylabel="BCA",
            outpath=os.path.join(OUT_EVAL_DIR, f"bca_fit_{point_tag}.png"),
            y_band_std=bca_std,
            truth_curve=bca_truth,
        )

        print(f"[{point_tag}] wrote:")
        print(f"  - {os.path.basename(hist_path)}")
        print(f"  - {os.path.basename(csv_path)}")
        print(f"  - xs_fit_{point_tag}.png / bsa_fit_{point_tag}.png / bca_fit_{point_tag}.png")

    print("\nDone. Output dir:", OUT_EVAL_DIR)


if __name__ == "__main__":
    main()
