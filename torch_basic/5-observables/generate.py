#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate.py
There is nothing spectial about the setting used in this example.
Closure dataset generator for DVCS CFF extraction using a five observables example
computed from a KM15/BKM10-style forward model implemented in:

    cross_section_script.py

Observables (per-phi)
---------------------
  - XS  : unpolarized (helicity-averaged) cross section
  - BSA : beam-spin asymmetry analyzing power
  - BCA : beam-charge asymmetry analyzing power
  - TSA : longitudinal target-spin asymmetry analyzing power
  - DSA : longitudinal double-spin asymmetry analyzing power (beam x target)

Important convention note
-------------------------
BSA/BCA/TSA/DSA are computed here as *analyzing powers* from ratios of cross
sections evaluated at idealized polarization states (beam helicity +/-1, charge ±1,
target S_L=±0.5). Therefore you do NOT need to set a nonzero experimental
polarization magnitude to generate these analyzing powers.

Outputs
-------
<OUTPUT_DIR>/data/dataset_<TAG>.npz
    x         : (N,4) [t, xB, Q2, phi(rad)]
    y_central : (N,5) [XS, BSA, BCA, TSA, DSA]
    y_sigma   : (N,5) [XS_err, BSA_err, BCA_err, TSA_err, DSA_err]

<OUTPUT_DIR>/data/truth_<TAG>.json
    Kinematics + forward settings + truth CFFs (8 components)

Replica noise
-------------
Typically you keep y_central equal to truth and sample replica-to-replica noise
inside the training loop as:

    y_rep = y_central + Normal(0,1) * y_sigma

"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from cross_section_script import compute_observables


# ============================================================
# CONFIG (edit here)
# ============================================================

OUTPUT_DIR = "output"  # writes into OUTPUT_DIR/data/
TAG = "v_1"

DATA_NPZ = os.path.join(OUTPUT_DIR, "data", f"dataset_{TAG}.npz")
TRUTH_JSON = os.path.join(OUTPUT_DIR, "data", f"truth_{TAG}.json")

# ---- Kinematics ----
# Add multiple dict entries for multi-kinematics if you want, but the truth JSON
# is primarily designed for the single-point closure test.
KIN_POINTS: List[Dict[str, float]] = [
    {"t": -0.17, "xB": 0.34, "Q2": 1.82},
]

K_BEAM = 5.75
USING_WW = True

# ---- Phi grid ----
PHI_N = 100
PHI_MIN_DEG = 0.0
PHI_MAX_DEG = 360.0
PHI_ENDPOINT = False

# ---- Truth CFFs (8-vector) ----
# Order: [ReH, ReE, ReHtilde, ReEtilde, ImH, ImE, ImHtilde, ImEtilde]
TRUTH_CFFS_SINGLE = np.array(
    [
        1.0,  # ReH
        2.2173543720,  # ReE
        1.4093937265,  # ReHtilde
        144.410164202,  # ReEtilde
        1.0,  # ImH
        0.0,  # ImE
        1.5773644026,  # ImHtilde
        0.0,  # ImEtilde
    ],
    dtype=float,
)

# ---- Experimental uncertainties (used for replica sampling) ----
# XS errors: relative + absolute
XS_ERR_REL = 0.03
XS_ERR_ABS = 0.0

# Asymmetry errors: absolute + relative (typically absolute dominates)
BSA_ERR_ABS = 0.01
BSA_ERR_REL = 0.0

BCA_ERR_ABS = 0.02
BCA_ERR_REL = 0.0

TSA_ERR_ABS = 0.02
TSA_ERR_REL = 0.0

DSA_ERR_ABS = 0.02
DSA_ERR_REL = 0.0

# If True, y_central is sampled once from truth using y_sigma.
# If False, y_central == truth (recommended for closure debugging).
NOISY_CENTRAL = False
CENTRAL_SEED = 123

# ============================================================
# END CONFIG
# ============================================================

CFF_NAMES = ["ReH", "ReE", "ReHtilde", "ReEtilde", "ImH", "ImE", "ImHtilde", "ImEtilde"]
OBS_NAMES = ["XS", "BSA", "BCA", "TSA", "DSA"]


def safe_mkdir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def truth_vec_to_km15_dict(v: np.ndarray) -> Dict[str, float]:
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.shape[0] != 8:
        raise ValueError("Truth CFF vector must have length 8")
    return {
        "cff_real_h_km15": float(v[0]),
        "cff_real_e_km15": float(v[1]),
        "cff_real_ht_km15": float(v[2]),
        "cff_real_et_km15": float(v[3]),
        "cff_imag_h_km15": float(v[4]),
        "cff_imag_e_km15": float(v[5]),
        "cff_imag_ht_km15": float(v[6]),
        "cff_imag_et_km15": float(v[7]),
    }


def main() -> None:
    safe_mkdir(os.path.join(OUTPUT_DIR, "data"))

    # Phi grid
    phi_deg = np.linspace(float(PHI_MIN_DEG), float(PHI_MAX_DEG), int(PHI_N), endpoint=bool(PHI_ENDPOINT))
    phi_rad = np.deg2rad(phi_deg).astype(np.float64)

    # Truth CFFs per point (same truth vector for each point unless you extend this)
    truth_cffs = np.stack([np.array(TRUTH_CFFS_SINGLE, dtype=float).copy() for _ in KIN_POINTS], axis=0)  # (P,8)

    # Build X grid: each kinematic point has all phi values
    X_rows = []
    point_ids = []
    for pid, k in enumerate(KIN_POINTS):
        for ph in phi_rad:
            X_rows.append([float(k["t"]), float(k["xB"]), float(k["Q2"]), float(ph)])
            point_ids.append(pid)

    X = np.asarray(X_rows, dtype=np.float32)  # (N,4)
    point_ids = np.asarray(point_ids, dtype=np.int64)

    # Forward truth values
    y_truth = np.zeros((X.shape[0], 5), dtype=np.float64)
    for pid, k in enumerate(KIN_POINTS):
        mask = point_ids == pid
        phis = X[mask, 3].astype(np.float64)
        v = truth_cffs[pid]
        cffs = {
            "re_h": float(v[0]),
            "re_e": float(v[1]),
            "re_ht": float(v[2]),
            "re_et": float(v[3]),
            "im_h": float(v[4]),
            "im_e": float(v[5]),
            "im_ht": float(v[6]),
            "im_et": float(v[7]),
        }
        obs = compute_observables(
            phi_rad=phis,
            k_beam=float(K_BEAM),
            q_squared=float(k["Q2"]),
            xb=float(k["xB"]),
            t=float(k["t"]),
            cffs=cffs,
            using_ww=bool(USING_WW),
        )
        y_truth[mask, 0] = obs["xs"]
        y_truth[mask, 1] = obs["bsa"]
        y_truth[mask, 2] = obs["bca"]
        y_truth[mask, 3] = obs["tsa"]
        y_truth[mask, 4] = obs["dsa"]

    y_truth = y_truth.astype(np.float32)

    # Per-point uncertainties
    xs = y_truth[:, 0]
    bsa = y_truth[:, 1]
    bca = y_truth[:, 2]
    tsa = y_truth[:, 3]
    dsa = y_truth[:, 4]

    xs_sig = (float(XS_ERR_REL) * np.abs(xs) + float(XS_ERR_ABS)).astype(np.float32)
    bsa_sig = (float(BSA_ERR_REL) * np.abs(bsa) + float(BSA_ERR_ABS)).astype(np.float32)
    bca_sig = (float(BCA_ERR_REL) * np.abs(bca) + float(BCA_ERR_ABS)).astype(np.float32)
    tsa_sig = (float(TSA_ERR_REL) * np.abs(tsa) + float(TSA_ERR_ABS)).astype(np.float32)
    dsa_sig = (float(DSA_ERR_REL) * np.abs(dsa) + float(DSA_ERR_ABS)).astype(np.float32)

    y_sigma = np.column_stack([xs_sig, bsa_sig, bca_sig, tsa_sig, dsa_sig]).astype(np.float32)

    if bool(NOISY_CENTRAL):
        rng = np.random.default_rng(int(CENTRAL_SEED))
        noise = rng.normal(0.0, 1.0, size=y_truth.shape).astype(np.float32)
        y_central = y_truth + noise * y_sigma
    else:
        y_central = y_truth.copy()

    np.savez(DATA_NPZ, x=X.astype(np.float32), y_central=y_central.astype(np.float32), y_sigma=y_sigma.astype(np.float32))
    print("Wrote:", DATA_NPZ)

    # Truth JSON (keeps older-style keys for single-kin evaluations)
    kin0 = KIN_POINTS[0]
    truth_obj = {
        "kinematics_points": KIN_POINTS,
        "kinematics": {
            "beam_energy": float(K_BEAM),
            "q_squared": float(kin0["Q2"]),
            "x_b": float(kin0["xB"]),
            "t": float(kin0["t"]),
        },
        "bkm10_settings": {
            "using_ww": bool(USING_WW),
            # These are not needed to *compute* the analyzing powers, but we store them
            # for completeness / compatibility with older scripts.
            "target_polarization": 0.0,
            "lepton_beam_polarization": 0.0,
        },
        "truth_cffs_vector_order": CFF_NAMES,
        "truth_cffs_vector": [float(x) for x in TRUTH_CFFS_SINGLE.reshape(-1)],
        "km15_truth_cffs": truth_vec_to_km15_dict(TRUTH_CFFS_SINGLE),
        "observable_names": OBS_NAMES,
        "phi_grid_deg": [float(x) for x in phi_deg.tolist()],
        "y_sigma_convention": {
            "XS": {"rel": float(XS_ERR_REL), "abs": float(XS_ERR_ABS)},
            "BSA": {"rel": float(BSA_ERR_REL), "abs": float(BSA_ERR_ABS)},
            "BCA": {"rel": float(BCA_ERR_REL), "abs": float(BCA_ERR_ABS)},
            "TSA": {"rel": float(TSA_ERR_REL), "abs": float(TSA_ERR_ABS)},
            "DSA": {"rel": float(DSA_ERR_REL), "abs": float(DSA_ERR_ABS)},
        },
    }
    save_json(TRUTH_JSON, truth_obj)
    print("Wrote:", TRUTH_JSON)


if __name__ == "__main__":
    main()
