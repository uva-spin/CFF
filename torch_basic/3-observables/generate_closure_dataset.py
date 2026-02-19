#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This will just generate one point as its set now.  The point is from the data but not that meaningful.

Closure dataset generator (pure torch forward) for a DVCS fixed-kinematics (or multi-kinematics)
closure test.

This generator is compatible with:
  - bkm10_torch_forward.BKM10Forward.forward_xs_bsa_bca  (XS, BSA, BCA)
  - train_cffs_8cff_xsbsa_bca.py  (replica training)
  - closure_evaluate_torch_xsbsa_bca.py (evaluation)

Outputs
-------
<OUTPUT_DIR>/data/dataset_<TAG>.npz with:
  x         : (N,4)  [t, xB, Q2, phi(rad)]
  y_central : (N,3)  [XS, BSA, BCA]
  y_sigma   : (N,3)  [XS_err, BSA_err, BCA_err]

<OUTPUT_DIR>/data/truth_<TAG>.json contains kinematics + truth CFFs (per point).

Important note on "sampling"
-----------------------------
This script primarily defines:
  * the pseudo-data central values (y_central)
  * the per-point 1σ uncertainties (y_sigma)

Replica-to-replica *sampling* typically happens during training:
  y_rep = y_central + Normal(0,1)*y_sigma

If you want the central dataset itself to be a single noisy pseudo-experiment, set
NOISY_CENTRAL=True.

"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from bkm10_torch_forward import BKM10Config, BKM10Forward


# ============================================================
# CONFIG (edit here)
# ============================================================

OUTPUT_DIR = "output"     # will write into OUTPUT_DIR/data/
TAG = "v_1"

DATA_NPZ = os.path.join(OUTPUT_DIR, "data", f"dataset_{TAG}.npz")
TRUTH_JSON = os.path.join(OUTPUT_DIR, "data", f"truth_{TAG}.json")

# ---- Kinematics ----
# If you want multi-kinematics, add multiple dict entries.
KIN_POINTS: List[Dict[str, float]] = [
    {"t": -0.17, "xB": 0.34, "Q2": 1.82},
]

K_BEAM = 5.75
USING_WW = True
TARGET_POLARIZATION = 0.0

# For clarity only: the BSA returned by the forward model is an analyzing power
# corresponding to 100% beam polarization. If you want to model a measured
# asymmetry with finite polarization P_b, you would typically use
#   BSA_meas ≈ P_b * BSA
# We store this in the truth JSON for provenance.
LEPTON_BEAM_POLARIZATION = 1.0

# Small stabilizer for denominators
EPS = 1e-12

# ---- Phi grid ----
PHI_N = 24
PHI_MIN_DEG = 0.0
PHI_MAX_DEG = 360.0
PHI_ENDPOINT = False

# ---- Truth CFFs per point (8-vector) ----
# Order: [ReH, ReE, ReHtilde, ReEtilde, ImH, ImE, ImHtilde, ImEtilde]
# If you provide a single vector, it is used for all kinematic points.
TRUTH_CFFS_SINGLE = np.array([
    1.0,            # ReH
    2.2173543720,   # ReE
    1.4093937265,   # ReHtilde
    144.410164202,  # ReEtilde
    1.0,            # ImH
    0.0,            # ImE
    1.5773644026,   # ImHtilde
    0.0,            # ImEtilde
], dtype=float)

# ---- Experimental uncertainties used for replica sampling ----
# XS errors: relative + absolute
XS_ERR_REL = 0.03
XS_ERR_ABS = 0.0

# BSA errors: absolute + relative
BSA_ERR_ABS = 0.01
BSA_ERR_REL = 0.0

# BCA errors: absolute + relative
BCA_ERR_ABS = 0.02
BCA_ERR_REL = 0.0

# If True, y_central is sampled once from truth with y_sigma.
# If False, y_central == truth and replicas are sampled in training.
NOISY_CENTRAL = False
CENTRAL_SEED = 123

# ============================================================
# END CONFIG
# ============================================================

CFF_NAMES = ["ReH","ReE","ReHtilde","ReEtilde","ImH","ImE","ImHtilde","ImEtilde"]
OBS_NAMES = ["XS","BSA","BCA"]


def safe_mkdir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def truth_vec_to_km15_dict(v: np.ndarray) -> Dict[str, float]:
    v = np.asarray(v, dtype=float).reshape(-1)
    assert v.shape[0] == 8
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
    phi_rad = np.deg2rad(phi_deg).astype(np.float32)

    # Truth CFFs per point
    truth_cffs = []
    for _ in KIN_POINTS:
        truth_cffs.append(np.array(TRUTH_CFFS_SINGLE, dtype=float).copy())
    truth_cffs = np.stack(truth_cffs, axis=0)  # (P,8)

    # Build X grid: each point has all phi values
    X_rows = []
    point_ids = []
    for pid, k in enumerate(KIN_POINTS):
        for ph in phi_rad:
            X_rows.append([float(k["t"]), float(k["xB"]), float(k["Q2"]), float(ph)])
            point_ids.append(pid)

    X = np.asarray(X_rows, dtype=np.float32)  # (N,4)
    point_ids = np.asarray(point_ids, dtype=np.int64)  # (N,)

    # Forward model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fwd = BKM10Forward(BKM10Config(k_beam=float(K_BEAM), target_polarization=float(TARGET_POLARIZATION), using_ww=bool(USING_WW), eps=float(EPS))).to(device=device)
    fwd.eval()

    # Build per-row CFFs (each row uses the truth for its point)
    cffs_rows = truth_cffs[point_ids]  # (N,8)

    # Torch tensors
    t_t = torch.tensor(X[:, 0], device=device)
    xb_t = torch.tensor(X[:, 1], device=device)
    q2_t = torch.tensor(X[:, 2], device=device)
    phi_t = torch.tensor(X[:, 3], device=device)
    cffs_t = torch.tensor(cffs_rows, device=device, dtype=torch.float32)

    with torch.no_grad():
        xs_t, bsa_t, bca_t = fwd.forward_xs_bsa_bca(t_t, xb_t, q2_t, phi_t, cffs_t)

    xs = xs_t.detach().cpu().numpy().astype(np.float32)
    bsa = bsa_t.detach().cpu().numpy().astype(np.float32)
    bca = bca_t.detach().cpu().numpy().astype(np.float32)

    # Per-point uncertainties
    xs_sig = (float(XS_ERR_REL) * np.abs(xs) + float(XS_ERR_ABS)).astype(np.float32)
    bsa_sig = (float(BSA_ERR_REL) * np.abs(bsa) + float(BSA_ERR_ABS)).astype(np.float32)
    bca_sig = (float(BCA_ERR_REL) * np.abs(bca) + float(BCA_ERR_ABS)).astype(np.float32)

    y_truth = np.column_stack([xs, bsa, bca]).astype(np.float32)
    y_sigma = np.column_stack([xs_sig, bsa_sig, bca_sig]).astype(np.float32)

    if bool(NOISY_CENTRAL):
        rng = np.random.default_rng(int(CENTRAL_SEED))
        noise = rng.normal(0.0, 1.0, size=y_truth.shape).astype(np.float32)
        y_central = y_truth + noise * y_sigma
    else:
        y_central = y_truth

    np.savez(DATA_NPZ, x=X, y_central=y_central, y_sigma=y_sigma)

    # Truth JSON
    k0 = KIN_POINTS[0]
    truth_obj = {
        "generator": {
            "script": "generate_closure_dataset_torch_xsbsa_bca.py",
            "TAG": str(TAG),
            "OBS_NAMES": OBS_NAMES,
            "PHI_N": int(PHI_N),
            "PHI_MIN_DEG": float(PHI_MIN_DEG),
            "PHI_MAX_DEG": float(PHI_MAX_DEG),
            "PHI_ENDPOINT": bool(PHI_ENDPOINT),
            "NOISY_CENTRAL": bool(NOISY_CENTRAL),
            "XS_ERR_REL": float(XS_ERR_REL),
            "XS_ERR_ABS": float(XS_ERR_ABS),
            "BSA_ERR_REL": float(BSA_ERR_REL),
            "BSA_ERR_ABS": float(BSA_ERR_ABS),
            "BCA_ERR_REL": float(BCA_ERR_REL),
            "BCA_ERR_ABS": float(BCA_ERR_ABS),
        },
        "kinematics": {
            "beam_energy": float(K_BEAM),
            "q_squared": float(k0["Q2"]),
            "x_b": float(k0["xB"]),
            "t": float(k0["t"]),
        },
        "bkm10_settings": {
            "using_ww": bool(USING_WW),
            "target_polarization": float(TARGET_POLARIZATION),
            # Kept for backwards compatibility; BSA here is analyzing power (100% polarization)
            "lepton_beam_polarization": float(LEPTON_BEAM_POLARIZATION),
            "eps": float(EPS),
        },
        "km15_truth_cffs": truth_vec_to_km15_dict(truth_cffs[0]),
        "kinematics_points": [
            {"t": float(k["t"]), "x_b": float(k["xB"]), "q_squared": float(k["Q2"]), "beam_energy": float(K_BEAM)}
            for k in KIN_POINTS
        ],
        "truth_cffs_per_point": [
            {
                "t": float(k["t"]),
                "x_b": float(k["xB"]),
                "q_squared": float(k["Q2"]),
                "beam_energy": float(K_BEAM),
                "km15_truth_cffs": truth_vec_to_km15_dict(v),
            }
            for k, v in zip(KIN_POINTS, truth_cffs)
        ],
    }

    save_json(TRUTH_JSON, truth_obj)

    print("Wrote dataset:", DATA_NPZ)
    print("Wrote truth:  ", TRUTH_JSON)
    print(f"Points P={len(KIN_POINTS)}, phi={len(phi_rad)}, total N={X.shape[0]}")
    print("XS range:", float(xs.min()), float(xs.max()))
    print("BSA range:", float(bsa.min()), float(bsa.max()))
    print("BCA range:", float(bca.min()), float(bca.max()))
    print("y_sigma medians:", float(np.median(y_sigma[:,0])), float(np.median(y_sigma[:,1])), float(np.median(y_sigma[:,2])))

    if len(KIN_POINTS) == 1:
        print("Truth CFFs (single point):")
        for name, val in zip(CFF_NAMES, truth_cffs[0].tolist()):
            print(f"  {name:8s} = {val:.6g}")


if __name__ == "__main__":
    main()
