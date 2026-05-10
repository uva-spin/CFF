#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""jacobian_worker.py

Worker script for a single kinematic point in the Jacobian sweep.
Designed to be called by a Slurm array job via:

    python jacobian_worker.py --task-id $SLURM_ARRAY_TASK_ID \
                              --grid kin_grid.json \
                              --outdir results/

For each kinematic point, computes the Jacobian via finite
differences over all phi grid values, performs SVD, and saves results
for every subset of observables.

Outputs (per point, in --outdir):
    point_{task_id:05d}.npz
        Arrays saved:
          kin       : (3,) [t, xB, Q2]
          subsets   : object array of tuples, shape (31,)
          rank      : (31,) int
          sigma_min : (31,) float
          cond      : (31,) float
          sv_all    : (31, 8) float  -- all 8 singular values per subset
          best_subset_rank_idx : scalar int
          best_subset_cond_idx : scalar int

Usage notes
-----------
- Place this file in the same directory as cross_section_script.py
- The kin_grid.json is read-only; all writes go to --outdir
- Each task is independent
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ============================================================
# DEFAULTS (override via CLI)
# ============================================================
DEFAULT_GRID = "kin_grid.json"
DEFAULT_OUTDIR = "results"
K_BEAM = 5.75
USING_WW = True
FD_REL_EPS = 1e-3        # finite-difference relative step
PHI_N = 50               # phi points per observable
PHI_MIN_DEG = 0.0
PHI_MAX_DEG = 360.0
RANK_TOL = 1e-3          # S[i]/S[0] threshold for effective rank
N_CFFS = 8
ALL_OBS = ["XS", "BSA", "BCA", "TSA", "DSA"]

# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Single-point Jacobian worker")
    p.add_argument("--task-id", type=int, required=True,
                   help="Slurm array task ID (0-indexed into kin_grid.json)")
    p.add_argument("--grid", type=str, default=DEFAULT_GRID,
                   help="Path to kin_grid.json")
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR,
                   help="Directory for output .npz files")
    p.add_argument("--k-beam", type=float, default=K_BEAM)
    p.add_argument("--phi-n", type=int, default=PHI_N)
    p.add_argument("--fd-eps", type=float, default=FD_REL_EPS)
    p.add_argument("--rank-tol", type=float, default=RANK_TOL)
    return p.parse_args()


def load_forward():
    """Import compute_observables lazily so the script fails fast on import error."""
    try:
        from cross_section_script import compute_observables
        return compute_observables
    except ImportError as e:
        print(f"ERROR: cannot import cross_section_script: {e}", file=sys.stderr)
        sys.exit(1)


def forward_vec(compute_obs, phi_rad, k_beam, q2, xb, t, c8, active_obs):
    """Evaluate selected observables for the given CFF vector, returns 1-D array.
    active_obs entries are upper-case (e.g. 'XS', 'BSA'); mapped to lowercase
    keys used internally by compute_observables.
    (This is a stupid fix to a name error that was making this script incompatible with the functions in "compute_observables")
    """
    _key = {"XS": "xs", "BSA": "bsa", "BCA": "bca", "TSA": "tsa", "DSA": "dsa"}
    cffs = dict(
        re_h=float(c8[0]), re_e=float(c8[1]),
        re_ht=float(c8[2]), re_et=float(c8[3]),
        im_h=float(c8[4]), im_e=float(c8[5]),
        im_ht=float(c8[6]), im_et=float(c8[7]),
    )
    obs = compute_obs(
        phi_rad=phi_rad,
        k_beam=float(k_beam),
        q_squared=float(q2),
        xb=float(xb),
        t=float(t),
        cffs=cffs,
        using_ww=bool(USING_WW),
    )
    return np.column_stack([obs[_key[o]] for o in active_obs]).reshape(-1)


def compute_jacobian(compute_obs, phi_rad, k_beam, q2, xb, t, truth8, active_obs, fd_eps):
    """Finite-difference Jacobian: shape (N_obs*N_phi, N_CFFS), column-normalized (see whitening section in Dustin's Readme for details)."""
    y0 = forward_vec(compute_obs, phi_rad, k_beam, q2, xb, t, truth8, active_obs)
    N = y0.size
    J = np.zeros((N, N_CFFS), dtype=float)
    for j in range(N_CFFS):
        step = fd_eps * max(1.0, abs(truth8[j]))
        cp = truth8.copy(); cp[j] += step
        cm = truth8.copy(); cm[j] -= step
        J[:, j] = (
            forward_vec(compute_obs, phi_rad, k_beam, q2, xb, t, cp, active_obs) -
            forward_vec(compute_obs, phi_rad, k_beam, q2, xb, t, cm, active_obs)
        ) / (2 * step)
    # Column-normalization
    col_norms = np.linalg.norm(J, axis=0)
    col_norms[col_norms == 0] = 1.0
    J /= col_norms[None, :]
    return J


def svd_metrics(J, rank_tol):
    """Return (singular_values_8, effective_rank, sigma_min, condition_number)."""
    k = min(J.shape)
    _, S_raw, _ = np.linalg.svd(J, full_matrices=False)
    # Pad to length 8 in case k < 8
    S = np.zeros(N_CFFS)
    S[:len(S_raw)] = S_raw

    rank = int(np.sum(S / max(S[0], 1e-300) > rank_tol)) if S[0] > 0 else 0
    sigma_min = float(S[rank-1]) if rank > 0 else 0.0
    cond = float(S[0] / S[rank-1]) if (rank > 0 and S[rank-1] > 0) else np.inf
    return S, rank, sigma_min, cond


def all_subsets(obs_list: List[str]):
    """Generate all 2^n - 1 non-empty subsets in order of size."""
    for k in range(1, len(obs_list) + 1):
        yield from combinations(obs_list, k)


def main():
    args = parse_args()

    # Load grid
    grid_path = Path(args.grid)
    if not grid_path.exists():
        print(f"ERROR: grid file not found: {grid_path}", file=sys.stderr)
        sys.exit(1)
    with open(grid_path) as f:
        grid = json.load(f)

    tid = args.task_id
    if tid < 0 or tid >= len(grid):
        print(f"ERROR: task_id {tid} out of range [0, {len(grid)-1}]", file=sys.stderr)
        sys.exit(1)

    point = grid[tid]
    t   = float(point["t"])
    xb  = float(point["xB"])
    q2  = float(point["Q2"])
    truth8 = np.array(point["truth_cffs"], dtype=float)

    # Phi grid
    phi_rad = np.deg2rad(
        np.linspace(PHI_MIN_DEG, PHI_MAX_DEG, args.phi_n, endpoint=False)
    ).astype(float)

    compute_obs = load_forward()

    # Enumerate all subsets
    subsets = list(all_subsets(ALL_OBS))   # 31 non-empty subsets of 5 observables
    n_sub = len(subsets)

    ranks      = np.zeros(n_sub, dtype=int)
    sigma_mins = np.zeros(n_sub, dtype=float)
    conds      = np.zeros(n_sub, dtype=float)
    sv_all     = np.zeros((n_sub, N_CFFS), dtype=float)

    for i, subset in enumerate(subsets):
        try:
            J = compute_jacobian(
                compute_obs, phi_rad, args.k_beam,
                q2, xb, t, truth8, list(subset), args.fd_eps
            )
            S, rank, sigma_min, cond = svd_metrics(J, args.rank_tol)
        except Exception as e:
            print(f"  WARNING: subset {subset} failed: {e}", file=sys.stderr)
            S = np.zeros(N_CFFS)
            rank, sigma_min, cond = 0, 0.0, np.inf

        ranks[i]      = rank
        sigma_mins[i] = sigma_min
        conds[i]      = cond
        sv_all[i]     = S

    # Best subsets
    full_rank_idx = np.where(ranks >= N_CFFS)[0]
    if len(full_rank_idx) > 0:
        best_rank_idx = int(full_rank_idx[np.argmin(conds[full_rank_idx])])
    else:
        best_rank_idx = int(np.argmax(ranks))   # fallback: highest rank

    best_cond_idx = int(np.argmin(np.where(np.isfinite(conds), conds, np.inf)))

    # Save
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"point_{tid:05d}.npz"

    # Store subsets as a fixed-length string array (max 30 chars)
    subset_strs = np.array([",".join(s) for s in subsets], dtype="U64")

    np.savez(
        outpath,
        kin=np.array([t, xb, q2], dtype=float),
        truth_cffs=truth8,
        subsets=subset_strs,
        rank=ranks,
        sigma_min=sigma_mins,
        cond=conds,
        sv_all=sv_all,
        best_subset_rank_idx=np.array(best_rank_idx),
        best_subset_cond_idx=np.array(best_cond_idx),
    )

    # Human-readable summary to stdout (captured in Slurm log)
    print(f"Task {tid}: t={t:.3f} xB={xb:.3f} Q2={q2:.3f}")
    print(f"  Full-rank ({N_CFFS}) subsets: {len(full_rank_idx)} / {n_sub}")
    if len(full_rank_idx) > 0:
        best = subsets[best_rank_idx]
        print(f"  Best full-rank subset (lowest cond): {best}  cond={conds[best_rank_idx]:.2e}")
    else:
        print(f"  No full-rank subset found. Max rank={ranks.max()}")
    print(f"  Saved: {outpath}")


if __name__ == "__main__":
    main()
