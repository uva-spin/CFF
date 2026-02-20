#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_cffs_8cff.py

Basic training example.  Not well tuned or optimized yet.
Deterministic (or noisy) replica trainer for extracting 8 CFF components
from a simultaneous fit to five observables at fixed kinematics:

    XS, BSA, BCA, TSA, DSA

where each observable is computed by:

    cross_section_script_xsbsa_bca_tsa_dsa.compute_observables(...)

This script is intentionally:
  - easy to edit (all configuration is at the top)
  - replica-friendly (writes replica_XXX/ directories)
  - diagnostic-friendly (prints per-observable residuals during optimization)

Optimization
------------
This version uses a NumPy forward model with finite-difference gradients
(so it is CPU-bound). For strict closure debugging (0-noise), LBFGS is usually
more reliable than Adam.

Outputs
-------
<OUT_ROOT>/replicas/replica_XXX/
  model.pt                        (safe torch tensors + metadata)
  extracted_cffs_per_point.npz    (unique_kin + cffs)
  history.json

"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from cross_section_script import compute_observables


# ============================================================
# CONFIG (edit here)
# ============================================================

DATA_NPZ = "output/data/dataset_v_1.npz"
TRUTH_JSON = "output/data/truth_v_1.json"   # optional (for init/debug)

OUT_ROOT = "output_torch_8cff"

N_REPLICAS = 10
REPLICA_SEED = 2222

# For strict closure debugging, set to 1.0
TRAIN_FRACTION = 1.0
SPLIT_SEED = 42

# Model mode:
#   "embedding": one independent 8-vector per unique (t,xB,Q2) point.
MODEL_MODE = "embedding"

# ---- Loss behavior ----
# "scale_mse": normalized MSE using fixed scales per observable.
# This is recommended for 0-error closure debugging.
LOSS_MODE = "scale_mse"

# Normalization scales (not dataset sigmas)
SCALE_XS_REL = 1e-3         # XS scale = SCALE_XS_REL * median(|XS|)
SCALE_ASYM_ABS = 1e-3       # for BSA/BCA/TSA/DSA

# Residual clipping (usually OFF for strict closure)
RATIO_CLIP = None

# ---- Finite difference ----
# eps_j = FD_EPS * (|theta_j| + 1)
FD_EPS = 2e-3

# ---- Optimization ----
OPTIMIZER = "lbfgs"   # "adam" or "lbfgs"

# Adam
ADAM_LR = 3e-3
ADAM_CLIPNORM = 5.0
EPOCHS = 4000
PATIENCE = 600

# LBFGS (recommended for P=1 deterministic fits)
LBFGS_LR = 1.0
LBFGS_MAX_ITER_PER_STEP = 50
LBFGS_STEPS = 200
LBFGS_HISTORY_SIZE = 100
LBFGS_TOL_GRAD = 1e-12
LBFGS_TOL_CHANGE = 1e-12

# Print diagnostics every N outer steps
PRINT_EVERY = 10

# Multi-start (restarts) per replica; keep best
N_RESTARTS = 3

# Initialization:
#   "random"       : theta ~ Uniform(-INIT_SCALE, INIT_SCALE)
#   "truth"        : init exactly at truth (requires TRUTH_JSON)
#   "truth+jitter" : truth + small jitter
INIT_MODE = "random"
INIT_SCALE = 0.3
TRUTH_JITTER = 0.05

# ---- Forward-model settings ----
K_BEAM = 5.75
USING_WW = True

# Parameter scaling (critical if Etilde ~ O(100))
# Optimize theta and map to physical CFFs by: cffs = theta * CFF_SCALES
CFF_SCALES = np.array([1.0, 1.0, 1.0, 150.0, 1.0, 1.0, 1.0, 150.0], dtype=np.float64)

# ============================================================
# END CONFIG
# ============================================================

CFF_NAMES = ["ReH", "ReE", "ReHtilde", "ReEtilde", "ImH", "ImE", "ImHtilde", "ImEtilde"]
OBS_NAMES = ["XS", "BSA", "BCA", "TSA", "DSA"]

DTYPE = torch.float64


def safe_mkdir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_truth8(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    tr = json.load(open(path, "r"))
    c = tr.get("km15_truth_cffs", {})
    keys = [
        "cff_real_h_km15", "cff_real_e_km15", "cff_real_ht_km15", "cff_real_et_km15",
        "cff_imag_h_km15", "cff_imag_e_km15", "cff_imag_ht_km15", "cff_imag_et_km15",
    ]
    if not all(k in c for k in keys):
        return None
    return np.array([float(c[k]) for k in keys], dtype=np.float64)


def unique_kinematics(X: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Return unique kin (P,3) and point_id (N,) for X=(N,4)=[t,xB,Q2,phi]."""
    kin = X[:, :3].astype(np.float64)
    q = np.round(kin / tol) * tol
    uniq, inv = np.unique(q, axis=0, return_inverse=True)
    return uniq.astype(np.float64), inv.astype(np.int64)


def _compute_scales(y: np.ndarray) -> np.ndarray:
    """Per-observable scales for LOSS_MODE='scale_mse'."""
    xs = y[:, 0]
    med_xs = float(np.median(np.abs(xs[np.isfinite(xs)]))) if np.isfinite(xs).any() else 1.0
    med_xs = med_xs if med_xs > 0 else 1.0
    s_xs = float(SCALE_XS_REL) * med_xs

    s_asym = float(SCALE_ASYM_ABS)
    scales = np.array([s_xs, s_asym, s_asym, s_asym, s_asym], dtype=np.float64)
    scales = np.where(scales > 0, scales, 1.0)
    return scales


class ScaleMSELoss(nn.Module):
    def __init__(self, scales: np.ndarray, ratio_clip: Optional[float] = None):
        super().__init__()
        self.register_buffer("scales", torch.tensor(scales, dtype=DTYPE))
        self.ratio_clip = float(ratio_clip) if (ratio_clip is not None and ratio_clip > 0) else None

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        r = (y_true - y_pred) / self.scales[None, :]
        if self.ratio_clip is not None:
            c = torch.tensor(self.ratio_clip, dtype=r.dtype, device=r.device)
            r = torch.clamp(r, -c, c)
        return 0.5 * torch.mean(r * r)


class BKM10AllObsFD(torch.autograd.Function):
    """NumPy forward + FD backward for (XS,BSA,BCA,TSA,DSA)."""

    @staticmethod
    def forward(ctx, theta_per_point: torch.Tensor, unique_kin: torch.Tensor,
                point_id: torch.Tensor, phi: torch.Tensor,
                fd_eps: float, k_beam: float, using_ww: bool):

        cff_scales_t = torch.tensor(CFF_SCALES, dtype=theta_per_point.dtype, device=theta_per_point.device)
        cffs_per_point = theta_per_point * cff_scales_t[None, :]

        # Move to CPU numpy
        uniq_np = unique_kin.detach().cpu().numpy().astype(np.float64)  # (P,3)
        pid_np = point_id.detach().cpu().numpy().astype(np.int64)       # (N,)
        phi_np = phi.detach().cpu().numpy().astype(np.float64)          # (N,)
        cffs_np = cffs_per_point.detach().cpu().numpy().astype(np.float64)  # (P,8)

        N = phi_np.shape[0]
        y_pred = np.zeros((N, 5), dtype=np.float64)

        # Loop over unique kinematic points
        for p, (t, xb, q2) in enumerate(uniq_np):
            m = pid_np == p
            if not np.any(m):
                continue
            ph = phi_np[m]
            v = cffs_np[p]
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
                phi_rad=ph,
                k_beam=float(k_beam),
                q_squared=float(q2),
                xb=float(xb),
                t=float(t),
                cffs=cffs,
                using_ww=bool(using_ww),
            )
            y_pred[m, 0] = obs["xs"]
            y_pred[m, 1] = obs["bsa"]
            y_pred[m, 2] = obs["bca"]
            y_pred[m, 3] = obs["tsa"]
            y_pred[m, 4] = obs["dsa"]

        y_pred_t = torch.tensor(y_pred, dtype=theta_per_point.dtype, device=theta_per_point.device)

        # Save for backward
        ctx.save_for_backward(theta_per_point, unique_kin, point_id, phi, y_pred_t)
        ctx.fd_eps = float(fd_eps)
        ctx.k_beam = float(k_beam)
        ctx.using_ww = bool(using_ww)
        ctx.cff_scales = cff_scales_t

        return y_pred_t

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        theta_per_point, unique_kin, point_id, phi, y0 = ctx.saved_tensors
        fd_eps = float(ctx.fd_eps)
        k_beam = float(ctx.k_beam)
        using_ww = bool(ctx.using_ww)
        cff_scales_t = ctx.cff_scales

        # Grad wrt theta_per_point only
        grad_theta = torch.zeros_like(theta_per_point)

        theta_np = theta_per_point.detach().cpu().numpy().astype(np.float64)  # (P,8)
        uniq_np = unique_kin.detach().cpu().numpy().astype(np.float64)
        pid_np = point_id.detach().cpu().numpy().astype(np.int64)
        phi_np = phi.detach().cpu().numpy().astype(np.float64)
        g_np = grad_output.detach().cpu().numpy().astype(np.float64)          # (N,5)

        P = theta_np.shape[0]

        for p in range(P):
            # Only compute FD for points that appear
            m = pid_np == p
            if not np.any(m):
                continue

            t, xb, q2 = uniq_np[p]
            ph = phi_np[m]

            # Base physical cffs for this point
            cffs0 = (theta_np[p] * CFF_SCALES).astype(np.float64)

            # For each parameter j, finite-difference the 5-observable curve at masked points
            for j in range(8):
                eps_j = fd_eps * (abs(theta_np[p, j]) + 1.0)

                cffs_p = cffs0.copy(); cffs_p[j] += eps_j * CFF_SCALES[j]
                cffs_m = cffs0.copy(); cffs_m[j] -= eps_j * CFF_SCALES[j]

                def _obs_from_vec(v):
                    cffs = {
                        "re_h": float(v[0]), "re_e": float(v[1]), "re_ht": float(v[2]), "re_et": float(v[3]),
                        "im_h": float(v[4]), "im_e": float(v[5]), "im_ht": float(v[6]), "im_et": float(v[7]),
                    }
                    o = compute_observables(
                        phi_rad=ph,
                        k_beam=k_beam,
                        q_squared=float(q2),
                        xb=float(xb),
                        t=float(t),
                        cffs=cffs,
                        using_ww=using_ww,
                    )
                    return np.column_stack([o["xs"], o["bsa"], o["bca"], o["tsa"], o["dsa"]]).astype(np.float64)

                yp = _obs_from_vec(cffs_p)
                ym = _obs_from_vec(cffs_m)
                dy_dphys = (yp - ym) / (2.0 * (eps_j * CFF_SCALES[j]))  # derivative wrt physical cff_j

                # Chain: physical cff_j = theta_j * CFF_SCALES[j]
                # so d/dtheta_j = d/dphys * CFF_SCALES[j]
                dy_dtheta = dy_dphys * CFF_SCALES[j]

                # Accumulate gradient: sum_n sum_k g[n,k] * dy[n,k]/dtheta
                # but only for masked points m
                g = g_np[m]  # (n_m,5)
                grad_theta_pj = float(np.sum(g * dy_dtheta))

                grad_theta[p, j] = grad_theta[p, j] + grad_theta_pj

        return grad_theta, None, None, None, None, None, None


class EmbeddingCFFModel(nn.Module):
    """One learnable 8-vector (theta) per unique kinematic point."""

    def __init__(self, n_points: int, init_theta: np.ndarray):
        super().__init__()
        assert init_theta.shape == (n_points, 8)
        self.theta = nn.Parameter(torch.tensor(init_theta, dtype=DTYPE))

    def forward(self, unique_kin: torch.Tensor, point_id: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        theta = self.theta
        return BKM10AllObsFD.apply(theta, unique_kin, point_id, phi, float(FD_EPS), float(K_BEAM), bool(USING_WW))


def split_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(SPLIT_SEED))
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(np.floor(float(TRAIN_FRACTION) * n))
    n_train = max(1, min(n, n_train))
    return idx[:n_train], idx[n_train:]


def init_theta(n_points: int, truth8: Optional[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    if INIT_MODE == "truth" or INIT_MODE == "truth+jitter":
        if truth8 is None:
            raise FileNotFoundError("TRUTH_JSON not found or missing truth CFFs; can't init from truth")
        theta0 = np.tile((truth8 / CFF_SCALES).reshape(1, 8), (n_points, 1))
        if INIT_MODE == "truth+jitter":
            theta0 = theta0 + rng.normal(0.0, float(TRUTH_JITTER), size=theta0.shape)
        return theta0.astype(np.float64)

    # random
    return rng.uniform(-float(INIT_SCALE), float(INIT_SCALE), size=(n_points, 8)).astype(np.float64)


def max_abs_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    d = np.max(np.abs(y_pred - y_true), axis=0)
    return {OBS_NAMES[i]: float(d[i]) for i in range(len(OBS_NAMES))}


def main() -> None:
    safe_mkdir(OUT_ROOT)
    rep_dir = os.path.join(OUT_ROOT, "replicas")
    safe_mkdir(rep_dir)

    d = load_npz(DATA_NPZ)
    X = d["x"].astype(np.float64)
    y_central = d["y_central"].astype(np.float64)
    y_sigma = d["y_sigma"].astype(np.float64)

    if X.shape[1] != 4:
        raise ValueError(f"Expected x shape (N,4); got {X.shape}")
    if y_central.shape[0] != X.shape[0] or y_central.shape[1] != 5:
        raise ValueError(f"Expected y_central shape (N,5); got {y_central.shape}")
    if y_sigma.shape != y_central.shape:
        raise ValueError(f"Expected y_sigma shape {y_central.shape}; got {y_sigma.shape}")

    uniq, pid = unique_kinematics(X)
    P = uniq.shape[0]

    phi = X[:, 3].astype(np.float64)

    # Train/val split over points (rows)
    train_idx, val_idx = split_indices(X.shape[0])

    # Loss scales
    scales = _compute_scales(y_central)
    loss_fn = ScaleMSELoss(scales=scales, ratio_clip=RATIO_CLIP)

    print("Unique kinematic points P =", P)
    print("LOSS_MODE=", LOSS_MODE)
    print("Scales:", {OBS_NAMES[i]: float(scales[i]) for i in range(5)})

    # Optional truth init
    truth8 = load_truth8(TRUTH_JSON)

    rng_rep = np.random.default_rng(int(REPLICA_SEED))

    for r in range(int(N_REPLICAS)):
        t0 = time.time()

        # Replica pseudo-data: y_rep = y_central + N(0,1)*y_sigma
        noise = rng_rep.normal(0.0, 1.0, size=y_central.shape)
        y_rep = y_central + noise * y_sigma

        y_train = y_rep[train_idx]
        y_val = y_rep[val_idx] if val_idx.size > 0 else None

        # Torch tensors
        uniq_t = torch.tensor(uniq, dtype=DTYPE)
        pid_t = torch.tensor(pid, dtype=torch.long)
        phi_t = torch.tensor(phi, dtype=DTYPE)

        y_train_t = torch.tensor(y_train, dtype=DTYPE)
        y_val_t = torch.tensor(y_val, dtype=DTYPE) if y_val is not None else None

        # Create replica directory early
        out_rep = os.path.join(rep_dir, f"replica_{r+1:03d}")
        safe_mkdir(out_rep)

        best = {"loss": float("inf"), "state": None, "history": None}

        for restart in range(int(N_RESTARTS)):
            rng_init = np.random.default_rng(int(REPLICA_SEED) + 1000 * (r + 1) + restart)
            theta0 = init_theta(P, truth8, rng_init)

            model = EmbeddingCFFModel(n_points=P, init_theta=theta0)
            model.train()

            history = {"loss": [], "val_loss": [], "max_abs": []}

            if OPTIMIZER.lower() == "adam":
                opt = torch.optim.Adam(model.parameters(), lr=float(ADAM_LR))

                best_val = float("inf")
                bad = 0

                for epc in range(int(EPOCHS)):
                    opt.zero_grad(set_to_none=True)
                    y_pred = model(uniq_t, pid_t, phi_t)
                    loss = loss_fn(y_train_t, y_pred[train_idx])
                    loss.backward()

                    if ADAM_CLIPNORM and float(ADAM_CLIPNORM) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(ADAM_CLIPNORM))

                    opt.step()

                    with torch.no_grad():
                        loss_v = float(loss.detach().cpu().item())
                        history["loss"].append(loss_v)

                        if y_val_t is not None:
                            val_pred = y_pred[val_idx]
                            val_loss = float(loss_fn(y_val_t, val_pred).detach().cpu().item())
                        else:
                            val_loss = loss_v
                        history["val_loss"].append(val_loss)

                        # diagnostics in observable space (max abs)
                        maxabs = max_abs_residuals(y_train, y_pred[train_idx].detach().cpu().numpy())
                        history["max_abs"].append(maxabs)

                        if (epc % PRINT_EVERY) == 0 or epc == (EPOCHS - 1):
                            print(f"[replica {r+1:03d} restart {restart}] epoch {epc:04d} loss={loss_v:.6g} val={val_loss:.6g} maxabs={maxabs}")

                        if val_loss < best_val - 1e-12:
                            best_val = val_loss
                            bad = 0
                        else:
                            bad += 1
                            if bad >= int(PATIENCE):
                                break

            else:
                # LBFGS
                opt = torch.optim.LBFGS(
                    model.parameters(),
                    lr=float(LBFGS_LR),
                    max_iter=int(LBFGS_MAX_ITER_PER_STEP),
                    history_size=int(LBFGS_HISTORY_SIZE),
                    tolerance_grad=float(LBFGS_TOL_GRAD),
                    tolerance_change=float(LBFGS_TOL_CHANGE),
                    line_search_fn="strong_wolfe",
                )

                for step in range(int(LBFGS_STEPS)):

                    def closure():
                        opt.zero_grad(set_to_none=True)
                        y_pred = model(uniq_t, pid_t, phi_t)
                        loss = loss_fn(y_train_t, y_pred[train_idx])
                        loss.backward()
                        return loss

                    loss = opt.step(closure)

                    with torch.no_grad():
                        y_pred = model(uniq_t, pid_t, phi_t)
                        loss_v = float(loss.detach().cpu().item())
                        history["loss"].append(loss_v)

                        if y_val_t is not None:
                            val_loss = float(loss_fn(y_val_t, y_pred[val_idx]).detach().cpu().item())
                        else:
                            val_loss = loss_v
                        history["val_loss"].append(val_loss)

                        maxabs = max_abs_residuals(y_train, y_pred[train_idx].detach().cpu().numpy())
                        history["max_abs"].append(maxabs)

                        if (step % PRINT_EVERY) == 0 or step == (LBFGS_STEPS - 1):
                            print(f"[replica {r+1:03d} restart {restart}] step {step:04d} loss={loss_v:.6g} val={val_loss:.6g} maxabs={maxabs}")

            # Track best restart by final val_loss
            final_val = float(history["val_loss"][-1]) if history["val_loss"] else float("inf")
            if final_val < best["loss"]:
                best["loss"] = final_val
                best["state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best["history"] = history

        # Restore best
        model = EmbeddingCFFModel(n_points=P, init_theta=np.zeros((P, 8), dtype=np.float64))
        model.load_state_dict(best["state"], strict=True)
        model.eval()

        with torch.no_grad():
            theta_hat = model.theta.detach().cpu().numpy().astype(np.float64)
        cffs_hat = theta_hat * np.asarray(CFF_SCALES, dtype=np.float64)[None, :]

        # Save outputs (avoid numpy arrays in model.pt so torch.load(weights_only=True) works)
        torch.save(
            {
                "mode": "embedding_scaled_float64",
                "unique_kin": torch.tensor(uniq, dtype=torch.float64),
                "cffs": torch.tensor(cffs_hat, dtype=torch.float64),
                "cff_names": CFF_NAMES,
                "obs_names": OBS_NAMES,
                "loss_mode": str(LOSS_MODE),
                "optimizer": str(OPTIMIZER),
                "cff_scales": [float(x) for x in CFF_SCALES.tolist()],
                "fd_eps": float(FD_EPS),
                "best_val": float(best["loss"]),
                "init_mode": str(INIT_MODE),
                "n_restarts": int(N_RESTARTS),
            },
            os.path.join(out_rep, "model.pt"),
        )

        np.savez(
            os.path.join(out_rep, "extracted_cffs_per_point.npz"),
            unique_kin=uniq.astype(np.float64),
            cffs=cffs_hat.astype(np.float64),
            cff_names=np.array(CFF_NAMES),
        )
        save_json(os.path.join(out_rep, "history.json"), best["history"] or {})

        dt = time.time() - t0
        v = cffs_hat[0]
        msg = f"[replica {r+1:03d}] done in {dt:.1f}s (best_val={best['loss']:.6g}): " + ", ".join([f"{CFF_NAMES[i]}={v[i]:.6g}" for i in range(8)])
        print(msg)

    print("Saved replicas to:", rep_dir)


if __name__ == "__main__":
    main()
