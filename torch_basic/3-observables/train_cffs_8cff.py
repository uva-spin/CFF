#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Training Example using three observables xs, bsa, and bca

Train an ensemble of replica fits to extract *all eight* CFF components at
fixed- or multi-kinematics, using three observables per point:

  - XS(phi)  : unpolarized (helicity-averaged) cross section
  - BSA(phi) : beam-spin asymmetry
  - BCA(phi) : beam-charge asymmetry

This script is designed to be:
  - GPU-friendly: forward model is pure torch ops (autograd works)
  - Adaptable: simple config-at-top, no CLI required
  - Compatible with the closure generator:
        generate_closure_dataset_torch_xsbsa_bca.py

Outputs (for each replica)
--------------------------
<OUT_ROOT>/replicas/replica_XXX/
  model.pt
  extracted_cffs_per_point.npz    (canonical for evaluators)
  history.json
  config.json

"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bkm10_torch_forward import BKM10Config, BKM10Forward

# ============================================================
# CONFIG (edit here)
# ============================================================

DATA_NPZ = "output/data/dataset_v_1.npz"
OUT_ROOT = "output_torch_8cff"

# Replica ensemble
N_REPLICAS = 50
REPLICA_SEED = 2222
USE_REPLICA_NOISE = True

# Training split
TRAIN_FRACTION = 0.8
SPLIT_SEED = 42
SPLIT_BY_POINT = True  # for fixed-kin, doesn't matter

# Model mode:
#   "embedding" -> one learnable 8-vector per unique (t,xB,Q2)
#   "mlp"       -> MLP mapping (t,xB,Q2)->8 (useful for multi-kin)
MODE = "embedding"

# Optimizer/training
EPOCHS = 4000
PATIENCE = 400
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.0

BATCH_SIZE = 0  # 0 => full batch

# AMP on CUDA
AMP = True

# Gradient clipping
CLIP_NORM = 5.0  # 0 to disable

# Residual clipping in units of sigma_soft (post-division)
RATIO_CLIP = 1e4  # 0/None to disable

# ---- Soft-chi2 floors (weighting only) ----
USE_POINTWISE_SIGMAS = True

SOFT_XS_REL = 0.02
SOFT_XS_ABS = 0.0

SOFT_BSA_REL = 0.0
SOFT_BSA_ABS = 0.01

SOFT_BCA_REL = 0.0
SOFT_BCA_ABS = 0.02

# CFF parameter scaling (helps conditioning; model outputs are scaled)
# Order: [ReH, ReE, ReHtilde, ReEtilde, ImH, ImE, ImHtilde, ImEtilde]
CFF_SCALES = np.array([1.0, 3.0, 3.0, 200.0, 1.0, 3.0, 3.0, 200.0], dtype=float)

# Physics forward settings (must match generator truth)
K_BEAM = 5.75
USING_WW = True
TARGET_POLARIZATION = 0.0
EPS = 1e-12

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


def load_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ not found: {path}")
    d = np.load(path, allow_pickle=True)
    req = {"x", "y_central", "y_sigma"}
    if not req.issubset(set(d.files)):
        raise KeyError(f"NPZ missing keys {req}; found {d.files}")
    return {k: d[k] for k in d.files}


def soft_sigmas(y_central: np.ndarray, y_sigma: np.ndarray) -> np.ndarray:
    """Construct per-point sigma_soft for (XS,BSA,BCA)."""
    y_central = np.asarray(y_central, dtype=float)
    y_sigma = np.asarray(y_sigma, dtype=float)

    xs = y_central[:, 0]
    bsa = y_central[:, 1]
    bca = y_central[:, 2]

    xs_sig = y_sigma[:, 0]
    bsa_sig = y_sigma[:, 1]
    bca_sig = y_sigma[:, 2]

    def _robust_scale(v: np.ndarray) -> float:
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return 1.0
        s = float(np.median(np.abs(v)))
        if s <= 1e-12:
            s = float(np.percentile(np.abs(v), 90)) if v.size else 1.0
        return s if s > 0 else 1.0

    xs_scale = _robust_scale(xs)
    bsa_scale = _robust_scale(bsa)
    bca_scale = _robust_scale(bca)

    if not bool(USE_POINTWISE_SIGMAS):
        xs_sig = np.zeros_like(xs_sig)
        bsa_sig = np.zeros_like(bsa_sig)
        bca_sig = np.zeros_like(bca_sig)

    xs_floor = float(SOFT_XS_REL) * xs_scale
    bsa_floor = float(SOFT_BSA_REL) * bsa_scale
    bca_floor = float(SOFT_BCA_REL) * bca_scale

    xs_soft = np.sqrt(xs_sig**2 + xs_floor**2 + float(SOFT_XS_ABS)**2)
    bsa_soft = np.sqrt(bsa_sig**2 + bsa_floor**2 + float(SOFT_BSA_ABS)**2)
    bca_soft = np.sqrt(bca_sig**2 + bca_floor**2 + float(SOFT_BCA_ABS)**2)

    xs_soft = np.where(xs_soft > 0, xs_soft, 1.0)
    bsa_soft = np.where(bsa_soft > 0, bsa_soft, 1.0)
    bca_soft = np.where(bca_soft > 0, bca_soft, 1.0)

    return np.column_stack([xs_soft, bsa_soft, bca_soft]).astype(np.float32)


class CFFEmbeddingModel(nn.Module):
    """One trainable 8-vector per unique kinematic point."""

    def __init__(self, n_points: int, cff_scales: np.ndarray, seed: int = 0):
        super().__init__()
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        self.register_buffer("cff_scales", torch.tensor(cff_scales, dtype=torch.float32))
        # Parameter table in *scaled* space
        init = 0.05 * torch.randn((n_points, 8), generator=g)
        self.cffs_scaled = nn.Parameter(init)

    def forward(self, point_id: torch.Tensor) -> torch.Tensor:
        # point_id: (N,) int64
        c = self.cffs_scaled[point_id]  # (N,8)
        return c * self.cff_scales

    def cffs_table(self) -> torch.Tensor:
        return self.cffs_scaled * self.cff_scales


class CFFMLP(nn.Module):
    """MLP mapping (t,xB,Q2) -> 8 CFFs."""

    def __init__(self, cff_scales: np.ndarray, seed: int = 0, width: int = 64, depth: int = 3):
        super().__init__()
        torch.manual_seed(int(seed))
        self.register_buffer("cff_scales", torch.tensor(cff_scales, dtype=torch.float32))

        layers = []
        in_dim = 3
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_dim, int(width)))
            layers.append(nn.ReLU())
            in_dim = int(width)
        layers.append(nn.Linear(in_dim, 8))
        self.net = nn.Sequential(*layers)

    def forward(self, kin3: torch.Tensor) -> torch.Tensor:
        # kin3: (N,3)
        out = self.net(kin3)
        return out * self.cff_scales


def build_unique_kin(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (unique_kin (P,3), point_id (N,))."""
    kin = np.asarray(X[:, :3], dtype=float)
    # Use a robust exact-match by rounding (avoid float32 jitter)
    kin_key = np.round(kin, decimals=8)
    uniq, inv = np.unique(kin_key, axis=0, return_inverse=True)
    return uniq.astype(np.float64), inv.astype(np.int64)


def make_splits(point_id: np.ndarray, n_points: int, n_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(SPLIT_SEED))
    if bool(SPLIT_BY_POINT):
        pts = np.arange(n_points)
        rng.shuffle(pts)
        n_tr = int(np.floor(float(TRAIN_FRACTION) * n_points))
        n_tr = max(1, min(n_points, n_tr))
        tr_pts = set(pts[:n_tr].tolist())
        tr_mask = np.array([pid in tr_pts for pid in point_id], dtype=bool)
        va_mask = ~tr_mask
        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]
    else:
        idx = np.arange(n_rows)
        rng.shuffle(idx)
        n_tr = int(np.floor(float(TRAIN_FRACTION) * n_rows))
        n_tr = max(1, min(n_rows, n_tr))
        tr_idx = idx[:n_tr]
        va_idx = idx[n_tr:]
    return tr_idx.astype(np.int64), va_idx.astype(np.int64)


def train_one_replica(
    replica_id: int,
    seed: int,
    X: np.ndarray,
    y_central: np.ndarray,
    y_sigma: np.ndarray,
    sigma_soft: np.ndarray,
    unique_kin: np.ndarray,
    point_id: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    rep_dir = os.path.join(OUT_ROOT, "replicas", f"replica_{replica_id:03d}")
    safe_mkdir(rep_dir)

    # Replica sampling
    rng = np.random.default_rng(int(seed))
    if bool(USE_REPLICA_NOISE):
        noise = rng.normal(0.0, 1.0, size=y_central.shape).astype(np.float32)
        y_rep = y_central + noise * y_sigma
    else:
        y_rep = y_central.copy()

    # Pack y_true: [XS_obs, BSA_obs, BCA_obs, XS_sigma_soft, BSA_sigma_soft, BCA_sigma_soft]
    y_pack = np.concatenate([y_rep, sigma_soft], axis=1).astype(np.float32)

    X_t = torch.tensor(X, dtype=dtype)
    y_t = torch.tensor(y_pack, dtype=dtype)
    pid_t = torch.tensor(point_id, dtype=torch.long)

    # Dataloaders
    tr_ds = TensorDataset(X_t[train_idx], y_t[train_idx], pid_t[train_idx])
    va_ds = TensorDataset(X_t[val_idx], y_t[val_idx], pid_t[val_idx]) if len(val_idx) else None

    bs = int(BATCH_SIZE)
    if bs <= 0:
        bs = len(train_idx)

    dl_tr = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    dl_va = DataLoader(va_ds, batch_size=min(bs, len(val_idx)), shuffle=False) if va_ds else None

    # Model
    if str(MODE).lower() == "embedding":
        model: nn.Module = CFFEmbeddingModel(n_points=unique_kin.shape[0], cff_scales=CFF_SCALES, seed=seed)
        kin_input_mode = "point_id"
    elif str(MODE).lower() == "mlp":
        model = CFFMLP(cff_scales=CFF_SCALES, seed=seed)
        kin_input_mode = "kin3"
    else:
        raise ValueError("MODE must be 'embedding' or 'mlp'")

    model.to(device=device, dtype=dtype)

    # Physics forward
    fwd = BKM10Forward(BKM10Config(k_beam=float(K_BEAM), using_ww=bool(USING_WW), target_polarization=float(TARGET_POLARIZATION), eps=float(EPS)), dtype=dtype)
    fwd.to(device=device)

    opt = torch.optim.Adam(model.parameters(), lr=float(LEARNING_RATE), weight_decay=float(WEIGHT_DECAY))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(AMP) and device.type == "cuda")

    ratio_clip = float(RATIO_CLIP) if (RATIO_CLIP and float(RATIO_CLIP) > 0) else None

    def loss_batch(batch) -> torch.Tensor:
        x_b, y_b, pid_b = batch
        x_b = x_b.to(device=device)
        y_b = y_b.to(device=device)
        pid_b = pid_b.to(device=device)

        # y_true unpack
        xs_obs = y_b[:, 0]
        bsa_obs = y_b[:, 1]
        bca_obs = y_b[:, 2]
        xs_sig = y_b[:, 3]
        bsa_sig = y_b[:, 4]
        bca_sig = y_b[:, 5]

        # kinematics
        t = x_b[:, 0]
        xb = x_b[:, 1]
        q2 = x_b[:, 2]
        phi = x_b[:, 3]

        # predict CFFs
        if kin_input_mode == "point_id":
            cffs = model(pid_b)
        else:
            kin3 = x_b[:, :3]
            cffs = model(kin3)

        xs_pred, bsa_pred, bca_pred = fwd.forward_xs_bsa_bca(t, xb, q2, phi, cffs)

        # residuals
        rx = (xs_obs - xs_pred) / xs_sig
        rbsa = (bsa_obs - bsa_pred) / bsa_sig
        rbca = (bca_obs - bca_pred) / bca_sig

        if ratio_clip is not None:
            c = float(ratio_clip)
            rx = torch.clamp(rx, -c, c)
            rbsa = torch.clamp(rbsa, -c, c)
            rbca = torch.clamp(rbca, -c, c)

        # average of three soft-chi2 terms
        return 0.5 * (rx.pow(2).mean() + rbsa.pow(2).mean() + rbca.pow(2).mean()) / 3.0

    # Training loop
    best = float("inf")
    best_state = None
    bad = 0
    hist = {"loss": [], "val_loss": []}

    t0 = time.time()
    for epoch in range(int(EPOCHS)):
        model.train()
        run = 0.0
        seen = 0

        for batch in dl_tr:
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(AMP) and device.type == "cuda"):
                loss = loss_batch(batch)

            scaler.scale(loss).backward()

            if CLIP_NORM and float(CLIP_NORM) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(CLIP_NORM))

            scaler.step(opt)
            scaler.update()

            bs_now = int(batch[0].shape[0])
            run += float(loss.detach().cpu().item()) * bs_now
            seen += bs_now

        tr_loss = run / max(1, seen)

        # validation
        if dl_va is not None:
            model.eval()
            with torch.no_grad():
                vrun = 0.0
                vseen = 0
                for batch in dl_va:
                    with torch.cuda.amp.autocast(enabled=bool(AMP) and device.type == "cuda"):
                        vloss = loss_batch(batch)
                    bs_now = int(batch[0].shape[0])
                    vrun += float(vloss.detach().cpu().item()) * bs_now
                    vseen += bs_now
                va_loss = vrun / max(1, vseen)
        else:
            va_loss = tr_loss

        hist["loss"].append(float(tr_loss))
        hist["val_loss"].append(float(va_loss))

        if float(va_loss) < best - 1e-12:
            best = float(va_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= int(PATIENCE):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Extract per-point CFF table (P,8)
    model.eval()
    with torch.no_grad():
        if kin_input_mode == "point_id":
            cffs_table = model.cffs_table().detach().cpu().numpy().astype(float)
        else:
            kin3 = torch.tensor(unique_kin, dtype=dtype, device=device)
            cffs_table = model(kin3).detach().cpu().numpy().astype(float)

    # Save canonical NPZ for evaluator
    np.savez(os.path.join(rep_dir, "extracted_cffs_per_point.npz"),
             unique_kin=unique_kin,
             cffs=cffs_table)

    # Save model checkpoint
    torch.save({
        "state_dict": model.state_dict(),
        "mode": str(MODE),
        "unique_kin": unique_kin,
        "cffs": cffs_table,
        "cff_names": CFF_NAMES,
        "obs_names": OBS_NAMES,
    }, os.path.join(rep_dir, "model.pt"))

    # Save config + history
    save_json(os.path.join(rep_dir, "history.json"), hist)
    save_json(os.path.join(rep_dir, "config.json"), {
        "replica_id": int(replica_id),
        "seed": int(seed),
        "USE_REPLICA_NOISE": bool(USE_REPLICA_NOISE),
        "MODE": str(MODE),
        "physics": {
            "k_beam": float(K_BEAM),
            "using_ww": bool(USING_WW),
            "target_polarization": float(TARGET_POLARIZATION),
            "eps": float(EPS),
        },
        "CFF_SCALES": list(map(float, CFF_SCALES)),
        "TRAIN_FRACTION": float(TRAIN_FRACTION),
        "SPLIT_BY_POINT": bool(SPLIT_BY_POINT),
        "OBS": OBS_NAMES,
    })

    dt = time.time() - t0
    if unique_kin.shape[0] == 1:
        msg = ", ".join([f"{CFF_NAMES[i]}={cffs_table[0,i]:.6g}" for i in range(8)])
        print(f"[replica {replica_id:03d}] done in {dt:.1f}s: {msg}")
    else:
        print(f"[replica {replica_id:03d}] done in {dt:.1f}s: extracted {unique_kin.shape[0]} point(s)")


def main() -> None:
    safe_mkdir(OUT_ROOT)
    safe_mkdir(os.path.join(OUT_ROOT, "replicas"))

    d = load_npz(DATA_NPZ)
    X = np.asarray(d["x"], dtype=np.float32)
    y_central = np.asarray(d["y_central"], dtype=np.float32)
    y_sigma = np.asarray(d["y_sigma"], dtype=np.float32)

    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"X must be (N,4); got {X.shape}")
    if y_central.ndim != 2 or y_central.shape[1] != 3:
        raise ValueError(f"y_central must be (N,3) for (XS,BSA,BCA); got {y_central.shape}")
    if y_sigma.shape != y_central.shape:
        raise ValueError(f"y_sigma shape mismatch: {y_sigma.shape} vs {y_central.shape}")

    N = X.shape[0]

    # Build unique kinematics mapping
    unique_kin, point_id = build_unique_kin(X)
    P = unique_kin.shape[0]

    # Weighting only
    sigma_soft = soft_sigmas(y_central, y_sigma)

    print("Dataset summary:")
    print(f"  rows N={N}, unique points P={P}")
    print("  sigma_soft medians:",
          float(np.median(sigma_soft[:, 0])),
          float(np.median(sigma_soft[:, 1])),
          float(np.median(sigma_soft[:, 2])))
    print("  USE_POINTWISE_SIGMAS:", bool(USE_POINTWISE_SIGMAS))

    # Train/val split
    tr_idx, va_idx = make_splits(point_id=point_id, n_points=P, n_rows=N)

    # Replica loop
    for r in range(int(N_REPLICAS)):
        seed = int(REPLICA_SEED) + 1000 * (r + 1)
        train_one_replica(
            replica_id=r + 1,
            seed=seed,
            X=X,
            y_central=y_central,
            y_sigma=y_sigma,
            sigma_soft=sigma_soft,
            unique_kin=unique_kin,
            point_id=point_id,
            train_idx=tr_idx,
            val_idx=va_idx,
        )

    print("\nSaved replicas to:", os.path.join(OUT_ROOT, "replicas"))


if __name__ == "__main__":
    main()
