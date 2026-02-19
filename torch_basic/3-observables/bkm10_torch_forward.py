
"""
bkm10_torch_forward.py

A thin, *fully differentiable* PyTorch wrapper around the BKM10-style
cross-section/BSA formulas (ported from `cross_section_script.py`).

This file focuses on *batched* evaluation for training:
  - supports per-row kinematics (t, xB, Q2, phi) and per-row CFFs
  - stays in pure torch ops => runs on GPU and supports autograd

It assumes:
  - phi is in radians
  - k_beam is either a scalar (float) or a tensor broadcastable to (N,)
  - unpolarized target and BSA built from +/- beam helicity

You can extend the wrapper to additional observables (TSA, DSA, etc.)
by calling `bkm10_cross_section` with the desired (lep_helicity, target_polar).
All that is from the bmk10 calls is from Dimas original code.  Give him credit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import torch

# Import the torch-port of the BKM10 algebra (functions are pure torch)
import bkm10_torch as bkm


TensorLike = Union[torch.Tensor, float, int]


def _to_tensor(x: TensorLike, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


@dataclass(frozen=True)
class BKM10Config:
    """Forward-model configuration knobs."""
    k_beam: float = 5.75
    target_polarization: float = 0.0
    using_ww: bool = True
    # tiny eps for safe divisions in derived observables (e.g. BSA denominator)
    eps: float = 1e-12


class BKM10Forward(torch.nn.Module):
    """
    Fully differentiable forward model: (t, xB, Q2, phi, CFFs) -> (XS, BSA)

    Inputs (batched):
      t, xB, Q2, phi: shape (N,)
      cffs: shape (N, 8) ordered as:
        [ReH, ReE, ReHtilde, ReEtilde, ImH, ImE, ImHtilde, ImEtilde]

    Outputs:
      xs:  (N,)
      bsa: (N,)

    Notes on polarization conventions
    -------------------------------
    The returned BSA is an *analyzing power* computed from the theoretical
    helicity-dependent cross sections.

    It corresponds to 100% beam polarization. If you want to model a measured
    asymmetry with finite beam polarization P_b, you would typically use:
        BSA_meas ≈ P_b * BSA
    """

    def __init__(self, cfg: BKM10Config = BKM10Config(), *, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype

    def _common_kinematics_and_cffs(
        self,
        t: torch.Tensor,
        xb: torch.Tensor,
        q2: torch.Tensor,
        phi: torch.Tensor,
        cffs: torch.Tensor,
    ):
        """Compute all common tensors needed for observables."""
        device = t.device
        dtype = self.dtype

        t = t.to(dtype=dtype)
        xb = xb.to(dtype=dtype)
        q2 = q2.to(dtype=dtype)
        phi = phi.to(dtype=dtype)
        cffs = cffs.to(dtype=dtype)

        if self.training:
            self._check_shapes(t, xb, q2, phi, cffs)

        k_beam = _to_tensor(self.cfg.k_beam, device=device, dtype=dtype)

        ep = bkm.compute_epsilon(xb, q2)
        y = bkm.compute_y(k_beam, q2, ep)
        xi = bkm.compute_skewness(xb, t, q2)

        tmin = bkm.compute_t_min(xb, q2, ep)
        tprime = bkm.compute_t_prime(t, tmin)

        # Clamp small negative numerical noise in (tmin - t)
        tmin_minus_t = torch.clamp(tmin - t, min=0.0)
        ktilde = torch.sqrt(tmin_minus_t) * torch.sqrt(
            (1.0 - xb) * torch.sqrt(1.0 + ep ** 2)
            + (tmin_minus_t) * (ep ** 2 + 4.0 * (1.0 - xb) * xb) / (4.0 * q2)
        )

        k = bkm.compute_k(q2, y, ep, ktilde)

        fe = bkm.compute_fe(t)
        fg = bkm.compute_fg(fe)
        f2 = bkm.compute_f2(t, fe, fg)
        f1 = bkm.compute_f1(fg, f2)

        kdd = bkm.compute_k_dot_delta(q2, xb, t, phi, ep, y, k)
        p1 = bkm.prop_1(q2, kdd)
        p2 = bkm.prop_2(q2, t, kdd)

        re_h, re_e, re_ht, re_et, im_h, im_e, im_ht, im_et = torch.unbind(cffs, dim=1)

        return (
            t, xb, q2, phi,
            ep, y, xi, k,
            f1, f2, ktilde, tprime,
            p1, p2,
            re_h, re_ht, re_e, re_et,
            im_h, im_ht, im_e, im_et,
        )

    @torch.no_grad()
    def _check_shapes(self, t, xb, q2, phi, cffs):
        if t.ndim != 1 or xb.ndim != 1 or q2.ndim != 1 or phi.ndim != 1:
            raise ValueError("t, xb, q2, phi must be 1D tensors of shape (N,).")
        if not (t.shape == xb.shape == q2.shape == phi.shape):
            raise ValueError(f"Shape mismatch: t{t.shape}, xb{xb.shape}, q2{q2.shape}, phi{phi.shape}")
        if cffs.ndim != 2 or cffs.shape[0] != t.shape[0] or cffs.shape[1] != 8:
            raise ValueError(f"cffs must have shape (N,8); got {tuple(cffs.shape)}")

    def forward(
        self,
        t: torch.Tensor,
        xb: torch.Tensor,
        q2: torch.Tensor,
        phi: torch.Tensor,
        cffs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            t, xb, q2, phi,
            ep, y, xi, k,
            f1, f2, ktilde, tprime,
            p1, p2,
            re_h, re_ht, re_e, re_et,
            im_h, im_ht, im_e, im_et,
        ) = self._common_kinematics_and_cffs(t, xb, q2, phi, cffs)

        # --------
        # Observables
        # --------
        xs = bkm.bkm10_cross_section(
            0.0,  # unpolarized beam XS (average of +/- helicity in the original code)
            float(self.cfg.target_polarization),
            q2, xb, t,
            ep, y, xi, k,
            f1, f2, ktilde, tprime, phi,
            p1, p2,
            re_h, re_ht, re_e, re_et,
            im_h, im_ht, im_e, im_et,
            use_ww=bool(self.cfg.using_ww),
        )

        bsa = bkm.bkm10_bsa(
            0.0,  # ignored by bkm10_bsa; it explicitly computes +/- helicity
            float(self.cfg.target_polarization),
            q2, xb, t,
            ep, y, xi, k,
            f1, f2, ktilde, tprime, phi,
            p1, p2,
            re_h, re_ht, re_e, re_et,
            im_h, im_ht, im_e, im_et,
            use_ww=bool(self.cfg.using_ww),
        )

        # Optional safety: avoid NaNs in rare near-zero denominators
        if self.cfg.eps and float(self.cfg.eps) > 0:
            bsa = torch.nan_to_num(bsa, nan=0.0, posinf=0.0, neginf=0.0)

        return xs, bsa

    def forward_xs_bsa_bca(
        self,
        t: torch.Tensor,
        xb: torch.Tensor,
        q2: torch.Tensor,
        phi: torch.Tensor,
        cffs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (XS, BSA, BCA) for each row."""
        (
            t, xb, q2, phi,
            ep, y, xi, k,
            f1, f2, ktilde, tprime,
            p1, p2,
            re_h, re_ht, re_e, re_et,
            im_h, im_ht, im_e, im_et,
        ) = self._common_kinematics_and_cffs(t, xb, q2, phi, cffs)

        xs, bsa, bca = bkm.bkm10_xs_bsa_bca(
            float(self.cfg.target_polarization),
            q2, xb, t,
            ep, y, xi, k,
            f1, f2,
            ktilde, tprime,
            phi,
            p1, p2,
            re_h, re_ht, re_e, re_et,
            im_h, im_ht, im_e, im_et,
            use_ww=bool(self.cfg.using_ww),
            eps=float(self.cfg.eps) if self.cfg.eps else 0.0,
        )

        if self.cfg.eps and float(self.cfg.eps) > 0:
            bsa = torch.nan_to_num(bsa, nan=0.0, posinf=0.0, neginf=0.0)
            bca = torch.nan_to_num(bca, nan=0.0, posinf=0.0, neginf=0.0)

        return xs, bsa, bca

