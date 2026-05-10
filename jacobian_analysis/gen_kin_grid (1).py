#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_kin_grid.py

Generate a JSON file listing all kinematic points to be analyzed.
Each entry is a dict with keys: t, xB, Q2, and truth_cffs (8-vector).

Run once on the login node before submitting the array job:

    python gen_kin_grid.py

Outputs:
    kin_grid.json:  list of dicts, one per kinematic point
    (printed) total number of points (which needs to be input in the slurm script)
"""

import json
import itertools
import numpy as np

# ============================================================
# GRID CONFIG
# ============================================================

# xB values
XB_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.36, 0.42, 0.50, 0.58]

# |t| values in GeV^2
ABS_T_VALUES = [0.10, 0.17, 0.25, 0.35, 0.45, 0.58]

# clanker suggestion
def q2_from_xb(xB: float) -> float:
    """Q2 ~ 3.5*xB (GeV^2), clamped to [1.0, 6.0]. Approximate JLab acceptance."""
    return float(np.clip(3.5 * xB, 1.0, 6.0))

# Validity cuts; this is also something the clanker suggested, since apparently the twist-2 expansion holds for |t| << Q2 (look this up)
T_MAX_FRACTION_OF_Q2 = 0.7
XB_MIN = 0.08
XB_MAX = 0.65

# ============================================================
# TRUTH CFF GENERATING FUNCTION
# ============================================================
# CFF order: [ReH, ReE, ReHtilde, ReEtilde, ImH, ImE, ImHtilde, ImEtilde]

# clanker suggestion
def _regge(xB: float, t: float, N: float, alpha: float, beta: float, b: float) -> float:
    """
    Regge/power-law CFF component:
        F(xB, t) = N * xB^(-alpha) * (1 - xB)^beta * exp(b * t)

    Parameters
    ----------
    alpha : Regge intercept
    beta  : Large-xB suppression
    b     : t-slope in GeV^-2
    t     : Must be negative
    """
    return float(N * xB**(-alpha) * (1.0 - xB)**beta * np.exp(b * t))

# Deermined by clanker fit to reproduce Dustin's "truth" point (xB=0.34, t=-0.17, Q2=1.82) -> CFFs = [1.0, 2.22, 1.41, 144.4, 1.0, 0.0, 1.58, 0.0]
#                  N        alpha  beta    b (GeV^-2)
_CFF_REGGE_PARAMS = {
    "ImH":  (  3.124,  0.48,   3.5,   1.2),   # calibrated: ImH(ref) = 1.000
    "ReH":  (  3.124,  0.48,   3.5,   1.2),   # Re ~ Im at JLab kinematics (rough)
    "ImHt": (  4.927,  0.48,   3.5,   1.2),   # calibrated: ImHt(ref) = 1.577
    "ReHt": (  4.253,  0.48,   3.5,   1.0),   # calibrated: ReHt(ref) = 1.409
    "ImE":  (  0.00,   0.50,   4.0,   2.0),   # set to 0; truth ~ 0 at JLab
    "ReE":  (  7.033,  0.40,   3.0,   2.0),   # calibrated: ReE(ref) = 2.217
    "ImEt": (  0.00,   0.50,   4.0,   2.0),   # set to 0; truth ~ 0 at JLab
    "ReEt": (291.0,    0.20,   2.0,   0.5),   # calibrated: ReEt(ref) = 144.4
}

# At the reference point (xB=0.34, t=-0.17), the above produces the following:
#   ReH=1.03  ReE=3.0  ReHt=1.35  ReEt=145
#   ImH=1.03  ImE=0.09 ImHt=1.60  ImEt=0.0
# The "truth" values are [1.0, 2.22, 1.41, 144.4, 1.0, 0.0, 1.58, 0.0]

def truth_cffs_at_point(xB: float, t: float, Q2: float) -> list:
    """
    Return [ReH, ReE, ReHtilde, ReEtilde, ImH, ImE, ImHtilde, ImEtilde]
    using the Regge/power-law parametrization above
    """
    p = _CFF_REGGE_PARAMS
    return [
        _regge(xB, t, *p["ReH"]),
        _regge(xB, t, *p["ReE"]),
        _regge(xB, t, *p["ReHt"]),
        _regge(xB, t, *p["ReEt"]),
        _regge(xB, t, *p["ImH"]),
        _regge(xB, t, *p["ImE"]),
        _regge(xB, t, *p["ImHt"]),
        _regge(xB, t, *p["ImEt"]),
    ]

def validate_at_reference() -> None:
    """Prints comparison between generated CFFs and "truth" CFFs """
    xB_ref, t_ref, Q2_ref = 0.34, -0.17, 1.82
    v = truth_cffs_at_point(xB_ref, t_ref, Q2_ref)
    names = ["ReH", "ReE", "ReHt", "ReEt", "ImH", "ImE", "ImHt", "ImEt"]
    truth  = [1.0, 2.2173543720, 1.4093937265, 144.410164202,
              1.0, 0.0, 1.5773644026, 0.0]
    print(f"\nValidation at reference point (xB={xB_ref}, t={t_ref}, Q2={Q2_ref}):")
    print(f"  {'CFF':8s}  {'Regge model':>14s}  {'generate.py truth':>18s}  {'ratio':>8s}")
    for name, val, tru in zip(names, v, truth):
        ratio = val / tru if abs(tru) > 1e-6 else float("nan")
        print(f"  {name:8s}  {val:14.4f}  {tru:18.4f}  {ratio:8.3f}")
    print()

# ============================================================
# MAIN
# ============================================================

def main():
    validate_at_reference()
    points = []

    for xB, abs_t in itertools.product(XB_VALUES, ABS_T_VALUES):
        Q2 = q2_from_xb(xB)
        t = -abs_t
        if xB < XB_MIN or xB > XB_MAX:
            continue
        if abs_t > T_MAX_FRACTION_OF_Q2 * Q2:
            continue
        points.append({
            "t":  round(float(t),  6),
            "xB": round(float(xB), 6),
            "Q2": round(float(Q2), 6),
            "truth_cffs": truth_cffs_at_point(xB, t, Q2),
        })

    out = "kin_grid.json"
    with open(out, "w") as f:
        json.dump(points, f, indent=2)

    print(f"Wrote {len(points)} kinematic points to {out}")
    print(f"Set Slurm array to:  --array=0-{len(points)-1}")

    # CFF range across the grid
    cff_names = ["ReH", "ReE", "ReHt", "ReEt", "ImH", "ImE", "ImHt", "ImEt"]
    cff_matrix = np.array([p["truth_cffs"] for p in points])
    print(f"\nCFF range across {len(points)} kinematic points:")
    print(f"  {'CFF':8s}  {'min':>10s}  {'max':>10s}  {'mean':>10s}")
    for j, name in enumerate(cff_names):
        col = cff_matrix[:, j]
        print(f"  {name:8s}  {col.min():10.4f}  {col.max():10.4f}  {col.mean():10.4f}")

    print(f"\nFirst 3 points:")
    for p in points[:3]:
        v = p["truth_cffs"]
        print(f"  xB={p['xB']:.3f}  t={p['t']:.3f}  Q2={p['Q2']:.3f}"
              f"  ImH={v[4]:.3f}  ReH={v[0]:.3f}  ImHt={v[6]:.3f}")


if __name__ == "__main__":
    main()
