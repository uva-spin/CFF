import numpy as np
import json
from cross_section_script import compute_observables

# =============================
# Basic CONFIG
# =============================
TAG = "v_1"
DATA_PATH = f"output/data/dataset_{TAG}.npz"
TRUTH_PATH = f"output/data/truth_{TAG}.json"
K_BEAM = 5.75
USING_WW = True
FD_REL_EPS = 1e-3
# =============================

# ---- Load dataset ----
data = np.load(DATA_PATH)
X = data["x"].astype(float)
phi = X[:, 3]
t, xb, q2 = X[0, 0], X[0, 1], X[0, 2]

# ---- Load truth CFFs ----
truth = json.load(open(TRUTH_PATH))
c = truth["km15_truth_cffs"]

truth8 = np.array([
    c["cff_real_h_km15"],
    c["cff_real_e_km15"],
    c["cff_real_ht_km15"],
    c["cff_real_et_km15"],
    c["cff_imag_h_km15"],
    c["cff_imag_e_km15"],
    c["cff_imag_ht_km15"],
    c["cff_imag_et_km15"],
], dtype=float)

names = ["ReH","ReE","ReHtilde","ReEtilde",
         "ImH","ImE","ImHtilde","ImEtilde"]

# ---- Forward wrapper ----
def forward_vec(c8):
    cffs = dict(
        re_h=c8[0], re_e=c8[1], re_ht=c8[2], re_et=c8[3],
        im_h=c8[4], im_e=c8[5], im_ht=c8[6], im_et=c8[7],
    )
    obs = compute_observables(
        phi_rad=phi,
        k_beam=K_BEAM,
        q_squared=q2,
        xb=xb,
        t=t,
        cffs=cffs,
        using_ww=USING_WW,
    )
    # stack all observables into one long vector
    return np.column_stack([
        obs["xs"],
        obs["bsa"],
        obs["bca"],
        obs["tsa"],
        obs["dsa"],
    ]).reshape(-1)

# ---- Evaluate at truth ----
y0 = forward_vec(truth8)
N = y0.size

# ---- Build Jacobian by finite difference ----
J = np.zeros((N, 8), dtype=float)

for j in range(8):
    step = FD_REL_EPS * max(1.0, abs(truth8[j]))
    cp = truth8.copy()
    cm = truth8.copy()
    cp[j] += step
    cm[j] -= step
    J[:, j] = (forward_vec(cp) - forward_vec(cm)) / (2 * step)

# ---- SVD ----
U, S, Vt = np.linalg.svd(J, full_matrices=False)

print("\nSingular values:")
print(S)

cond = S[0] / S[-1] if S[-1] > 0 else np.inf
print("\nCondition number:", cond)

print("\nWeakest direction (right singular vector):")
weak = Vt[-1]
for name, val in sorted(zip(names, weak), key=lambda x: -abs(x[1])):
    print(f"{name:10s} {val:+.6f}")
