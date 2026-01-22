# Hard-DR example — enforcing a fixed‑t dispersion relation (ImH + C0 → ReH)

This folder contains a **closure-test pipeline** that demonstrates a “Hard‑DR” extraction
of the DVCS Compton Form Factor (CFF) **𝓗**, where:

- **Im𝓗(ξ)** is learned (or fitted) on a set of ξ nodes, and
- **Re𝓗(ξ)** is **not fitted independently** — it is computed from Im𝓗 through a
  **fixed‑t dispersion relation (DR)** plus a **subtraction constant**.

In other words, the learning problem is:
> fit **Im𝓗** and **C₀**, and compute **Re𝓗** from DR.

This makes the extraction **non‑local in ξ**: Re𝓗 at one ξ depends on Im𝓗 across the whole ξ grid.

---

## What “ReH(ξ) = C₀ + K · ImH” means (physics origin)

### Fixed‑t DR (continuum form)

At leading twist and fixed momentum transfer *t* (and suppressing (t, Q²) labels for brevity),
a standard once‑subtracted dispersion relation for a CFF 𝓖 gives:

\[
\mathrm{Re}\,\mathcal{G}(\xi)
=
C_{\mathcal{G}}
+
\frac{1}{\pi}\,\mathrm{PV}\!\int_0^1 d\xi'\;
\mathrm{Im}\,\mathcal{G}(\xi')\left[\frac{1}{\xi-\xi'} \mp \frac{1}{\xi+\xi'}\right].
\]

- The **PV** (principal value) is required because the kernel is singular at \(\xi'=\xi\).
- For \(\mathcal{G}\in\{\mathcal{H},\mathcal{E}\}\) the **minus** sign is used in the bracket;
  for \(\widetilde{\mathcal{H}},\widetilde{\mathcal{E}}\) the sign differs.
- \(C_{\mathcal{G}}\) is the **subtraction constant**, sometimes written as \(\Delta(t)\) for 𝓗,
  and is commonly discussed in connection with the **GPD D‑term** and related mechanical/pressure
  interpretations.

**This example uses that relation for \(\mathcal{H}\)**.

> Reference: see Eq. (5) in Moutarde et al., EPJ C (2019), arXiv:1905.02089.  
> A compact PV/subtraction‑constant form and D‑term discussion is also shown in Kumerički (Low‑x 2025).

---

## From the integral to a discretized DR kernel (what the code enforces)

In practice we do not know \(\mathrm{Im}\,\mathcal{H}(\xi')\) for all \(\xi'\in(0,1)\).
Instead we represent it on a finite set of **B ξ nodes**:

\[
\xi_1,\xi_2,\ldots,\xi_B.
\]

We then approximate the PV integral with a quadrature rule on the same grid, i.e.

\[
\mathrm{PV}\!\int_0^1 d\xi' \; f(\xi') \;\approx\; \sum_{j=1}^{B} w_j f(\xi_j),
\]

which yields, for each node \(i\),

\[
\mathrm{Re}\,\mathcal{H}(\xi_i)
=
C_0
+
\sum_{j=1}^{B} K_{ij}\, \mathrm{Im}\,\mathcal{H}(\xi_j),
\]

with the **discretized DR kernel matrix**

\[
K_{ij}
=
\frac{1}{\pi}\,w_j\left(\frac{1}{\xi_i-\xi_j}-\frac{1}{\xi_i+\xi_j}\right),
\qquad
K_{ii}\;\text{handled via PV (typically set to 0 in the discrete sum).}
\]

In vector form:

\[
\boxed{\;\mathrm{Re}\,\mathbf{H} = C_0\,\mathbf{1} + K\cdot \mathrm{Im}\,\mathbf{H}\;}
\]

This is exactly the statement “ReH = C0 + K·ImH” used throughout the Hard‑DR example.

### Why multiple ξ points are mandatory
Because ReH(ξᵢ) depends on **ImH at all ξⱼ**, you cannot run Hard‑DR on a single ξ bin.
The method requires a **multi‑bin dataset**.

---

## What each script does

### 1) `generator.py` — generate a multi‑bin closure dataset
Creates pseudodata for XS(ϕ) and BSA(ϕ) across multiple ξ (xB) bins at fixed (t, Q², beam energy).

Typical workflow:
- Choose or auto‑generate a ξ (xB) grid and a shared ϕ grid per bin
- Define **truth** ImH(ξ) (often from KM15 via `gepard`, or a toy truth)
- Compute truth ReH(ξ) using the **same discretized DR kernel** used in training:
  \(\mathrm{ReH}=C_0+K\cdot\mathrm{ImH}\)
- Generate XS(ϕ), BSA(ϕ) using the BKM forward model
- Write NPZ/CSV/JSON outputs under `<OUT_DIR>/data`

---

### 2) `HardDR_training.py` — train Hard‑DR replicas
Trains an ensemble of replicas where:
- the network (or parameterization) learns **ImH(ξ)** on the ξ grid,
- a scalar **C0** is learned as the subtraction constant,
- **ReH(ξ)** is computed deterministically via the discretized DR kernel.

The fit is driven by matching:
- XS(ϕ) and BSA(ϕ) pseudodata
using the BKM forward model, while enforcing DR *hard*.

Outputs:
- replica weights/models + metadata
- training histories

(Exact paths and naming are set in the USER CONFIG section at the top of the script.)

---

### 3) `evaluation.py` — evaluate + plot
Loads replicas and produces:
- mean ± 1σ bands for ImH(ξ) and ReH(ξ)
- diagnostics for C0 and replica spread
- comparisons of predicted vs truth XS(ϕ), BSA(ϕ) across bins

Outputs are written under the evaluation directory specified in the script config.

---

## Quickstart

From this directory (or repo root), run:

```bash
python generator.py
python HardDR_training.py
python evaluation.py

