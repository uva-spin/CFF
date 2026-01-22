# Hard-DR example (hard dispersion-relation constraint)

This folder provides a minimal, self-contained example of **CFF extraction** where the **real part** is not fit independently, but is instead **computed from the imaginary part** using a **fixed‑t dispersion relation** (“Hard‑DR”).

In other words: we fit **Im ℋ(ξ)** (and a subtraction constant **C₀**) and **derive Re ℋ(ξ)** via a discretized dispersion-relation operator.

> This is an example implementation meant to be readable and hackable, not a production-grade numerical dispersion solver.

---

## What’s in this folder

Typical files in this example:

- `generator.py`  
  Creates a synthetic dataset on a chosen ξ-grid. Produces pairs like `(ξ, Imℋ(ξ), Reℋ(ξ))`, where `Reℋ` is generated from `Imℋ` using the same discretized DR kernel used during training.

- `HardDR_training.py`  
  Trains a model that predicts **Imℋ(ξ)** (and optionally learns **C₀**) and computes **Reℋ(ξ)** by applying the **Hard‑DR kernel**. Losses are evaluated against the training targets.

- `evaluation.py`  
  Loads a trained checkpoint and evaluates / plots predictions, typically comparing predicted vs. true `Imℋ` and `Reℋ`.

> Exact CLI flags / config names can differ slightly — run each script with `--help` (if supported) or open the file and edit the config block at the top.

---

## The physics idea (fixed‑t dispersion relation)

For the (crossing-even) CFF **ℋ**, a commonly used fixed‑t dispersion form relates its real and imaginary parts with a subtraction constant:

$$\Re \mathcal{H}(\xi,t)=
C_0(t)
+
\frac{1}{\pi}\,\mathrm{PV}\!\int_{0}^{1} d\xi'\;
\Im \mathcal{H}(\xi',t)\,
\left(
\frac{1}{\xi-\xi'}-\frac{1}{\xi+\xi'}
\right).$$

- **PV** denotes the **Cauchy principal value** (the kernel is singular at $\xi'=\xi$).
- $C_0(t)$ is the **subtraction constant** (independent of $\xi$ at fixed $t$).

In this “Hard‑DR” approach, we do **not** fit $\Re\mathcal{H}$ freely; we enforce the relationship above by construction.

---

## Where the vector/matrix form comes from

### Step 1: choose a ξ grid and quadrature weights

Let the ξ grid be points $${\xi_j}_{j=1}^{N}$$, with quadrature weights $\{w_j\}_{j=1}^{N}$ for integrals on $[0,1]$.

A generic quadrature approximation is:

$$
\int_0^1 f(\xi')\,d\xi'
\;\approx\;
\sum_{j=1}^N w_j\,f(\xi_j).
$$

### Step 2: discretize the integral operator (Nyström / quadrature discretization)

Define the DR kernel (for targets $\xi_i$ and sources $\xi_j$):

$$K_{ij}=\frac{w_j}{\pi}\left(
\frac{1}{\xi_i-\xi_j}-
\frac{1}{\xi_i+\xi_j}
\right),
\qquad i,j=1,\dots,N.$$

Then the discretized DR becomes:

$$\Re\mathcal{H}(\xi_i)
\approx
C_0
+
\sum_{j=1}^N K_{ij}\,\Im\mathcal{H}(\xi_j).$$

In vector form:

$$
\Re\boldsymbol{\mathcal{H}}
\approx
C_0\,\mathbf{1}
+
\mathbf{K}\;\Im\boldsymbol{\mathcal{H}},
$$

where:

- $\Im\boldsymbol{\mathcal{H}} = [\Im\mathcal{H}(\xi_1),\dots,\Im\mathcal{H}(\xi_N)]^\top$
- $\Re\boldsymbol{\mathcal{H}} = [\Re\mathcal{H}(\xi_1),\dots,\Re\mathcal{H}(\xi_N)]^\top$
- $\mathbf{1}$ is an all-ones vector.

This is the exact origin of the “discretized DR kernel” statement:

$$
\Re\mathcal{H}(\xi) = C_0 + \mathbf{K}\cdot \Im\mathcal{H}.
$$

---

## Handling the principal value (PV) numerically

The term $\frac{1}{\xi_i-\xi_j}$ is singular at $i=j$. Numerically, there are a few standard strategies:

1. **Omit / neutralize the diagonal term** and interpret the sum as a PV approximation  
   (common in simple demo implementations).

2. Use a **dedicated PV quadrature rule** (e.g., Clenshaw–Curtis-type PV rules or corrected trapezoid rules).

3. Recast the transform as a **Hilbert-transform-style** computation and use specialized numerical methods.

This example intentionally keeps things simple so you can see the mechanics. If you need high precision (or stability at high N), you’ll likely want to upgrade the PV treatment.

---

## How the “Hard‑DR” constraint is used in training

Instead of training a network to output both $\Re\mathcal{H}$ and $\Im\mathcal{H}$ independently, we:

1. Predict $\Im\mathcal{H}$ on the grid (and optionally fit/learn $C_0$).
2. Compute:
   $$\Re\boldsymbol{\mathcal{H}}_{\text{pred}}=
   C_0\,\mathbf{1} + \mathbf{K}\,\Im\boldsymbol{\mathcal{H}}_{\text{pred}}.
   $$
4. Compare $(\Im\mathcal{H}_{\text{pred}}, \Re\mathcal{H}_{\text{pred}})$ against the dataset targets.

This forces every prediction to satisfy the discretized DR by construction — hence “Hard‑DR”.

---

## Quickstart

A typical workflow is:

1. **Generate synthetic data**
   - Run `generator.py` to create a dataset file.

2. **Train**
   - Run `HardDR_training.py` to train a model checkpoint.

3. **Evaluate**
   - Run `evaluation.py` to compute metrics / plots.

If scripts support CLI args, use:

- `python generator.py --help`
- `python HardDR_training.py --help`
- `python evaluation.py --help`

Otherwise, open each script and edit the config section near the top.

---

## Practical notes / gotchas

- **Grid consistency matters:** the ξ grid and weights used to build $\mathbf{K}$ must match between:
  - dataset generation
  - training
  - evaluation

- **$C_0$ identifiability:** since $C_0$ is a ξ-independent offset to $\Re\mathcal{H}$, it can trade off with systematic offsets in the data (and with discretization error). That’s normal; keep an eye on it.

- **Don’t wrap this README in triple backticks** in GitHub.  
  GitHub renders LaTeX math when you use `$...$` or `$$...$$` directly in Markdown.

---

## References (physics + numerical justification)

### Dispersion relations / CFF context
- A.V. Belitsky, D. Müller, A. Kirchner,  
  *Theory of deeply virtual Compton scattering on the nucleon*,  
  **Nucl. Phys. B629 (2002) 323–392**. DOI: 10.1016/S0550-3213(02)00144-X.  
  (See arXiv:hep-ph/0112108)

- K. Kumerički, D. Müller, K. Passek‑Kumerički,  
  *Sum rules and dualities for generalized parton distributions: is there a holographic principle?*,  
  **Eur. Phys. J. C58 (2008) 193–215**. DOI: 10.1140/epjc/s10052-008-0741-0.  
  (See arXiv:0805.0152)

- M. Diehl, D.Yu. Ivanov,  
  *Dispersion representations for hard exclusive processes*,  
  **Eur. Phys. J. C52 (2007) 919–932**. DOI: 10.1140/epjc/s10052-007-0401-9.  
  (See arXiv:0707.0351)

- H. Moutarde, P. Sznajder, J. Wagner,  
  *Unbiased determination of DVCS Compton Form Factors*,  
  **Eur. Phys. J. C79 (2019) 614**. DOI: 10.1140/epjc/s10052-019-7117-5.  
  (See arXiv:1905.02089)

### Numerical discretization of integral operators (why the matrix form is standard)
- S. Hao, A.H. Barnett, P.G. Martinsson, P. Young,  
  *Quadrature for Nyström discretization of integral equations on curves*,  
  (Notes / technical report; see PDF: “ai,j = k(xi, xj) wj” as the basic Nyström construction)

- M.M. Chawla, N. Jayarajan,  
  *Quadrature formulas for Cauchy principal value integrals*,  
  **Computing 15 (1975) 347–355**. DOI: 10.1007/BF02260318.

- L.N. Trefethen, J.A.C. Weideman,  
  *The Exponentially Convergent Trapezoidal Rule*,  
  **SIAM Review 56 (2014) 385–458**. DOI: 10.1137/130932132.

### Example of DR/Kramers–Kronig discretization on sampled data (analogous numerics)
- P.D. Fitzgerald,  
  *Numerical Approximation of Kramers-Kronig Relations to Transform Discretized Absorption Data*,  
  arXiv:2012.02369.
