# DVCS CFF Extraction: Jacobian + SVD Diagnostic Guide (XS/BSA/BCA/TSA/DSA)

The example here is similar to what is in the TF basic example.  You first generate the data, run the training script
and the run the evaluation.  Afterward you can look at what is in the evaluation output and then study the Jacobian by
running the Jacobian script.

This is a **diagnostic** guide for using a Jacobian + Singular Value Decomposition (SVD)
analysis to understand **identifiability**, **degeneracies**, and **conditioning** when extracting **8 CFF components**
from the **5-observable** set:

- **XS**: unpolarized cross section 
- **BSA**: beam-spin asymmetry (analyzing power)
- **BCA**: beam-charge asymmetry (analyzing power)  
- **TSA**: target-spin asymmetry (longitudinal, analyzing power)  
- **DSA**: double-spin asymmetry (beam helicity ⨉ longitudinal target, analyzing power)

The main utility script here is primarily just an example of how one can try to determine if the data
we are testing with is at least a good starting point for the given constraints we have to work with.

This tool answers questions like:

- *Why do different replica CFFs produce nearly identical observable curves?*
- *Which CFF combinations are weakly constrained (flat directions)?*
- *How do I decide what observable to add next?*
- *How should I regularize/put priors without biasing well-constrained parameters?*

> **Key concept:** Even with many φ points and even with zero noise, the inverse problem can be **ill-conditioned**:
> some parameter directions change the observables strongly, while others barely change anything.

---

## Contents

1. [Model and notation](#model-and-notation)  
2. [What the Jacobian is](#what-the-jacobian-is)  
3. [How to compute the Jacobian (finite differences)](#how-to-compute-the-jacobian-finite-differences)  
4. [Why SVD and what it means](#why-svd-and-what-it-means)  
5. [How to interpret singular values](#how-to-interpret-singular-values)  
6. [How to interpret the weakest direction vector](#how-to-interpret-the-weakest-direction-vector)  
7. [Condition number and “effective rank”](#condition-number-and-effective-rank)  
8. [Weighted/whitened Jacobian (important)](#weightedwhitened-jacobian-important)  
9. [Actionable diagnostics checklist](#actionable-diagnostics-checklist)  
10. [What to do with the results](#what-to-do-with-the-results)  
11. [How to use SVD to design regularization/priors](#how-to-use-svd-to-design-regularizationpriors)  
12. [How to use SVD to choose new observables](#how-to-use-svd-to-choose-new-observables)  
13. [Multi-kinematics: when it helps, when it doesn’t](#multi-kinematics-when-it-helps-when-it-doesnt)  
14. [Numerical best practices](#numerical-best-practices)  
15. [Appendix: Fisher information and covariance](#appendix-fisher-information-and-covariance)  
16. [Column-scaled diagnostics for individual CFF constraints](#column-scaled-diagnostics-for-individual-cff-constraints)  

---

## Model and notation

### Parameter vector (8 CFF components)

We work with an 8-vector of real parameters:

$$
p =
\begin{bmatrix}
\Re H \\ \Re E \\ \Re \tilde{H} \\ \Re \tilde{E} \\
\Im H \\ \Im E \\ \Im \tilde{H} \\ \Im \tilde{E}
\end{bmatrix}
\in \mathbb{R}^8
$$

### Observable vector (stacked over φ)

For each φ point, the forward model returns five observables:

$$
y(\phi;p) = \big(XS(\phi), BSA(\phi), BCA(\phi), TSA(\phi), DSA(\phi)\big)
$$

If you have **N** φ points, define the **stacked** forward map:

$$
F(p) =
\begin{bmatrix}
XS(\phi_1) \\ \vdots \\ XS(\phi_N) \\
BSA(\phi_1) \\ \vdots \\ BSA(\phi_N) \\
BCA(\phi_1) \\ \vdots \\ BCA(\phi_N) \\
TSA(\phi_1) \\ \vdots \\ TSA(\phi_N) \\
DSA(\phi_1) \\ \vdots \\ DSA(\phi_N)
\end{bmatrix}
\in \mathbb{R}^{5N}
$$

Your implementation produces these via:

```python
from cross_section_script import compute_observables
obs = compute_observables(...)
# obs["xs"], obs["bsa"], obs["bca"], obs["tsa"], obs["dsa"]
```

> Note: BSA/BCA/TSA/DSA are treated as **analyzing powers** (ratios formed from polarized cross sections).
> If you later want “measured asymmetries” you would typically multiply by beam/target polarization factors externally.

---

## What the Jacobian is

The **Jacobian** is the matrix of partial derivatives of the stacked observables with respect to parameters:

$$
J(p) = \frac{\partial F(p)}{\partial p} \in \mathbb{R}^{(5N)\times 8}
$$

Entry-wise:

$$
J_{ij} = \frac{\partial F_i}{\partial p_j}
$$

Interpretation:

- Each **column** tells you how the entire set of curves changes if you perturb one parameter.
- Each **row** corresponds to one observable value at one φ point.

### Why it matters (local linearization)

Near a reference point \(p_0\) (often the truth in closure studies):

$$
F(p_0 + \Delta p) \approx F(p_0) + J(p_0)\,\Delta p
$$

So locally, the inverse problem becomes:

$$
\Delta F \approx J\,\Delta p
$$

This is the key relationship that makes SVD useful: it explains what directions of \(\Delta p\) can (or can’t) be inferred from changes in the observables.

---

## How to compute the Jacobian (finite differences)

Because the forward model is complicated, compute derivatives numerically using **central finite differences**:

$$
\frac{\partial F}{\partial p_j} \approx
\frac{F(p+\epsilon e_j) - F(p-\epsilon e_j)}{2\epsilon}
$$

Where \(e_j\) is the unit vector in direction \(j\).

### Why central differences?

- Forward difference has error \(O(\epsilon)\)
- Central difference has error \(O(\epsilon^2)\) and is usually much more accurate.

### Choosing the step size

Use a relative step:

$$
\epsilon_j = \text{FD\_REL\_EPS}\cdot \max(1,|p_j|)
$$

This prevents problems when some parameters are O(1) and others are O(100) (e.g. \(\Re \tilde E\)).

**Practical guidance:**

- Too small ε ⇒ numerical noise dominates (derivatives become junk)
- Too large ε ⇒ you’re not in the linear regime (derivatives get biased)

Typical values that usually work:
- `FD_REL_EPS = 1e-3` (often good)
- try `5e-4` or `2e-3` if results are unstable

---

## Why SVD and what it means

Compute the SVD of the Jacobian:

$$
J = U\,\Sigma\,V^T
$$

Where:

- \(U \in \mathbb{R}^{(5N)\times 8}\): orthonormal basis in **observable space**
- \(\Sigma = \text{diag}(\sigma_1,\ldots,\sigma_8)\): singular values, ordered \(\sigma_1 \ge \cdots \ge \sigma_8 \ge 0\)
- \(V \in \mathbb{R}^{8\times 8}\): orthonormal basis in **parameter space**
 
### Core interpretation

Let \(v_i\) be the \(i\)-th column of \(V\). Then:

$$
\|J v_i\| = \sigma_i
$$

So:

- **Large** \(\sigma_i\): moving parameters along direction \(v_i\) strongly changes observables ⇒ **well constrained**
- **Small** \(\sigma_i\): moving along \(v_i\) barely changes observables ⇒ **weakly constrained / nearly degenerate**

This is the most direct way to find “flat directions” near \(p_0\).

---

## How to interpret singular values

### Singular values are sensitivity scales

In the linear model \(\Delta F \approx J \Delta p\):

- If you move \(\Delta p = \alpha v_i\), then \(\|\Delta F\| \approx |\alpha|\sigma_i\).

So for a fixed acceptable observable mismatch \(\|\Delta F\|\), the allowed parameter motion scales like:

$$
|\alpha| \approx \frac{\|\Delta F\|}{\sigma_i}
$$

Thus **parameter uncertainty grows like \(1/\sigma_i\)**.

### Your example result (real output)

You reported:

```
Singular values:
[6.2789e-01 4.8398e-01 4.2332e-01 1.5281e-01
 3.6640e-02 3.8680e-03 9.4935e-04 9.6503e-05]
Condition number: 6506
```

This means:

- There is a strong “top mode” with sensitivity ~0.63
- The weakest mode has sensitivity ~1e-4
- The weakest mode is ~6500× less sensitive than the strongest

So you should expect **large replica spread** along that weakest direction even when observables fit extremely well.

---

## How to interpret the weakest direction vector

SVD also tells you *which parameter combination* is weakly constrained.

The script prints the last right singular vector \(v_8\) (associated with \(\sigma_8\)). Your output:

```
Weakest direction:
ImEtilde   -0.984527
ReEtilde   -0.131563
ImE        +0.107947
ImHtilde   -0.034803
ReE        +0.020785
ImH        +0.008995
ReHtilde   -0.004596
ReH        +0.000399
```

Interpretation:

- The “flattest” direction is dominated by **ImEtilde**.
- There is a smaller coupling to **ReEtilde** and **ImE**.

In symbols, approximately:

$$
\Delta p \propto
-0.985\,\Delta(\Im \tilde E)\;
-0.132\,\Delta(\Re \tilde E)\;
+0.108\,\Delta(\Im E)\;
+\cdots
$$

Meaning:

> You can change ImEtilde a lot (with small compensations in ReEtilde and ImE) while hardly changing XS/BSA/BCA/TSA/DSA curves.

This precisely explains:

- replicas with wildly different ImEtilde (and sometimes ReEtilde)  
- nearly identical predicted observables  
- the “curve from mean(CFFs)” being poor even when the mean-of-curves matches data

---

## Condition number and effective rank

### Condition number

$$
\kappa(J)=\frac{\sigma_{\max}}{\sigma_{\min}}
$$

Interpreting \(\kappa\) **only makes sense if your Jacobian is properly scaled/weighted** (see next section). But as a rough guide:

- \(\kappa \lesssim 10^2\): nicely conditioned, expect tight CFF recovery
- \(10^2 \lesssim \kappa \lesssim 10^4\): moderate ill-conditioning, weak-mode drift is common
- \(\kappa \gtrsim 10^6\): severe, inverse is extremely unstable

Your \(\kappa \approx 6500\) is **moderately ill-conditioned**: not hopeless, but strong enough that the weakest mode will wander unless constrained.

### Effective rank

Even if all \(\sigma_i>0\), some may be so small that they are effectively unusable given your noise level. A practical threshold is:

$$
\sigma_i \lesssim \sigma_{\max}\times \tau
$$

where \(\tau\) might be \(10^{-6}\) to \(10^{-3}\) depending on noise and numerical precision.

---

## Weighted/whitened Jacobian (important)

### Why you should whiten

The raw Jacobian mixes:
- XS (units of cross section; magnitude maybe 0.1)
- asymmetries (dimensionless; magnitude maybe 0.1)

If you do SVD on the unweighted Jacobian, the singular values can be dominated by whichever observable has larger numeric scale.

**Correct approach for inference:** whiten by the measurement uncertainties (or your chosen soft floors).

### Weighted least squares viewpoint

If your loss is:

$$
\mathcal{L}(p)=\frac12 \sum_i \left(\frac{F_i(p)-y_i}{\sigma_i}\right)^2
$$

Define \(W=\text{diag}(1/\sigma_i^2)\). Then the local Hessian is approximately:

$$
H \approx J^T W J
$$

Define the whitened Jacobian:

$$
J_w = W^{1/2}J
$$

Then SVD on \(J_w\) is the meaningful conditioning analysis for the *actual fit*.

### How to whiten in code

Let `sigma_soft` be your per-point uncertainty floor (nonzero even for closure if you want a stable normalization):

```python
# sigma_soft shape (N,5) for [XS,BSA,BCA,TSA,DSA]
# Flatten in the same stacking order as forward_vec
w = 1.0 / sigma_soft.reshape(-1)   # W^{1/2} diagonal entries
Jw = J * w[:, None]                # row-wise scaling
U,S,Vt = np.linalg.svd(Jw, full_matrices=False)
```

> If you used a “scale-normalized” loss (e.g. `SCALE_ASYM_ABS=1e-3`), then your whitening effectively uses those scales.

**Recommendation:** For interpreting identifiability, always compute SVD on the Jacobian whitened exactly like your training loss.

---

## Actionable diagnostics checklist

When “closure fails” (CFFs don’t recover truth), do these in order:

### 1) Generator–forward consistency check (must pass)
Compute forward observables at the truth CFFs and compare to dataset `y_central`.

- If max|truth − data| is **not** near machine precision (for 0-noise data), you have a mismatch in definitions/settings.

### 2) Replica observable residual check
Pick two very different replicas and compute:

- max|pred−data| for each observable
- max|pred1−pred2|

If those are tiny while CFFs differ wildly → **identifiability issue**, not a bug.

### 3) Jacobian/SVD
Compute SVD and inspect:

- spread of singular values
- weakest direction(s)

This tells you exactly which CFF combinations are weakly constrained.

### 4) Mode projection diagnostic (optional but powerful)
Given a weak direction \(v_{\min}\), project each replica onto it:

$$
a_r = v_{\min}^T (p_r - p_0)
$$

If \(a_r\) is broad or bimodal, you have a real degeneracy/multimodality along that mode.

---

## What to do with the results

### If observables fit well but CFFs are broad
This is the most common outcome in nonlinear inverse problems.

**Do not** interpret mean(CFF) as your point estimate. In nonlinear models:

$$
\mathbb{E}[F(p)] \neq F(\mathbb{E}[p])
$$

Instead:

- summarize CFFs with **median + quantiles** or **mode clusters**
- summarize observables with **posterior predictive curves**: mean of curves across replicas

### If the weakest direction is dominated by one CFF (your case: ImEtilde)
Then that CFF component is effectively weakly constrained by your current observable set at that kinematics.

**You should expect:**
- large spread and bias in ImEtilde
- coupling drift in ReEtilde and ImE (as seen in the weak vector)

---

## How to use SVD to design regularization/priors

If the data do not constrain a mode, you can stabilize it with a **targeted** prior.

### Mode-targeted quadratic prior (recommended)

Let \(v_{\min}\) be the weakest right singular vector and choose a reference \(p_0\) (truth for closure, or physics expectation). Add:

$$
\mathcal{L}_{prior} = \lambda \left(\frac{v_{\min}^T(p-p_0)}{\sigma_{prior}}\right)^2
$$

This penalizes **only** the weak mode rather than all parameters, which minimizes bias to well-constrained components.

### Simple component prior (quick alternative)

Since your weak mode is ~ImEtilde, you could use:

$$
\mathcal{L}_{prior} = \lambda\left(\frac{\Im \tilde E - \mu}{\sigma}\right)^2
$$

This is easier, but less precise than a mode prior.

---

## How to use SVD to choose new observables

SVD tells you what’s missing: your weak mode is mostly ImEtilde.

So you want an observable whose Jacobian column has strong component along that direction.

Practical guidance:

- Asymmetries (ratios) can hide information. Consider fitting **state-resolved cross sections** instead of only ratios.
- Transverse target observables (UT) and additional double-spin combinations often couple differently to \(\tilde E\) components.
- Multi-kinematics can help if you enforce shared structure across kinematics (see next section).

A practical workflow:

1. Compute SVD for your current observable set. Record weak mode.
2. Add a candidate observable.
3. Recompute SVD. Check if \(\sigma_{\min}\) increases and if the weak vector rotates away from ImEtilde.
4. If yes, you’ve added information in the right direction.

---

## Multi-kinematics: when it helps, when it doesn’t

“Adding more kinematics” helps only if you **share parameters across kinematics**.

- If you fit 8 independent CFFs at each point, you just repeat the same weak directions point-by-point.
- If you impose a shared model (smoothness, functional form, neural net mapping kinematics → CFFs), then the sensitivity directions rotate with kinematics and the global fit can constrain modes that are weak at any single point.

---

## Numerical best practices

- Use **float64** for Jacobian computation whenever possible.
- Use **central finite differences**.
- Tune FD step if singular values vary wildly with small changes:
  - try `FD_REL_EPS = 5e-4`, `1e-3`, `2e-3`
- Whiten the Jacobian exactly like your training loss weighting.
- If your target polarization magnitude is not 1, remember TSA/DSA magnitudes scale accordingly. Keep conventions consistent across generator/train/eval.

---

## Appendix: Fisher information and covariance

If your loss is weighted least squares:

$$
\mathcal{L} = \frac12 \|W^{1/2}(F(p)-y)\|^2
$$

Then near the solution:

$$
H \approx J^T W J = J_w^T J_w
$$

where \(J_w=W^{1/2}J\) is the whitened Jacobian.

Under Gaussian noise assumptions, the parameter covariance is approximated by:

$$
\text{Cov}(p) \approx (J^T W J)^{-1}
$$

SVD makes this explicit:

- The eigenvalues of \(J_w^T J_w\) are \(\sigma_i^2\)
- So the variance along mode \(v_i\) scales like \(1/\sigma_i^2\)

That’s why small \(\sigma_i\) implies huge uncertainty along that direction.

---

## Summary

- The Jacobian tells you *how observables respond to CFF changes*.
- SVD decomposes the inverse problem into orthogonal parameter combinations ranked by how observable they are.
- Small singular values ⇒ weak modes ⇒ replica spread and non-unique CFFs even when observables are fit.
- Your measured weakest direction is dominated by **ImEtilde**, so you should not expect strong closure for ImEtilde from XS/BSA/BCA/TSA/DSA at that kinematics without additional information or priors.

---

## Column-scaled diagnostics for individual CFF constraints

The SVD analysis above is useful for finding weak **linear combinations** of CFFs. Sometimes, however, the goal is more specific:

> For a given subset of observables, how well is each individual CFF constrained?

For that question, start by studying the **columns** of the Jacobian before interpreting the SVD. The raw Jacobian columns are not directly comparable because the rows mix cross sections and asymmetries, and the columns may correspond to CFFs with very different natural scales. The recommended diagnostic object is therefore the **whitened, column-scaled Jacobian**

$$
A = W_O^{1/2} J S_p.
$$

Here:

- \(J\) is the physical finite-difference Jacobian, \(\partial F / \partial p\).
- \(W_O^{1/2}\) whitens the observable space, usually with \(1/\sigma_i\) or with the same scale factors used in the training loss.
- \(S_p = \mathrm{diag}(s_1,\ldots,s_8)\) sets the CFF scale used for a meaningful one-unit displacement.

With this definition, each column of \(A\) has a useful interpretation:

> Column \(A_{:j}\) is the change in all observables, measured in whitened residual units, caused by a one-scale change in CFF \(p_j\).

This avoids trying to make the Jacobian unitary. The correct object is not a unitary matrix; it is a Jacobian analyzed with physically meaningful metrics in observable space and CFF space.

### Choosing the row weights

There are two useful row-weight choices, and they answer different questions.

#### Statistical/experimental metric

Use the pseudo-data uncertainties:

$$
W_O^{1/2}=\mathrm{diag}(1/y_\sigma).
$$

This answers:

> Which CFFs are constrained by the assumed measurement uncertainties?

#### Training-loss metric

Use the same scales that appear in the training loss, for example an XS scale and a common asymmetry scale.

This answers:

> Which CFFs are constrained by the objective function currently being optimized?

Both are useful. When debugging the optimizer, use the training-loss metric. When interpreting physics constraints, use the statistical/experimental metric.

### Choosing the CFF scales

The diagonal scale matrix \(S_p\) should not be chosen from the numerical value of a CFF if that CFF may be close to zero. Better choices are:

- prior widths,
- model-ensemble widths,
- physically allowed ranges,
- or conservative reference scales used consistently across all kinematics.

For example:

```python
CFF_NAMES = [
    "ReH", "ReE", "ReHtilde", "ReEtilde",
    "ImH", "ImE", "ImHtilde", "ImEtilde",
]

# Example only. Replace these with model/prior widths when available.
cff_scales = np.array([
    1.0,   # ReH
    1.0,   # ReE
    1.0,   # ReHtilde
    150.0, # ReEtilde
    1.0,   # ImH
    1.0,   # ImE
    1.0,   # ImHtilde
    150.0, # ImEtilde
], dtype=float)
```

The exact values are less important than using a transparent and stable convention. Once \(S_p\) is chosen, all reported sensitivities should state that they refer to a one-scale CFF displacement.

### Building the scaled Jacobian in code

```python
def build_scaled_jacobian(J, sigma_vec, cff_scales):
    """
    Build A = W^{1/2} J S_p.

    Parameters
    ----------
    J : array, shape (n_observable_rows, 8)
        Physical finite-difference Jacobian.
    sigma_vec : array, shape (n_observable_rows,)
        Observable uncertainties or loss scales, flattened in the same row
        order as the forward vector used to build J.
    cff_scales : array, shape (8,)
        CFF prior widths, physical ranges, or other chosen parameter scales.
    """
    sigma_vec = np.asarray(sigma_vec, dtype=float)
    cff_scales = np.asarray(cff_scales, dtype=float)

    # Avoid division by zero. Rows with infinite scale effectively carry no weight.
    sigma_vec = np.where(sigma_vec > 0.0, sigma_vec, np.inf)

    return (J / sigma_vec[:, None]) * cff_scales[None, :]
```

Be careful that `sigma_vec` must be flattened in exactly the same order as the forward vector used in the Jacobian. If the implementation uses

```python
forward_vec = np.column_stack([xs, bsa, bca, tsa, dsa]).reshape(-1)
```

then the row order is interleaved by φ:

```text
XS(phi1), BSA(phi1), BCA(phi1), TSA(phi1), DSA(phi1),
XS(phi2), BSA(phi2), ...
```

If the implementation instead block-stacks all XS first, then all BSA, etc., the masks and `sigma_vec` flattening must be changed accordingly.

### Fixed-parameter column sensitivity

The simplest individual-CFF diagnostic is the column norm:

$$
d_j^{\mathrm{fixed}} = \|A_{:j}\|.
$$

Equivalently,

$$
\left(d_j^{\mathrm{fixed}}\right)^2 = \left(A^T A\right)_{jj}.
$$

Interpretation:

- Large \(d_j^{\mathrm{fixed}}\): changing CFF \(j\) alone visibly changes the observables.
- Small \(d_j^{\mathrm{fixed}}\): the observable subset is weakly sensitive to CFF \(j\).

This is a **fixed-others** diagnostic. It does not yet tell you whether CFF \(j\) can be separated from the other CFFs.

### Column correlations: detecting look-alike CFFs

Two CFFs may both have large column norms but still be hard to separate because they produce nearly the same pattern of observable changes. Measure this with the column-correlation matrix:

$$
\rho_{ij}
=
\frac{A_{:i}^T A_{:j}}
{\|A_{:i}\|\,\|A_{:j}\|}.
$$

If \(|\rho_{ij}|\approx 1\), then CFFs \(i\) and \(j\) are locally degenerate for that observable subset.

Important interpretation:

> A nuisance CFF being highly correlated with \(\Re H\) or \(\Im H\) does **not** mean it is safe to set to zero. It can mean the opposite: its effect may be absorbed into the fitted \(H\) CFFs and create bias.

### Unique/profiled sensitivity

To ask how well CFF \(j\) is constrained after the other seven CFFs are allowed to float, remove the part of column \(j\) that can be reproduced by the other columns.

Let \(A_{-j}\) be \(A\) with column \(j\) removed. Define the projector onto the span of the other columns:

$$
P_{-j}=A_{-j}A_{-j}^{+},
$$

where \(+\) denotes the pseudoinverse. The unique strength is

$$
d_j^{\mathrm{unique}}
=
\left\|\left(I-P_{-j}\right)A_{:j}\right\|.
$$

Interpretation:

- Large \(d_j^{\mathrm{unique}}\): CFF \(j\) has a distinguishable effect after profiling over the other CFFs.
- Small \(d_j^{\mathrm{unique}}\): CFF \(j\)'s effect can be mimicked by other CFFs.

Near the minimum, the profiled one-sigma uncertainty in the dimensionless variable \(\theta_j = p_j/s_j\) is approximately

$$
\sigma(\theta_j) \approx \frac{1}{d_j^{\mathrm{unique}}}.
$$

### Fisher matrix and marginal uncertainties

The scaled Fisher matrix is

$$
F = A^T A.
$$

If the matrix is well conditioned, the local covariance in the dimensionless CFF variables is

$$
\Sigma_\theta \approx F^{-1}.
$$

If it is singular or nearly singular, use a pseudoinverse or add a prior Fisher matrix:

$$
\Sigma_\theta \approx \left(F + F_{\mathrm{prior}}\right)^{-1}.
$$

Then

$$
\sigma_j^{\mathrm{marg}}
=
\sqrt{(\Sigma_\theta)_{jj}}
$$

is the local marginal uncertainty on the scaled CFF \(\theta_j\). In physical CFF units,

$$
\sigma(p_j)=s_j\,\sigma(\theta_j).
$$

### Implementation: column diagnostics

```python
def column_diagnostics(A, rcond=1e-12):
    """
    Compute fixed sensitivity, column correlations, unique/profiled
    sensitivity, and marginal Fisher uncertainties for A = W^{1/2} J S_p.
    """
    F = A.T @ A

    col_norm = np.linalg.norm(A, axis=0)

    denom = np.outer(col_norm, col_norm)
    rho = np.divide(F, denom, out=np.zeros_like(F), where=denom > 0.0)

    unique_norm = np.zeros(A.shape[1])
    for j in range(A.shape[1]):
        aj = A[:, [j]]
        Aminus = np.delete(A, j, axis=1)
        proj = Aminus @ (np.linalg.pinv(Aminus, rcond=rcond) @ aj)
        unique_norm[j] = np.linalg.norm(aj - proj)

    cov = np.linalg.pinv(F, rcond=rcond)
    sigma_marg = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

    return {
        "F": F,
        "col_norm": col_norm,
        "rho": rho,
        "unique_norm": unique_norm,
        "sigma_marg_scaled": sigma_marg,
    }
```

A useful printed table is:

```python
diag = column_diagnostics(A)

print("Name        fixed_norm   unique_norm  marg_sigma_scaled")
for j, name in enumerate(CFF_NAMES):
    print(
        f"{name:10s} "
        f"{diag['col_norm'][j]:12.4e} "
        f"{diag['unique_norm'][j]:12.4e} "
        f"{diag['sigma_marg_scaled'][j]:16.4e}"
    )
```

### Observable-subset diagnostics

To determine which observable subsets constrain which CFFs, repeat the diagnostics on row subsets of \(A\).

For interleaved row ordering:

```python
OBS_NAMES = np.array(["XS", "BSA", "BCA", "TSA", "DSA"])


def observable_mask_interleaved(n_phi, use_obs):
    keep_one_phi = np.array([name in use_obs for name in OBS_NAMES])
    return np.tile(keep_one_phi, n_phi)
```

Then compare subsets:

```python
subsets = {
    "XS": ["XS"],
    "BSA": ["BSA"],
    "XS+BSA": ["XS", "BSA"],
    "BCA": ["BCA"],
    "TSA": ["TSA"],
    "DSA": ["DSA"],
    "ALL": ["XS", "BSA", "BCA", "TSA", "DSA"],
}

for label, obs_list in subsets.items():
    mask = observable_mask_interleaved(n_phi, obs_list)
    As = A[mask, :]
    diag = column_diagnostics(As)

    print(f"\n=== {label} ===")
    print("Name        fixed_norm   unique_norm  marg_sigma_scaled")
    for j, name in enumerate(CFF_NAMES):
        print(
            f"{name:10s} "
            f"{diag['col_norm'][j]:12.4e} "
            f"{diag['unique_norm'][j]:12.4e} "
            f"{diag['sigma_marg_scaled'][j]:16.4e}"
        )
```

This gives a direct observable-by-observable map of constraint power:

- \(\|A_{:j}\|\): sensitivity if all other CFFs are fixed.
- \(d_j^{\mathrm{unique}}\): independent sensitivity after the other CFFs float.
- \(\sigma_j^{\mathrm{marg}}\): local marginal uncertainty from the Fisher matrix.

### Bias test for reduced CFF fits

This is useful when testing whether only \(\Re H\) and \(\Im H\) can be fitted while the other six CFFs are fixed to zero.

Partition the scaled Jacobian as

$$
A = \begin{bmatrix} A_K & A_N \end{bmatrix},
$$

where \(K\) are the fitted CFFs and \(N\) are the fixed nuisance CFFs. For the parameter ordering used here,

```python
K = [0, 4]             # ReH, ImH
N = [1, 2, 3, 5, 6, 7] # ReE, ReHtilde, ReEtilde, ImE, ImHtilde, ImEtilde
```

If the true nuisance CFFs are nonzero but the reduced fit fixes them to zero, the local bias in the fitted scaled variables is approximately

$$
\delta\hat{\theta}_K
=
\left(A_K^T A_K\right)^{+} A_K^T A_N\,\delta\theta_N,
$$

where

$$
\delta\theta_N = \frac{p_N^{\mathrm{true}}-p_N^{\mathrm{fixed}}}{s_N}.
$$

If the nuisance CFFs are fixed to zero, then \(p_N^{\mathrm{fixed}}=0\). Convert the predicted bias back to physical CFF units with

$$
\delta\hat{p}_K = S_K\,\delta\hat{\theta}_K.
$$

Code:

```python
def omitted_cff_bias(A, p_true, p_fixed, cff_scales, K, N, rcond=1e-12):
    """
    Predict local bias in fitted CFFs K when nuisance CFFs N are fixed
    to p_fixed[N] but truth is p_true[N].
    """
    AK = A[:, K]
    AN = A[:, N]

    delta_theta_N = (p_true[N] - p_fixed[N]) / cff_scales[N]

    bias_theta_K = (
        np.linalg.pinv(AK.T @ AK, rcond=rcond)
        @ AK.T @ AN @ delta_theta_N
    )

    bias_p_K = cff_scales[K] * bias_theta_K
    return bias_p_K, bias_theta_K

p_fixed = np.zeros_like(p_true)
K = [0, 4]
N = [1, 2, 3, 5, 6, 7]

bias_p_K, bias_theta_K = omitted_cff_bias(
    A=A,
    p_true=p_true,
    p_fixed=p_fixed,
    cff_scales=cff_scales,
    K=K,
    N=N,
)

for idx, b in zip(K, bias_p_K):
    print(f"Predicted bias in {CFF_NAMES[idx]} = {b:+.6g}")
```

The reduced two-CFF fit is safe only if this predicted bias is small compared with the statistical or replica uncertainty in \(\Re H\) and \(\Im H\). A nuisance column being correlated with \(H\) is not enough to justify setting it to zero; the bias estimate above should be checked directly.

### Pseudo-data validation tests

Use pseudo data to verify that the diagnostics are usable. The recommended closure tests are:

#### 1. Single-CFF perturbation test

For each CFF \(j\), perturb the truth by one scale:

$$
p \rightarrow p + s_j e_j.
$$

Compare the true nonlinear observable change

$$
\Delta F_j = F(p+s_j e_j)-F(p)
$$

with the linear prediction

$$
\Delta F_j^{\mathrm{lin}} = J_{:j}s_j.
$$

After whitening,

$$
\Delta\chi_j^2
=
\Delta F_j^T W_O \Delta F_j
\approx
\|A_{:j}\|^2.
$$

If this check fails for small perturbations, the finite-difference step, scaling, or local linear approximation needs attention.

#### 2. Profile-scan test

For each CFF \(j\):

1. Fix \(p_j\) at several displaced values.
2. Refit the other seven CFFs.
3. Plot \(\Delta\chi^2(p_j)\).

Near the minimum,

$$
\Delta\chi^2
\approx
\left(d_j^{\mathrm{unique}}\right)^2
\left(\frac{\Delta p_j}{s_j}\right)^2.
$$

This checks whether the unique/profiled sensitivity correctly predicts the actual profiled constraint.

#### 3. Omitted-CFF bias closure

Generate pseudo data with all eight CFFs nonzero. Then fit only \(\Re H\) and \(\Im H\), fixing the other six to zero. Compare the fitted bias with the local prediction from the omitted-CFF formula above.

This directly tests whether a reduced two-CFF extraction is safe for a given observable subset and kinematic point.

#### 4. Noisy replica coverage

Generate many noisy pseudo-data replicas and fit them. For each CFF, check whether the reported one-sigma intervals cover the truth about 68% of the time. If the Fisher estimate predicts small uncertainty but coverage fails, the problem is likely nonlinear, multimodal, or affected by unmodeled nuisance directions.

### Recommended interpretation table

For each observable subset, report the following quantities:

| Quantity | Formula | Meaning |
|---|---:|---|
| Fixed sensitivity | \(\|A_{:j}\|\) | Visibility of CFF \(j\) if all others are fixed |
| Column correlation | \(\rho_{ij}\) | Whether two CFFs create similar observable changes |
| Unique sensitivity | \(\|(I-P_{-j})A_{:j}\|\) | Independent constraint after other CFFs float |
| Marginal uncertainty | \(\sqrt{[(A^TA+F_{prior})^{-1}]_{jj}}\) | Local uncertainty in scaled CFF units |
| Omitted-CFF bias | \(S_K(A_K^TA_K)^+A_K^TA_N\delta\theta_N\) | Bias from fixing nuisance CFFs |

A compact rule of thumb is:

- Large fixed sensitivity and large unique sensitivity: the CFF is individually constrained.
- Large fixed sensitivity but small unique sensitivity: the CFF affects the data but is degenerate with other CFFs.
- Small fixed sensitivity and small unique sensitivity: the observable subset barely sees the CFF.
- Large omitted-CFF bias: the nuisance CFF cannot safely be fixed to zero, even if it is weakly constrained.

### Recommended workflow

Use the following sequence in the Jacobian script:

1. Build the physical finite-difference Jacobian \(J\).
2. Build \(A=W_O^{1/2}JS_p\).
3. Compute column norms, correlations, unique sensitivities, and Fisher uncertainties for each observable subset.
4. Run SVD on \(A\), not raw \(J\), to identify constrained and weak CFF combinations.
5. Run pseudo-data closure tests to verify the local predictions.
6. Use the omitted-CFF bias formula before deciding that any nuisance CFF can be fixed to zero.

This approach separates four issues that are often mixed together:

$$
\text{sensitivity},\quad
\text{degeneracy},\quad
\text{profiled constraint},\quad
\text{bias from omitted CFFs}.
$$

That separation is the main advantage over interpreting only the raw SVD singular values.

