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

The main utility script here is primarily just an example of how we can try to determine if the data
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
3. [How we compute the Jacobian (finite differences)](#how-we-compute-the-jacobian-finite-differences)  
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

## How we compute the Jacobian (finite differences)

Because the forward model is complicated, we compute derivatives numerically using **central finite differences**:

$$
\frac{\partial F}{\partial p_j} \approx
\frac{F(p+\epsilon e_j) - F(p-\epsilon e_j)}{2\epsilon}
$$

Where \(e_j\) is the unit vector in direction \(j\).

### Why central differences?

- Forward difference has error \(O(\epsilon)\)
- Central difference has error \(O(\epsilon^2)\) and is usually much more accurate.

### Choosing the step size

We use a relative step:

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

We compute the SVD of the Jacobian:

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
