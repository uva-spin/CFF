# DVCS CFF Extraction: Jacobian + SVD Diagnostic Guide

This document explains how to use the Jacobian + Singular Value
Decomposition (SVD) diagnostic to analyze identifiability, degeneracies,
and conditioning when extracting the 8 Compton Form Factors (CFFs) from
the observable set:

-   XS (unpolarized cross section)
-   BSA (beam-spin asymmetry)
-   BCA (beam-charge asymmetry)
-   TSA (target-spin asymmetry, longitudinal)
-   DSA (double-spin asymmetry, beam + longitudinal target)

The purpose of this guide is to help you determine:

-   Whether the inverse problem is uniquely solvable
-   Which CFF combinations are weakly constrained
-   Why replica fits may reproduce observables perfectly but still give
    very different CFF values
-   What to do when closure fails in parameter space

------------------------------------------------------------------------

# 1. Overview

You are solving an inverse problem:

You fit 8 real CFF parameters

\[ ReH, ReE, ReHtilde, ReEtilde, ImH, ImE, ImHtilde, ImEtilde \]

to a set of observables evaluated over many φ points at a fixed
kinematic point.

Even with: - 100 φ points - 5 observables - zero experimental noise

the CFF extraction can still be:

-   non-unique
-   ill-conditioned
-   multimodal

The Jacobian + SVD analysis tells you which parameter combinations are
weakly constrained and how weak they are.

------------------------------------------------------------------------

# 2. The Forward Map

Define the stacked observable vector:

F(p) = \[ XS(φ1) ... XS(φN), BSA(φ1) ... DSA(φN) \]

This lives in R\^(5N).

The CFF vector p lives in R\^8.

The forward model maps:

p → F(p)

------------------------------------------------------------------------

# 3. The Inverse Problem

We want to solve:

F(p) = y

If multiple different p produce the same F(p), the inverse is not
unique.

------------------------------------------------------------------------

# 4. What the Jacobian Is

The Jacobian is:

J_ij = ∂F_i / ∂p_j

Dimensions: - rows = observable values (5N) - columns = 8 CFF parameters

It tells you how each observable responds to small changes in each CFF.

------------------------------------------------------------------------

# 5. How the Jacobian Is Computed

Central finite differences:

∂F/∂p_j ≈ \[F(p + ε e_j) - F(p - ε e_j)\] / (2ε)

with relative step:

ε_j = FD_REL_EPS × max(1, \|p_j\|)

------------------------------------------------------------------------

# 6. Why We Use SVD

J = U Σ V\^T

Singular values σ measure how strongly parameter directions affect
observables.

Large σ → well constrained\
Small σ → weakly constrained

Condition number κ = σ_max / σ_min

κ \~ 1 → well conditioned\
κ large → ill conditioned

------------------------------------------------------------------------

# 7. Weakest Direction Interpretation

The last right singular vector gives the weakest parameter combination.

Example:

ImEtilde -0.98\
ReEtilde -0.13\
ImE +0.11

Meaning ImEtilde can vary strongly with little observable impact.

------------------------------------------------------------------------

# 8. Practical Diagnostics

Always check:

1)  Truth reproduces dataset\
2)  Replica reproduces dataset\
3)  Singular values and weakest direction

If observables fit but CFFs differ → inverse is weakly constrained.\
If observables do not fit → fix implementation mismatch.

------------------------------------------------------------------------

# 9. What To Do With Results

If degeneracy exists:

-   Report median + quantiles instead of mean
-   Avoid curve(mean(CFF))
-   Add targeted regularization
-   Add observables sensitive to weak directions
-   Use global multi-kinematics fits with shared parametrization

------------------------------------------------------------------------

# 10. Summary

The Jacobian + SVD diagnostic:

-   Quantifies identifiability
-   Identifies weak parameter combinations
-   Explains replica spread
-   Separates degeneracy from implementation bugs

If observables fit but CFFs vary, the inverse problem is weakly
constrained. If observables do not fit, there is a mismatch in the
pipeline.
