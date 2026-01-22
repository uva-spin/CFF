## Numerical discretization: why `ReH = C0 + K @ ImH`

In the continuum, the Hard-DR step is a (Cauchy principal value) dispersion relation of the form

ReH(ξ) = C0 + p.v. ∫ K(ξ, ξ') ImH(ξ') dξ' .

To compute this numerically, we discretize the integral operator with a quadrature / Nyström method:

1. Choose quadrature nodes {ξ_j} on [0, 1] and weights {w_j}.
2. Evaluate the relation at target points {ξ_i}.
3. Replace the integral by a weighted sum:

   ReH(ξ_i) ≈ C0 + Σ_j w_j K(ξ_i, ξ_j) ImH(ξ_j).

This turns the integral operator into a matrix acting on a vector of sampled values.
Define K_ij = w_j K(ξ_i, ξ_j), ImH_j = ImH(ξ_j), ReH_i = ReH(ξ_i), and 1_i = 1.
Then:

   ReH ≈ C0 * 1 + K * ImH .

### Principal value (PV) handling
Because the kernel is singular on ξ = ξ', the discretization must account for the PV.
Common approaches include:
- subtracting the singularity analytically and quadrature on the remaining smooth integrand;
- using quadrature rules designed for Cauchy principal value integrals;
- grid placement strategies (e.g. midpoint/trapezoid variants) that avoid evaluating at the singularity.

### Suggested numerical-method references
- Nyström / quadrature discretization of integral equations (matrix form arises from weights × kernel evaluation).
- Numerical evaluation of Hilbert transforms / Cauchy principal value integrals (PV-safe quadrature schemes).
- Numerical Kramers–Kronig (dispersion-relation) evaluation as an application of PV/Hilbert quadrature.

