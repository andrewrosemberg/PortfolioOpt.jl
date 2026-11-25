# Ambiguity sets

When the distribution of the returns is not known, a measure can be evaluated in the worst case
over a set of possible means (robust optimization) or of possible distributions (distributionally
robust optimization). Every set is a [`CenteredAmbiguitySet`](@ref) around a nominal distribution
`d` ([`distribution`](@ref)), with mean ``\hat{r}`` and covariance ``\Sigma``. Measures that only
use the covariance ([`Variance`](@ref), [`SqrtVariance`](@ref)) use the covariance of `d`.

All worst cases are concave measures (see the
[direction of optimization](@ref direction)): they are exact when
maximized or constrained from below. Each reformulation is the dual of the inner worst case
problem, so the auxiliary variables are dual variables and no inner problem is solved.

## Summary

| Set | Measures | Cone | Exact | Reference |
|:--|:--|:--|:--|:--|
| [`BudgetSet`](@ref) | [`ExpectedReturn`](@ref) | linear | yes (LP duality) | Bertsimas & Sim (2004) |
| [`EllipticalSet`](@ref) | [`ExpectedReturn`](@ref) | SOC | yes (closed form) | Ben-Tal & Nemirovski (2000) |
| [`MomentUncertainty`](@ref) | [`ExpectedReturn`](@ref), [`ConditionalExpectedReturn`](@ref), [`ExpectedUtility`](@ref) with [`PieceWiseUtility`](@ref) | PSD | yes for unbounded support; conservative with a box support | Delage & Ye (2010); Yang et al. (2014) |
| [`DuWassersteinBall`](@ref) | [`ExpectedReturn`](@ref) | norm cones | see below | Du et al. (2021) |

## [Budget uncertainty set](@id budget-set)

```math
\mathcal{U}_\Gamma = \left\{ \mu : \mu_i = \hat{r}_i + z_i \Delta_i, \; |z_i| \leq 1, \; \textstyle\sum_i |z_i| \leq \Gamma \right\},
\qquad \rho(w) = \min_{\mu \in \mathcal{U}_\Gamma} \mu^\top w .
```

``\Delta \geq 0`` is the maximum deviation of each mean and ``\Gamma \in [0, N]`` the number of
means that can deviate. ``\Gamma = 0`` is the nominal mean and ``\Gamma = N`` the box of
Soyster (1973). The worst case is a linear program; [`calculate_measure!`](@ref) adds the
constraints of its dual and returns the dual objective (see the docstring of `BudgetSet`).

By linear programming duality, for every portfolio, including short positions, the dual
expression equals the primal worst case, and both equal the greedy closed form

```math
\rho(w) = \hat{r}^\top w - \sum_{k=1}^{\lfloor \Gamma \rfloor} g_{(k)} - (\Gamma - \lfloor \Gamma \rfloor) \, g_{(\lfloor \Gamma \rfloor + 1)},
\qquad g_i = \Delta_i |w_i| \text{ sorted decreasingly.}
```

## [Elliptical uncertainty set](@id elliptical-set)

```math
\mathcal{U}_\delta = \left\{ \mu : \lVert \Sigma^{-1/2} (\mu - \hat{r}) \rVert_2 \leq \delta \right\},
\qquad \rho(w) = \min_{\mu \in \mathcal{U}_\delta} \mu^\top w = \hat{r}^\top w - \delta \lVert \Sigma^{1/2} w \rVert_2 .
```

For ``\delta \geq 0`` and ``\Sigma \succ 0``, the minimizer is
``\mu^\star = \hat{r} - \delta \Sigma w / \lVert \Sigma^{1/2} w \rVert_2`` (Cauchy–Schwarz), so
the second order cone reformulation equals the closed form.

## [Moment uncertainty (Delage & Ye)](@id moment-uncertainty)

The distributions whose first two moments are close to the nominal ones, and whose support is in
a box ``[\underline{r}, \overline{r}]`` (by default ``\mathbb{R}^N``):

```math
\mathcal{P}(\gamma_1, \gamma_2) = \left\{ P : \;
\begin{aligned}
& (\mathbb{E}_P[r] - \hat{r})^\top \Sigma^{-1} (\mathbb{E}_P[r] - \hat{r}) \leq \gamma_1, \\
& \mathbb{E}_P[(r - \hat{r})(r - \hat{r})^\top] \preceq \gamma_2 \Sigma, \\
& P(\underline{r} \leq r \leq \overline{r}) = 1
\end{aligned}
\right\}.
```

For a concave piecewise linear function ``h(r) = \min_k (a_k r^\top w + b_k)`` of the returns, the
worst case expectation ``\inf_{P \in \mathcal{P}} \mathbb{E}_P[h(r)]`` is the negative of the
optimal value of the semidefinite program below, an equivalent form of Delage & Ye (2010, Lemma 1)

```math
\begin{aligned}
\min_{Q, q, r_0, P, p, t} \quad & \gamma_2 \langle \Sigma, Q \rangle - \hat{r}^\top Q \hat{r} + r_0 + \langle \Sigma, P \rangle - 2 \hat{r}^\top p + \gamma_1 t \\
\text{s.t.} \quad & p = -q/2 - Q \hat{r}, \quad \begin{bmatrix} P & p \\ p^\top & t \end{bmatrix} \succeq 0, \\
& \xi^\top Q \xi + \xi^\top (q + a_k w) + r_0 + b_k \geq 0 \quad \forall \xi \in [\underline{r}, \overline{r}], \; \forall k.
\end{aligned}
```

With unbounded support, the last constraint is exactly
``\begin{bmatrix} Q & (q + a_k w)/2 \\ (q + a_k w)^\top/2 & r_0 + b_k \end{bmatrix} \succeq 0``.
With a box support, each finite bound gets a nonnegative multiplier ``\lambda`` that is added to
the linear term (Yang et al., 2014), which is a sufficient condition for the constraint to hold
on the box. The measures over this set are:

* [`ExpectedReturn`](@ref): ``h(r) = r^\top w``. With unbounded support it equals
  ``\hat{r}^\top w - \sqrt{\gamma_1} \lVert \Sigma^{1/2} w \rVert_2``.
* [`ConditionalExpectedReturn`](@ref): minus the worst case CVaR,
  ``\max_z \; z - \frac{1}{\alpha} \sup_{P} \mathbb{E}_P[(z - r^\top w)^+]``, with
  ``-(z - r^\top w)^+ = \min(0, r^\top w - z)``. With unbounded support and ``\gamma_1 = 0`` it
  equals ``\hat{r}^\top w - \sqrt{\gamma_2 (1 - \alpha) / \alpha} \lVert \Sigma^{1/2} w \rVert_2``.
  ``\alpha`` has to be in ``(0, 1]``.
* [`ExpectedUtility`](@ref) with a [`PieceWiseUtility`](@ref).

The set is defined for ``\gamma_1 \geq 0``, ``\gamma_2 \geq 1``, ``\Sigma \succ 0`` and
``\underline{r} \leq \hat{r} \leq \overline{r}``; with ``\gamma_2 \geq 1`` it contains the
nominal distribution. Some consequences:

* With unbounded support the reformulation is exact (Delage & Ye, 2010) and gives the closed forms
  above.
* With a box support the value is a lower bound of the exact worst case over the box, and it is
  never smaller than the value with unbounded support.
* The nominal distribution is in the set, so the worst case expected return is at most
  ``\hat{r}^\top w`` for every portfolio, and the optimal worst case objective is at most the
  optimal nominal objective.
* The sets are nested in ``\gamma_1`` and ``\gamma_2``, so the worst case is nonincreasing in both.
* The Gaussian ``\mathcal{N}(\hat{r}, \Sigma)`` is in the set, so the worst case conditional
  expected return is at most its Gaussian value
  ``\hat{r}^\top w - \lVert \Sigma^{1/2} w \rVert_2 \, \phi(\Phi^{-1}(\alpha)) / \alpha``.

## [Wasserstein ball](@id wasserstein-ball)

[`DuWassersteinBall`](@ref) is the ball of radius ``\epsilon`` (type-1 Wasserstein distance)
around the empirical distribution ``\hat{P}_S`` of the samples ``\xi_1, \dots, \xi_S``, with a
support bound ``\Lambda`` and a scaling matrix ``Q`` (default ``I``). The worst case expected
return is reformulated following Du et al. (2021, Corollaries 1–3) as

```math
\begin{aligned}
\max_{\lambda, s, \nu, \tau} \quad & -\left(\lambda \epsilon + \frac{1}{S} \sum_{i=1}^S s_i\right) \\
\text{s.t.} \quad & -w^\top \xi_i + \nu_i^\top Q^{-1} \xi_i + \Lambda \tau_i \leq s_i, \\
& \lVert w - Q^{-1} \nu_i \rVert_p \leq \lambda, \quad \lVert \nu_i \rVert_q \leq \tau_i, \quad i = 1, \dots, S,
\end{aligned}
```

where ``p \in \{1, 2, \infty\}`` is `norm_cone` and ``q`` its dual exponent (a new ``p`` can be
added with a method of `PortfolioOpt.primal_cone`).

The reformulation needs ``\epsilon \geq 0``, ``\Lambda`` at least the largest norm of a sample
(the default) and ``Q`` invertible. No closed form is available in general, but with
``\epsilon = 0`` the ball is the empirical distribution, so the value is the sample mean return;
the balls are nested in ``\epsilon``, so the value is nonincreasing in ``\epsilon`` and at most the
sample mean return, for ``p = 1, 2, \infty``.

## References

* Ben-Tal, A., & Nemirovski, A. (2000). Robust solutions of linear programming problems
  contaminated with uncertain data. *Mathematical Programming*, 88(3), 411–424.
* Bertsimas, D., & Sim, M. (2004). The price of robustness. *Operations Research*, 52(1), 35–53.
* Delage, E., & Ye, Y. (2010). Distributionally robust optimization under moment uncertainty with
  application to data-driven problems. *Operations Research*, 58(3), 595–612.
  [doi:10.1287/opre.1090.0741](https://doi.org/10.1287/opre.1090.0741)
* Du, N., Liu, Y., & Liu, Y. (2021). A new data-driven distributionally robust portfolio
  optimization method based on Wasserstein ambiguity set. *IEEE Access*, 9.
  [ieeexplore 9311154](https://ieeexplore.ieee.org/abstract/document/9311154)
* Soyster, A. L. (1973). Convex programming with set-inclusive constraints and applications to
  inexact linear programming. *Operations Research*, 21(5), 1154–1157.
* Yang, L., Li, Y., Zhou, Z., & Chen, K. (2014). Distributionally robust return-risk optimization
  models and their applications. *Journal of Applied Mathematics*, 2014, 784715.
  [doi:10.1155/2014/784715](https://doi.org/10.1155/2014/784715)
