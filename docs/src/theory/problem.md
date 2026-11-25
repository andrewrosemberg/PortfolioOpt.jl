# The portfolio problem

This page fixes the notation used in the Theory pages and states the one property that every
formulation of the package relies on.

## Notation

| Symbol | Meaning | In the code |
|:--|:--|:--|
| ``N`` | number of assets | `num_assets(market)` |
| ``w \in \mathbb{R}^N`` | money invested in each asset (the weights when the budget is 1) | the decision variables `w` |
| ``r \in \mathbb{R}^N`` | random returns of the assets over the next period | |
| ``R = r^\top w`` | random return of the portfolio | |
| ``P`` | distribution of ``r`` | a `Distributions.Sampleable`, e.g. `MvNormal` |
| ``\hat{r}, \Sigma`` | mean and covariance of the nominal distribution | `mean(d)`, `cov(d)` |
| ``r_1, \dots, r_S`` | samples (scenarios) of the returns, each with probability ``p_s = 1/S`` | columns of [`DeterministicSamples`](@ref) |
| ``\mathcal{P}`` | ambiguity set: a set of distributions that may be the true one | a [`CenteredAmbiguitySet`](@ref) |
| ``\mathcal{X}`` | feasible portfolios of the market (budget, short selling, ...) | [`market_model`](@ref) |
| ``\rho(w)`` | a measure of the portfolio return, e.g. its mean or its conditional value at risk | a [`PortfolioRiskMeasure`](@ref) |

``\Sigma \succ 0`` (positive definite) is assumed wherever a Cholesky factor
``\Sigma^{1/2}`` (with ``\Sigma = \Sigma^{\top/2} \Sigma^{1/2}``) is used.

## The problem

A [`PortfolioFormulation`](@ref) with objective terms ``(\rho_j, c_j)`` ([`ObjectiveTerm`](@ref))
and risk constraints ``(\rho_i, \mathcal{C}_i)`` ([`RiskConstraint`](@ref)) is the problem

```math
\begin{aligned}
\max_{w} \quad & \sum_j c_j \, \rho_j(w) \\
\text{s.t.} \quad & \rho_i(w) \in \mathcal{C}_i, \quad i = 1, \dots, m, \\
& w \in \mathcal{X},
\end{aligned}
```

(or ``\min`` with `MIN_SENSE`), where each ``\mathcal{C}_i`` is ``(-\infty, b]`` (`LessThan(b)`),
``[b, \infty)`` (`GreaterThan(b)`) or ``\{b\}`` (`EqualTo(b)`). [`portfolio_model!`](@ref) adds
the terms and constraints to a JuMP model whose variables ``w`` and constraints
``w \in \mathcal{X}`` were created by [`market_model`](@ref).

## [Conic representation of the measures](@id direction)

Most measures are not affine in ``w``: a worst case, a tail expectation, a norm. For each measure,
[`calculate_measure!`](@ref) adds auxiliary variables ``u`` and conic constraints
``(w, u) \in \mathcal{K}`` (linear, second order, exponential or semidefinite cones) to the model,
and returns an affine expression ``f(w, u)``. The measures fall in two classes:

* **concave** measures (returns, worst cases, utilities), for which
  ```math
  \rho(w) = \max_{u \,:\, (w, u) \in \mathcal{K}} f(w, u),
  ```
  so ``f(w, u) \leq \rho(w)`` for every feasible ``u``;
* **convex** measures (variance, standard deviation, norms), for which
  ```math
  \rho(w) = \min_{u \,:\, (w, u) \in \mathcal{K}} f(w, u),
  ```
  so ``f(w, u) \geq \rho(w)`` for every feasible ``u``.

!!! note "Assumption (direction)"
    The reformulation of a measure is **exact** when the optimization pushes ``f`` in the
    direction of its class: a concave measure is maximized (positive weight with `MAX_SENSE`,
    negative weight with `MIN_SENSE`) or constrained from below (`GreaterThan`); a convex measure
    is minimized or constrained from above (`LessThan`).

    Under this assumption, any optimal solution of the conic problem gives a portfolio ``w``
    that is optimal for the original problem and satisfies every risk constraint evaluated on the
    measure itself (not only on its reformulation), and the optimal values coincide.

    In the other direction the problem is unbounded or a relaxation: for example a concave
    measure with an `EqualTo(b)` constraint only enforces ``\rho(w) \geq b``. This is the
    usual convexity requirement: maximizing a convex function, or constraining a concave function
    from above, is not a convex problem.

The [Risk measures](measures.md) and [Ambiguity sets](ambiguity.md) pages give, for each
measure, its definition, its class, its conic reformulation and the assumptions under which the
reformulation is exact. Each claim has a name, listed in [Guarantees](guarantees.md), and is
checked numerically by the test with the same name.
