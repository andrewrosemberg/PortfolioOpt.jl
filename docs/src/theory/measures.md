# Risk measures

This page defines each measure over a *distribution* of the returns (a `Distributions.Sampleable`
or [`DeterministicSamples`](@ref)); worst cases over ambiguity sets are on the
[Ambiguity sets](ambiguity.md) page. The notation is defined in
[The portfolio problem](problem.md). Every measure is a subtype of [`PortfolioRiskMeasure`](@ref),
and its reformulation is added to the model by [`calculate_measure!`](@ref).

## Summary

| Measure | Value ``\rho(w)`` | Class | Cone | Exact |
|:--|:--|:--|:--|:--|
| [`ExpectedReturn`](@ref)`(d)` | ``\hat{r}^\top w`` | affine | none | always |
| [`Variance`](@ref)`(d)` | ``w^\top \Sigma w`` | convex | quadratic or rotated SOC | direction |
| [`SqrtVariance`](@ref)`(d)` | ``\lVert \Sigma^{1/2} w \rVert_2`` | convex | SOC | direction |
| [`WorstCaseReturn`](@ref)`(s)` | ``\min_s r_s^\top w`` | concave | linear | direction |
| [`ConditionalExpectedReturn`](@ref)`(s; α)` | ``-\mathrm{CVaR}_\alpha(R)`` | concave | linear | direction |
| [`ExpectedUtility`](@ref)`(d, `[`ExponentialUtility`](@ref)`(a))` | ``\mathbb{E}[(1 - e^{-aR})/a]`` | concave | exponential (+ rotated SOC) | direction |
| [`CertaintyEquivalent`](@ref)`(d, `[`ExponentialUtility`](@ref)`(a))` | ``-\frac{1}{a} \log \mathbb{E}[e^{-aR}]`` | concave | exponential (+ rotated SOC) | direction |
| [`ConeRegularizer`](@ref)`(K, M)` | ``\lVert M w \rVert_K`` | convex | ``K`` | direction |

"Direction" means exact under the [direction assumption](@ref direction).
`s` stands for [`DeterministicSamples`](@ref), and `d` for any distribution with a mean and a
covariance (for the exponential utility: `MvNormal`, a `MixtureModel` of `MvNormal`s, or
`DeterministicSamples`).

## Expected return

```math
\rho(w) = \mathbb{E}_P[r^\top w] = \hat{r}^\top w .
```

An affine expression: exact in any direction. Over an ambiguity set it is the worst case expected
return (see [Ambiguity sets](ambiguity.md)).

## [Variance and standard deviation](@id variance)

```math
\mathrm{Var}(R) = w^\top \Sigma w, \qquad \sigma(R) = \lVert \Sigma^{1/2} w \rVert_2 .
```

* [`Variance`](@ref) of plain variables (or numbers) returns the quadratic expression
  ``w^\top \Sigma w``. Of other JuMP expressions it returns an epigraph variable ``v`` with
  ``(v, 1/2, \Sigma^{1/2} w)`` in the rotated second order cone, i.e. ``v \geq w^\top \Sigma w``.
* [`SqrtVariance`](@ref) returns ``t`` with ``(t, \Sigma^{1/2} w)`` in the second order cone,
  i.e. ``t \geq \lVert \Sigma^{1/2} w \rVert_2``.

**Assumptions:** ``\Sigma \succ 0`` (a Cholesky factorization is computed). Over an ambiguity set,
the covariance of its nominal distribution is used.

**Guarantee** (`variance_epigraph_exact`, `risk_constraint_feasible`): both epigraphs are exact in
the convex direction; a portfolio optimal with `RiskConstraint(SqrtVariance(d), LessThan(σ))`
satisfies ``\sqrt{w^\top \Sigma w} \leq σ``.

## [Worst case return](@id worst-case-return)

For samples ``r_1, \dots, r_S``,

```math
\rho(w) = \min_{s} r_s^\top w = \max_{\theta} \{ \theta : \theta \leq r_s^\top w, \; s = 1, \dots, S \}.
```

**Guarantee** (`worst_case_return_exact`): the linear reformulation equals the smallest sample
return. This is the worst case over the convex hull of the samples
(Fernandes et al., 2016).

## [Conditional expected return (CVaR)](@id cvar)

The expected return over the worst ``\alpha`` fraction of outcomes, ``\alpha \in (0, 1]``, i.e.
minus the conditional value at risk (expected shortfall) of the return. Rockafellar & Uryasev
(2000) show that

```math
-\mathrm{CVaR}_\alpha(R) = \max_{z \in \mathbb{R}} \; z - \frac{1}{\alpha} \mathbb{E}\left[(z - R)^+\right],
```

which for samples is the linear program

```math
\begin{aligned}
\max_{z, y} \quad & z - \frac{1}{\alpha} \sum_{s=1}^{S} p_s y_s \\
\text{s.t.} \quad & y_s \geq z - r_s^\top w, \quad y_s \geq 0, \quad s = 1, \dots, S.
\end{aligned}
```

For samples sorted so that ``R_{(1)} \leq \dots \leq R_{(S)}`` and ``K = \lfloor \alpha S \rfloor``,
its value is the tail mean

```math
-\mathrm{CVaR}_\alpha(R) = \frac{1}{\alpha} \left( \frac{1}{S} \sum_{k=1}^{K} R_{(k)} + \left(\alpha - \frac{K}{S}\right) R_{(K+1)} \right),
```

the mean of the ``\alpha S`` worst returns when ``\alpha S`` is an integer. `α = 0` is the worst
case return, and `α = 1` the expected return.

**Assumptions:** ``\alpha \in [0, 1]``, samples with equal probabilities.

**Guarantees:**
* `cvar_ru_lower_bound`: for every ``z``, ``z - \mathbb{E}[(z - R)^+]/\alpha \leq -\mathrm{CVaR}_\alpha(R)``,
  so every feasible point of the linear program is a conservative estimate;
* `cvar_ru_exact`: the optimal value of the linear program equals the sorted tail mean above, also
  when ``\alpha S`` is not an integer;
* `risk_constraint_feasible`: a portfolio optimal with
  `RiskConstraint(ConditionalExpectedReturn(s; α), GreaterThan(b))` has a sample tail mean of at
  least ``b``.

## [Exponential utility and certainty equivalent](@id exponential-utility)

The exponential (constant absolute risk aversion) utility with risk aversion ``a > 0`` is

```math
u(x) = \frac{1 - e^{-a x}}{a},
```

a positive affine transformation of ``-e^{-ax}``. Both measures depend on
``M(w) = \mathbb{E}[e^{-a R}]``:

```math
\mathbb{E}[u(R)] = \frac{1 - M(w)}{a}, \qquad CE(R) = u^{-1}(\mathbb{E}[u(R)]) = -\frac{1}{a} \log M(w).
```

The certainty equivalent is minus the entropic risk measure, in units of return. ``M(w)`` is a
sum of exponentials of convex functions of ``w``:

* Gaussian ``\mathcal{N}(\hat{r}, \Sigma)``: ``M(w) = \exp(-a \hat{r}^\top w + \tfrac{a^2}{2} w^\top \Sigma w)``,
  with an epigraph ``v \geq w^\top \Sigma w`` (rotated SOC);
* Gaussian mixture with probabilities ``\pi_k``: ``M(w) = \sum_k \pi_k \exp(-a \hat{r}_k^\top w + \tfrac{a^2}{2} w^\top \Sigma_k w)``
  (Luxenberg & Boyd, 2024);
* samples: ``M(w) = \sum_s p_s \exp(-a r_s^\top w)``.

The expected utility uses one exponential cone ``t_k \geq e^{e_k}`` per term; the certainty
equivalent is the hypograph ``\theta \leq -\frac{1}{a} \log \sum_k \pi_k e^{e_k}``, written as
``\sum_k \pi_k e^{e_k + a \theta} \leq 1``. For a Gaussian, ``CE = \hat{r}^\top w - \frac{a}{2} w^\top \Sigma w``,
so maximizing it is the mean-variance problem with risk aversion ``a``.

**Assumptions:** ``a > 0``, ``\Sigma \succ 0`` for the Gaussian terms; a solver with exponential
cones (e.g. Clarabel, SCS, Mosek).

**Guarantees:** the closed forms above are checked for each distribution (tests of
`ExponentialUtility`); `certainty_equivalent_le_mean`: by Jensen's inequality,
``CE(R) \leq \mathbb{E}[R]`` and ``\mathbb{E}[u(R)] \leq u(\mathbb{E}[R])``;
`risk_constraint_feasible`: a certainty equivalent constraint `GreaterThan(b)` holds for the
closed form at the optimal portfolio.

## [Norm regularizers](@id norm-regularizers)

[`ConeRegularizer`](@ref)`(K, M)` is ``t`` with ``(t, M w) \in K`` for a norm cone ``K`` (e.g.
`MOI.NormOneCone(N + 1)`, `MOI.SecondOrderCone(N + 1)`), i.e. ``t \geq \lVert M w \rVert``. With a
negative weight in a maximization (a penalty) it is exact (`cone_regularizer_exact`).

## References

* Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of
  Risk*, 2(3), 21–41.
* Fernandes, B., Street, A., Valladão, D., & Fernandes, C. (2016). An adaptive robust portfolio
  optimization model with loss constraints based on data-driven polyhedral uncertainty sets.
  *European Journal of Operational Research*, 255(3), 961–970.
* Luxenberg, E., & Boyd, S. (2024). Portfolio construction with Gaussian mixture returns and
  exponential utility via convex optimization. *Optimization and Engineering*, 25, 555–574.
* Markowitz, H. (1952). Portfolio selection. *The Journal of Finance*, 7(1), 77–91.
