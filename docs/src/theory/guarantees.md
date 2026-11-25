# Guarantees

Each mathematical claim of the Theory pages has a name. The test with the same name, in
`test/guarantees.jl`, checks it numerically on random instances (fixed seeds) with the solver
tolerance `1e-5`. They are checks, not proofs, but a proof of a claim can refer to its name.

Unless stated otherwise the claims hold under the assumptions of the measure (see its page) and
the [direction assumption](@ref direction).

| Name | Claim | Page |
|:--|:--|:--|
| `cvar_ru_lower_bound` | ``z - \mathbb{E}[(z - R)^+]/\alpha \leq -\mathrm{CVaR}_\alpha(R)`` for every ``z`` | [Risk measures](@ref cvar) |
| `cvar_ru_exact` | the linear reformulation of `ConditionalExpectedReturn(s; α)` equals the sorted sample tail mean, for ``\alpha S`` integer or not, and ``\alpha = 1`` gives the mean | [Risk measures](@ref cvar) |
| `worst_case_return_exact` | `WorstCaseReturn(s)` and `ConditionalExpectedReturn(s; α=0)` equal the smallest sample return | [Risk measures](@ref worst-case-return) |
| `variance_epigraph_exact` | the rotated SOC epigraph of `Variance` and the SOC epigraph of `SqrtVariance` equal ``w^\top \Sigma w`` and its square root when minimized | [Risk measures](@ref variance) |
| `risk_constraint_feasible` | the optimal portfolio satisfies `SqrtVariance`, `Variance`, `ConditionalExpectedReturn` and `CertaintyEquivalent` constraints evaluated directly (not through the reformulation) | [The portfolio problem](@ref direction) |
| `cone_regularizer_exact` | a penalized `ConeRegularizer` equals the norm of the optimal portfolio | [Risk measures](@ref norm-regularizers) |
| `certainty_equivalent_le_mean` | ``CE(R) \leq \mathbb{E}[R]`` and ``\mathbb{E}[u(R)] \leq u(\mathbb{E}[R])`` (Jensen) for Gaussian and sampled returns | [Risk measures](@ref exponential-utility) |
| `elliptical_closed_form` | the worst case expected return over `EllipticalSet` is ``\hat{r}^\top w - \delta \lVert \Sigma^{1/2} w \rVert_2`` | [Ambiguity sets](@ref elliptical-set) |
| `budget_set_strong_duality` | the dual expression of `BudgetSet` equals the primal worst case LP and the greedy closed form, also with short positions | [Ambiguity sets](@ref budget-set) |
| `moment_worst_case_le_nominal` | worst case expected return over `MomentUncertainty` ``\leq`` nominal mean return, for every portfolio and at the optimum | [Ambiguity sets](@ref moment-uncertainty) |
| `moment_monotone_in_radius` | the worst case conditional expected return is nonincreasing in ``\gamma_1`` and ``\gamma_2`` | [Ambiguity sets](@ref moment-uncertainty) |
| `moment_cvar_le_member_cvar` | worst case conditional expected return ``\leq`` that of the Gaussian with the nominal moments | [Ambiguity sets](@ref moment-uncertainty) |
| `moment_bounded_support_between` | unbounded support value ``\leq`` box support value ``\leq`` nominal mean return | [Ambiguity sets](@ref moment-uncertainty) |
| `wasserstein_zero_radius_is_empirical` | `DuWassersteinBall` with ``\epsilon = 0`` gives the sample mean return | [Ambiguity sets](@ref wasserstein-ball) |
| `wasserstein_monotone_in_radius` | the worst case expected return over `DuWassersteinBall` is nonincreasing in ``\epsilon`` and below the sample mean, for ``p = 1, 2, \infty`` | [Ambiguity sets](@ref wasserstein-ball) |

The closed forms of the moment set (``\hat{r}^\top w - \sqrt{\gamma_1} \sigma`` and
``\hat{r}^\top w - \sqrt{\gamma_2 (1 - \alpha)/\alpha} \, \sigma``) and of the exponential utility
(Gaussian, Gaussian mixture and samples) are checked in `test/dro_mean.jl` and
`test/exponential_utility.jl`.

## Adding a guarantee

A new measure or ambiguity set should come with the claims its reformulation relies on (a closed
form, a bound, a monotonicity), stated in its docstring under "Guarantees" and checked by a named
testset in `test/guarantees.jl`.
