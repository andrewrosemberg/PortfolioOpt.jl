# Concepts

```@raw html
<figure class="fig">
  <picture>
    <source media="(max-width: 640px)" srcset="../../assets/diagrams/portfolioopt--formulation-compact.svg">
    <img src="../../assets/diagrams/portfolioopt--formulation.svg" alt="Formulation: three independent choices. A returns model (a distribution P, samples r_1 to r_S, or an ambiguity set of distributions); a measure rho of w (expected return, standard deviation, minus CVaR, a worst case over the ambiguity set, or your own); and a role (objective term c_j rho_j of w, or risk constraint rho_i of w in C_i). Together they make one portfolio problem, maximize the weighted measures subject to the risk constraints and w in X; each measure has a conic form, so JuMP and any conic solver give the decision w star." width="760" height="256">
  </picture>
  <figcaption><b>From three choices to a decision.</b> The returns model, the measure and its role are independent arguments of a formulation, so any one can be swapped. Notation of <a href="../../theory/problem/">The portfolio problem</a>; the conic form is explained in <a href="../../theory/problem/#direction">Conic representation of the measures</a>.</figcaption>
</figure>
```

Each box of the diagram is a type or a function of the package:

| In the diagram | In the code |
|:--|:--|
| returns model ``P`` or ``\mathcal{P}`` | a distribution (`MvNormal`, [`DeterministicSamples`](@ref), ...) or a [`CenteredAmbiguitySet`](@ref), built from past returns by an [`AbstractForecaster`](@ref) with [`forecast`](@ref) |
| measure ``\rho(w)`` | a [`PortfolioRiskMeasure`](@ref), or another [`AbstractPortfolioTerm`](@ref) |
| role ``c_j`` or ``\mathcal{C}_i`` | [`ObjectiveTerm`](@ref) (weight) or [`RiskConstraint`](@ref) (sense and level) |
| the problem | [`PortfolioFormulation`](@ref); ``\mathcal{X}`` is the [`PortfolioMarket`](@ref) |
| conic form, JuMP + conic solver | [`market_model`](@ref) + [`portfolio_model!`](@ref) ([`build_portfolio_model`](@ref)), solved by [`change_bids!`](@ref) |

The walk-forward backtest repeats this every period: [`sequential_backtest_market`](@ref) forecasts from a [`MarketHistory`](@ref), solves, clears the decision against the realized returns ([`clear_market!`](@ref), [`total_profit`](@ref)) and moves to the next date.

## Uncertainty: distributions and ambiguity sets

The returns of the next period are described by an [`AmbiguitySet`](@ref): either a distribution
(any multivariate `Distributions.Sampleable`, e.g. `MvNormal`, a `MixtureModel` or
[`DeterministicSamples`](@ref)), or a [`CenteredAmbiguitySet`](@ref) around a nominal
distribution ([`BudgetSet`](@ref), [`EllipticalSet`](@ref), [`MomentUncertainty`](@ref),
[`DuWassersteinBall`](@ref)). A forecaster ([`AbstractForecaster`](@ref), e.g.
[`EmpiricalGaussian`](@ref), [`EmpiricalSamples`](@ref)) builds one from past returns with
[`forecast`](@ref).

## Measures, terms and formulations

A [`PortfolioRiskMeasure`](@ref) wraps an ambiguity set: `ExpectedReturn(d)` is the mean return
for a distribution and the worst case mean for an ambiguity set. Measures, and other
[`AbstractPortfolioTerm`](@ref)s such as [`ConeRegularizer`](@ref), are used as weighted
[`ObjectiveTerm`](@ref)s and as [`RiskConstraint`](@ref)s of a [`PortfolioFormulation`](@ref).
Which measures are implemented for which sets, and the reformulation of each, is on the
[Theory](../theory/problem.md) pages.

A formulation is data: it does not depend on a model or a solver. [`portfolio_model!`](@ref)
adds it to a JuMP model, calling [`calculate_measure!`](@ref) for each term. With
`record_measures=true` it returns, for the [`uid`](@ref) of each term, the JuMP expression of its
value, which can be read after solving (`value(pointers[uid])`).

## Markets

A [`PortfolioMarket`](@ref) defines the feasible portfolios ([`market_model`](@ref)), holds the
budget and the current decision ([`bids`](@ref)), and computes the profit once the realized
returns are known ([`clear_market!`](@ref), [`total_profit`](@ref)). [`VolumeMarket`](@ref) is the
market of the package: long only or with short selling, a fee per unit invested and a risk free
rate for the money not invested.

The functions that connect formulations and markets are:

| Function | Does |
|:--|:--|
| [`build_portfolio_model`](@ref)`(market, formulation, optimizer)` | `market_model` + `portfolio_model!`, not solved |
| [`change_bids!`](@ref)`(market, model, w)` | solves and stores `value.(w)` as the bids |
| [`change_bids!`](@ref)`(market, formulation, optimizer)` | both of the above |
| [`portfolio_weights`](@ref)`(formulation, N, optimizer)` | the bids on a fresh `VolumeMarket` |

## Backtests

[`sequential_backtest_market`](@ref) walks forward over a [`MarketHistory`](@ref) (e.g. a
[`VolumeMarketHistory`](@ref) of a `TimeArray` of returns). For each date it calls the strategy
with the market and the returns strictly before that date ([`past_prices`](@ref)), clears the
market with the returns of the date, records the state, and reinvests the profit. The
[`StateRecorder`](@ref)s `:wealth`, `:returns` and `:decisions` are recorded by default
([`get_records`](@ref)); if the strategy returns the pointers of `change_bids!` (with
`record_measures=true`) and an `optimization_recorder` is passed, the value of every term is
recorded too. See the [Model Assessment](../examples/model_assessment.md) example.

## Optional integrations

The core depends only on JuMP, MathOptInterface, Distributions, PDMats, ProgressMeter, Reexport
and standard libraries. These integrations are
[package extensions](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)),
loaded when the other package is:

| Package | Adds |
|:--|:--|
| [TimeSeries.jl](https://github.com/JuliaStats/TimeSeries.jl) | a `TimeArray` of returns as the history of a `VolumeMarketHistory` |
| [MarketData.jl](https://github.com/JuliaQuant/MarketData.jl) | `TestUtils.get_test_data()` (and loads TimeSeries) |
| [OptimalBids.jl](https://github.com/andrewrosemberg/OptimalBids.jl) | `OptimalBids.change_bids!`, `clear_market!` and `calculate_profit` accept any `PortfolioMarket` |

## Test utilities

The submodule `PortfolioOpt.TestUtils` has `get_test_data` (prices of 6 assets, needs
`using MarketData`), `mean_variance` and `max_sharpe`.
