```@meta
CurrentModule = PortfolioOpt
```

```@raw html
<div style="width:100%; height:150px;border-width:4px;border-style:solid;padding-top:25px;
        border-color:#000;border-radius:10px;text-align:center;background-color:#99DDFF;
        color:#000">
    <h3 style="color: black;">Star us on GitHub!</h3>
    <a class="github-button" href="https://github.com/andrewrosemberg/PortfolioOpt.jl" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star andrewrosemberg/PortfolioOpt.jl on GitHub" style="margin:auto">Star</a>
    <script async defer src="https://buttons.github.io/buttons.js"></script>
</div>
```

# PortfolioOpt
Simple package with Portfolio Optimization (PO) formulations using [JuMP.jl](https://github.com/jump-dev/JuMP.jl).

## Installation

```julia
julia> ] add PortfolioOpt
```

## PO Strategies

The core functionalities of this package are implementations of risk measures (type `PortfolioRiskMeasure`) of the random variable representing the next period portfolio return (`R = w'r`). These are used to define the objective's terms (type `ObjectiveTerm`) and risk constraints (type `RiskConstraint`) of a PO formulation (type `PortfolioFormulation`). As with realistic applications, the decision maker might only have limited information about the individual asset returns, so their uncertainty may be described in different ways:

Currently acceptable uncertainty types:
 - Point distributions (type `Dirac`) if the decision maker has absolute certainty of the PO returns;
 - Any continuous multivariate distribution (type `Sampleable{Multivariate, Continuous}`) if the decision maker can confidently estimate the distribution for the next period's returns;
 - Distributionally robust ambiguity sets if a set of distributions may be be the true distribution:
    - type `MomentUncertainty`,
    - type `DuWassersteinBall`.
 - Robust uncertainty sets if the decision maker can only infer the support of the true distribution (also viewed as distributionally robust ambiguity sets containting just single point distributions):
    - type `BudgetSet`,
    - type `EllipticalSet`.

Currently implemented `PortfolioRiskMeasure`s are: 
 - Expected return (`ExpectedReturn`),
 - `Variance`,
 - Square root of the portfolio variance (`SqrtVariance`, i.e., Standard Deviation),
 - Conditional expected return (`ConditionalExpectedReturn`) - also called Conditional Value at Risk (CVAR) or (Expected Shortfall),
 - Expected utility (`ExpectedUtility`) which computes the expected value of a specified (hopefully concave) utility function (`ConcaveUtilityFunction`):
    - the only implemented one is the piece-wise concave utility function `PieceWiseUtility`.

The `PortfolioRiskMeasure`s can be used to define both the `RiskConstraint`s and the `ObjectiveTerm`s in a `PortfolioFormulation` that can be parsed into details of a `JuMP.Model` using the `portfolio_model!` function.

In addition, `ObjectiveTerm`s can also be `ConeRegularizer`s defined by a cone set (e.g. `norm-2`) and a linear transformation (default Identity).

## VolumeMarket
The `portfolio_model!` modifies an existing model `JuMP.Model` with decision variables already created and which, ideally, are already bounded by budget and bound constraints. In order to help users define market constraints, fees and clearing processess, this package also implements an interface with `OptimalBids.jl` (a framework for working with generic markets) through a simple market type called `VolumeMarket`.

`VolumeMarket` represents market models that only allow the strategic agent to bid at market price, thus their decision is restricted to the amount/volume traded of each of the available assets.

The current implementation allows the user to specify:
 - `budget::Real`: Total amount of resources/volume that can be invested (usually the sum the vector of individual invested amounts or the 1-norm of it);
 - `volume_fee::Real`: Cost per unit of volume invested;
 - `allow_short_selling::Bool`: If true allows decision variables to be negative; 
 - `risk_free_rate::Real`: Risk free return (known return of the money not invested).

Once an instance of a `VolumeMarket` is defined, one can call `market_model` to create a `JuMP.Model` with the equivalent constraints, objective terms and variables created. Moreover, after the strategic objective terms and constraints are added on top of this model, it can be passed to the `change_bids!` together with the `VolumeMarket` object to modify the `volume_bids::Vector{Real}`. Alternatively, `change_bids!` can receive the already calculated bids (if chosen elsewhere) or even just the `PortfolioFormulation`, leaving the work of creating the `JuMP.Model` and adding all constraints and objective terms (market based or strategy based) to this function.

A market with already defined strategic bids, i.e. `volume_bids`, can be cleared using the function `clear_market!` that receives the `VolumeMarket` and the `clearing_prices::Vector{Real}`.

To help backtesting, a type `VolumeMarketHistory` was created to contain:
 - `market::VolumeMarket`: The underlying market specifications;
 - `history_clearing_prices`: The vector of vectors representing the historical returns for all assets with index vector `timestamp`;
 - `history_risk_free_rates`: The vector of risk-free rates with index vector `timestamp`;
 - `timestamp`: timestamps indexing the historical asset and risk-free returns;

Instances of `VolumeMarketHistory` are the input of `sequential_backtest_market`: a function that provides a basic backtest using provided strategy and `VolumeMarketHistory` for a specified `date_range` (that needs to have the same `eltype` as `timestamp`).

## TestUtils

As an extra, some testing utilities are available through the submodule called `TestUtils`:
 - `get_test_data`: returns a TimeArray of Prices for 6 assets.
 - `mean_variance`: returns the mean and variance of a array of returns.
 - `max_sharpe`: portfolio optimization that maximizes the sharp ratio.

## Contents
```@contents
Pages = ["robust_po.md", "examples.md", "api.md"]
```
