# PortfolioOpt
Simple package with Portfolio Optimization (PO) formulations using [JuMP.jl](https://github.com/jump-dev/JuMP.jl).

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://andrewrosemberg.github.io/PortfolioOpt.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://andrewrosemberg.github.io/PortfolioOpt.jl/dev)
<!-- [![Build Status](https://travis-ci.com/andrewrosemberg/PortfolioOpt.jl.svg?branch=master)](https://travis-ci.com/andrewrosemberg/PortfolioOpt.jl) -->
<!-- [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) -->
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

## Installation

The current package is unregistered so you will need to add it as follows:

```julia
julia> ] add https://github.com/andrewrosemberg/PortfolioOpt.jl.git 
```

## PO Strategies

The core functionalities of this package are implementations of risk measures (type `PortfolioRiskMeasure`) of the random variable representing the next period portfolio return (`R = w'r`). The investment weights,`w`,  and asset returns,`r`, are used to define the objective's terms (type `ObjectiveTerm`) and risk constraints (type `RiskConstraint`) of a PO formulation (type `PortfolioFormulation`). As with realistic applications, the decision maker might only have limited information about the individual asset returns, so these can be described with ambiguity sets (type `AmbiguitySet`).

Currently acceptable `AmbiguitySet`s are all `CenteredAmbiguitySet`s, i.e. centered around a (usually Continuous) Multivariate `Sampleable`. E.g. :
 - Point distributions (type `Dirac`) if the decision maker has absolute certainty of the PO returns;
 - Any continuous multivariate distribution (type `Sampleable{Multivariate, Continuous}`) if the decision maker can confidently estimate the distribution for the next period's returns;
 - Distributionally robust ambiguity sets if a set of distributions are equally likely to be the true distribution:
    - type `MomentUncertainty`;
 - Robust uncertainty sets if the decision maker can only infer the support of the true distribution (also viewed as distributionally robust ambiguity sets containting just single point distributions):
    - type `BudgetSet`,
    - type `EllipticalSet`.

Currently implemented `PortfolioRiskMeasure`s are: 
 - Expected return (`ExpectedReturn`);
 - `Variance`;
 - Square root of the portfolio variance (`SqrtVariance`);
 - Conditional expected return (`ConditionalExpectedReturn`) - also called Conditional Value at Risk (CVAR) or (Expected Shortfall);
 - Expected utility (`ExpectedUtility`) which computes the expected value of a specified (hopefully concave) utility function (`ConcaveUtilityFunction`):
    - the only implemented one is the piece-wise concave utility function `PieceWiseUtility`.

Given that `AmbiguitySet`s might be sets of distributions, it is necessary to determine which distribution to use in the definition of the `PortfolioRiskMeasure`. This choice can be imposed by the user through their level of robustness (type `Robustness`):
 - `EstimatedCase` if dealing with `CenteredAmbiguitySet`s and the user doesn't want to add any robustness (default);
 - `WorstCase` if the decision maker wants to use the worst case distribution in the ambiguity set.

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

## Extras

Some benchmarks are available as "End-to-End" functions that receive parameters as inputs and output the weights of a portfolio summing up to the maximum wealth defined in the parameters. These are mainly simple rules or analytical solutions to simple PO formulations: 
    - `max_sharpe` 
    - `equal_weights` 

Normally, this package won't focus nor make available forecasting functionalities, but, as an exception, there is one univariate point-prediction forecasting function exported: 
 - `mixed_signals_predict_return`

## TestUtils

As an extra, some testing utilities are available through the submodule called `TestUtils`:
 - `get_test_data` that returns a TimeArray of Prices for 6 assets.
 - `mean_variance`

## Example Markowitz with Empirical Forecaster

Simple example of backtest with an available strategy.

```julia
using COSMO
using Distributions
using PortfolioOpt
using PortfolioOpt.TestUtils

# Read Prices 
prices = get_test_data();
numD, numA = size(prices) # A: Assets    D: Days

# Calculating returns 
returns_series = percentchange(prices);

# Backtest Parameters 
DEFAULT_SOLVER = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

date_range = timestamp(returns_series)[100:end];

# Backtest
backtest_results = Dict()
backtest_results["EP_markowitz_limit_var"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Parameters
    max_std = 0.03 / market_budget(market) # standard deviation limit
    k_back = 60 # forecaster lookback

    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    
    # Empirical Forecast
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    d = MvNormal(r̄, Σ)

    # PO Formulation
    formulation = PortfolioFormulation(MAX_SENSE, # Maximization problem
        ObjectiveTerm(ExpectedReturn(d)), # Objective: Max Expected return of forecasted distribution
        RiskConstraint(SqrtVariance(d), LessThan(max_std)), # Risk: limit PO standard deviation
    )
    
    change_bids!(market, formulation, DEFAULT_SOLVER)
end

```

## Example Markowitz with Empirical Forecaster, Soyster Uncertainty Box and L1 regularizer

```julia

backtest_results["EP_markowitz_with_soyster_l1"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Parameters
    max_std = 0.03 / market_budget(market)
    R = -0.06 / market_budget(market)
    l1_penalty = -0.0003
    k_back = 60
    
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)

    # Empirical Forecast
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    d = MvNormal(r̄, Σ)

    formulation = PortfolioFormulation(MAX_SENSE,
        [ # Objective Terms:
            ObjectiveTerm(ExpectedReturn(d)), # Max Expected return of forecasted distribution
            ObjectiveTerm(ConeRegularizer(MOI.NormOneCone(numA+1)), l1_penalty) # Regularize decisions through norm-1 regularizer with `l1_penalty` coeficient
        ],
        [ # Risk Constraints:
            RiskConstraint(SqrtVariance(d), LessThan(max_std)), # limit PO standard deviation
            RiskConstraint(ExpectedReturn(BudgetSet(d, maximum(abs.(returns); dims=1)'[:] .- r̄, numA * 1.0), WorstCase), GreaterThan(R)), # Worst case return has to be greater than `R`
        ]
    )
    
    change_bids!(market, formulation, DEFAULT_SOLVER)
end

```

## Plot Results
```
using Plots
using Plots.PlotMeasures

plt = plot(;title="Culmulative Wealth",
    xlabel="Time",
    ylabel="Wealth",
    legend=:outertopright,
    left_margin=10mm
);
for (strategy_name, recorders) in backtest_results
    plot!(plt, 
        axes(get_records(recorders[:wealth]), 1), get_records(recorders[:wealth]).data;
        label=strategy_name,
    )
end
plt
```
![](https://github.com/andrewrosemberg/PortfolioOpt/blob/master/docs/src/assets/cumwealth.png?raw=true)