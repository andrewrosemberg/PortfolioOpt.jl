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

The core functionalities of this package are implementations of `PortfolioRiskMeasure`s of `AmbiguitySets` that can be used to define `ObjectiveTerm`s  and `RiskConstraint`s. 
 - Currently implemented `AmbiguitySets` are all `CenteredAmbiguitySet`, i.e. centered around a Continuous Multivariate `Sampleable`. E.g. : 
    - Any multivariate distribution,
    - `MomentUncertainty`,
    - `BudgetSet`,
    - `EllipticalSet`

    PieceWiseUtility,
    Robustness,
    ExpectedReturn,
    Variance,
    SqrtVariance,
    ConditionalExpectedReturn,
    ExpectedUtility,
    RiskConstraint,
    ConeRegularizer,
    ObjectiveTerm,
    PortfolioFormulation,
    portfolio_model!,
    DeterministicSamples,

    EstimatedCase,
    WorstCase,

 - "End-to-End" functions that receive parameters as inputs and output the weights of a portfolio summing up to the maximum wealth defined in the parameters. These are mainly simple rules or analytical solutions to simple PO formulations: 
    - `max_sharpe` 
    - `equal_weights` 

Normally this package won't focus nor make available forecasting functionalities, but, as an exception, there is one univariate point-prediction forecasting function exported: 
 - `mixed_signals_predict_return`

## TestUtils

As an extra, some testing utilities are available through the submodule called `TestUtils`. 
Mainly:
 - `get_test_data` that returns a TimeArray of Prices for 6 assets.
 - `sequential_backtest_market` that provides a basic backtest using provided strategy and returns data.

But also:
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
    max_std = 0.003 / market_budget(market)
    k_back = 60

    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    
    # Forecast
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    d = MvNormal(r̄, Σ)

    # PO Formulation
    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(SqrtVariance(d), LessThan(max_std)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

```

### Plot Results
```
using Plots

plt = plot(;title="Culmulative Wealth",
    xlabel="Time",
    ylabel="Wealth",
    legend=:outertopright,
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