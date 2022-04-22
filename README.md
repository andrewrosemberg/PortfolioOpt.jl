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

There are two types of strategies implemented in this package: 
 - Optimization model creation functions that receive the formulation and parameters of the strategy as inputs, and returns a problem with the necessary variables and constraints. Solutions to the resulting optimization model can be computed using (`compute_solution`). Currently implemented ones are: 
    - `po_max_conditional_expectation_limit_predicted_return` 
    - `po_max_predicted_return_limit_conditional_expectation`
    - `po_max_predicted_return_limit_return`
    - `po_max_return_limit_variance`
    - `po_max_utility_return`
    - `po_min_variance_limit_return`

 - "End-to-End" functions that receive parameters as inputs and output the weights of a portfolio summing up to the maximum wealth defined in the parameters. These are mainly simple rules or analytical solutions to simple PO formulations: 
    - `max_sharpe` 
    - `equal_weights` 
    - `mean_variance_noRf_analytical`

Normally this package won't focus nor make available forecasting functionalities, but, as an exception, there is one univariate point-prediction forecasting function exported: 
 - `mixed_signals_predict_return`

## TestUtils

As an extra, some testing utilities are available through the submodule called `TestUtils`. 
Mainly:
 - `get_test_data` that returns a TimeArray of Prices for 6 assets.
 - `backtest_market` that provides a basic backtest using provided strategy and returns data.

But also:
 - `readjust_volumes`
 - `mean_variance`

## Example

Simple example of backtest with an available strategy.

```julia
using COSMO
using PortfolioOpt
using PortfolioOpt.TestUtils: 
    backtest_market, get_test_data, mean_variance, 
    percentchange, timestamp, rename!

prices = get_test_data()
numD, numA = size(prices) # A: Assets    D: Days
returns_series = percentchange(prices)

solver = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

start_date = timestamp(returns_series)[100]

wealth_strategy, returns_strategy =
    backtest_market(returns_series; start_date=start_date
    ) do past_returns, current_wealth, risk_free_return

        # Prep data provided by the backtest pipeline
        numD, numA = size(past_returns)
        returns = values(past_returns)
        # calculate mean and variance for the past 60 days
        Σ, r̄ = mean_variance(returns[(end - 60):end, :])

        # Parameters
        # maximum acceptable normalized variance for our portfolio
        max_risk = 0.8
        formulation = MeanVariance(;
            predicted_mean = r̄,
            predicted_covariance = Σ,
        )

        # Build PO model
        model = po_max_return_limit_variance(formulation, max_risk; rf = risk_free_return)

        # Optimize model and retrieve solution
        x = compute_solution(model, solver)

        # return invested portfolio in used currency
        return x * current_wealth
end

```

### Plot Results
```
using Plots

plot(
    wealth_strategy;
    title="Culmulative Wealth",
    xlabel="Time",
    ylabel="Wealth",
    legend=:outertopright,
)
```
![](https://github.com/andrewrosemberg/PortfolioOpt/blob/master/docs/src/assets/cumwealth.png?raw=true)