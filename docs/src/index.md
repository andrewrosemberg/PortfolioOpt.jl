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

The current package is unregistered so you will need to add it as follows:

```julia
julia> ] add https://github.com/andrewrosemberg/PortfolioOpt.jl.git 
```

## PO Strategies

There are two types of strategies implemented in this package: 
 - Modifying functions (identifiable by a `!` at the end of the function) that receive a `JuMP` model, a reference to the investment variable present in this model, the formulation and parameters of the strategy as inputs, and modifies the model by adding the necessary variables and constraints. Currently implemented ones are: 
    - `po_max_conditional_expectation_limit_predicted_return!` 
    - `po_max_predicted_return_limit_conditional_expectation!`
    - `po_max_predicted_return_limit_return!`
    - `po_max_return_limit_variance!`
    - `po_max_utility_return!`
    - `po_min_variance_limit_return!`

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
 - `backtest_po` that provides a basic backtest using provided strategy and returns data.
But also:
 - `readjust_volumes`
 - `base_model`
 - `compute_solution_backtest`
 - `mean_variance`

## Contents
```@contents
Pages = ["robust_po.md", "examples.md", "api.md"]
```
