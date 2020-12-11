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
 - Optimization model creation functions that receive the formulation and parameters of the strategy as inputs, and returns a problem with the necessary variables and constraints. Solutions to the resulting optimization model can be computed using ([`compute_solution`](@ref)). Currently implemented ones are: 
    - [`po_max_conditional_expectation_limit_predicted_return`](@ref)
    - [`po_max_predicted_return_limit_conditional_expectation`](@ref)
    - [`po_max_predicted_return_limit_return`](@ref)
    - [`po_max_return_limit_variance`](@ref)
    - [`po_max_utility_return`](@ref)
    - [`po_min_variance_limit_return`](@ref)

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
 - [`backtest_po`](@ref) that provides a basic backtest using provided strategy and returns data.

But also:
 - `readjust_volumes`
 - `mean_variance`

## Contents
```@contents
Pages = ["robust_po.md", "examples.md", "api.md"]
```
