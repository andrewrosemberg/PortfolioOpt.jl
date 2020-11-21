"""
    TestUtils

Some commom test utilities for portfolio optimization formulations.

Mainly:
    - `get_test_data` that returns a TimeArray of returns for 6 assets.
    - `backtest_po` that 
"""
module TestUtils
    using Distributions
    using JuMP
    using MarketData
    using LinearAlgebra
    import Reexport

    include("./auxilary_functions.jl")
    include("./backtest.jl")

    export reajust_volumes,
        backtest_po,
        base_model,
        compute_solution_backtest,
        get_test_data,
        mean_variance,
        returns_montecarlo

    Reexport.@reexport using MarketData
end