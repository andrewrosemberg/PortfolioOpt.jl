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

    include("./auxilary_functions.jl")
    include("./backtest.jl")

    export reajust_volumes,
        backtest_po
        get_test_data,
        mean_variance,
        returns_montecarlo,
        compute_solution_backtest
end