"""
    TestUtils

Some commom test utilities for portfolio optimization formulations.

Mainly:
    - `get_test_data` that returns a TimeArray of prices for 6 assets.
    - `backtest_po` that provides a basic backtest using provided strategy and returns data.
"""
module TestUtils
    using Distributions
    using JuMP
    using MarketData
    using LinearAlgebra
    import Reexport

    include("auxilary_functions.jl")
    include("backtest.jl")

    export readjust_volumes,
        backtest_po,
        get_test_data,
        mean_variance,
        returns_montecarlo

    Reexport.@reexport using MarketData
end
