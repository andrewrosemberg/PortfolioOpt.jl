"""
    TestUtils

Some commom test utilities for portfolio optimization formulations.

Mainly:
    - `get_test_data` that returns a `TimeArray` of prices for 6 assets (needs `using MarketData`).
"""
module TestUtils
using LinearAlgebra
using Statistics

include("auxilary_functions.jl")

export get_test_data, mean_variance, max_sharpe
end
