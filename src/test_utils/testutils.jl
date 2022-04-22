"""
    TestUtils

Some commom test utilities for portfolio optimization formulations.

Mainly:
    - `get_test_data` that returns a TimeArray of prices for 6 assets.
    - `backtest_market` that provides a basic backtest using provided strategy and returns data.
"""
module TestUtils
    using Distributions
    using JuMP
    using ..PortfolioOpt: MarketHistory, market_template, past_prices, clear_market!, current_prices, total_profit
    using MarketData
    using LinearAlgebra
    import Reexport

    include("auxilary_functions.jl")
    include("backtest.jl")

    export backtest_market,
        get_test_data,
        mean_variance

    Reexport.@reexport using MarketData
end
