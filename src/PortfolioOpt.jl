module PortfolioOpt

import Base: size, length
using Distributions
import Distributions: rand
using JuMP
using LinearAlgebra
using LinearAlgebra: dot
using MathOptInterface
using OptimalBids
import MathOptInterface: LessThan, EqualTo, GreaterThan, constant, MAX_SENSE, MIN_SENSE
import OptimalBids: Market, change_bids!, clear_market!, calculate_profit
using PDMats
using ProgressMeter
using Random
import Reexport
using Statistics
import Statistics: mean, cov
import Base: eachindex, eltype, size, keys, (==)
using UUIDs
using JuMP.Containers: DenseAxisArray

include("AmbiguitySet.jl")
include("formulations.jl")
include("VolumeMarket.jl")
include("DeterministicSamples.jl")
include("estimated_mean_variance.jl")
include("conditional_mean.jl")
include("robust_mean.jl")
include("dro_mean.jl")
include("backtest.jl")
include("test_utils/testutils.jl")

export AmbiguitySet,
    CenteredAmbiguitySet,
    ambiguityset,
    PieceWiseUtility,
    coefficients,
    intercepts,
    ExpectedReturn,
    Variance,
    SqrtVariance,
    ConditionalExpectedReturn,
    alpha_quantile,
    sample_size,
    ExpectedUtility,
    utility,
    RiskConstraint,
    ConeRegularizer,
    ObjectiveTerm,
    PortfolioFormulation,
    portfolio_model!,
    DeterministicSamples,
    MomentUncertainty,
    DuWassersteinBall,
    BudgetSet,
    EllipticalSet,
    # VolumeMarket
    VolumeMarket,
    change_bids!,
    clear_market!,
    calculate_profit,
    market_budget,
    set_market_budget,
    market_volume_fee,
    market_model,
    MarketHistory,
    VolumeMarketHistory,
    num_days,
    num_assets,
    market_template,
    current_prices,
    risk_free_rate,
    total_profit,
    # backtest
    sequential_backtest_market,
    get_records

Reexport.@reexport using JuMP

Reexport.@reexport using MathOptInterface: LessThan, EqualTo, GreaterThan

Reexport.@reexport using OptimalBids
end
