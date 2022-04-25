module PortfolioOpt

import Base: size, length
using Distributions
using JuMP
using LinearAlgebra
using LinearAlgebra: dot
using MathOptInterface
using OptimalBids
import MathOptInterface: LessThan, EqualTo, GreaterThan, constant, MAX_SENSE, MIN_SENSE
import OptimalBids: Market, change_bids!, clear_market!, calculate_profit
using PDMats
using Random
import Reexport
using Statistics
import Statistics: mean, cov
import Base: eachindex, eltype, size, keys
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
include("simple_decision_rules.jl")
include("forecasts.jl")
include("test_utils/testutils.jl")

export AmbiguitySet,
    CenteredAmbiguitySet,
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
    MomentUncertainty,
    BudgetSet,
    EllipticalSet,
    EstimatedCase,
    WorstCase,
    # VolumeMarket
    VolumeMarket,
    change_bids!,
    clear_market!,
    market_budget,
    market_volume_fee,
    market_model,
    MarketHistory,
    VolumeMarketHistory,
    market_template,
    current_prices,
    risk_free_rate,
    # backtest
    sequential_backtest_market,
    get_records,
    # end-to-end
    max_sharpe,
    equal_weights

Reexport.@reexport using JuMP

Reexport.@reexport using MathOptInterface: LessThan, EqualTo, GreaterThan

Reexport.@reexport using OptimalBids
end
