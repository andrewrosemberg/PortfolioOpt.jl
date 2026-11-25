module PortfolioOpt

import Base: size, length
using Distributions
import Distributions: rand
using JuMP
using LinearAlgebra
using LinearAlgebra: dot
using MathOptInterface
import MathOptInterface: LessThan, EqualTo, GreaterThan, constant, MAX_SENSE, MIN_SENSE
using PDMats
using ProgressMeter
using Random
using Reexport: Reexport
using Statistics
import Statistics: mean, cov
import Base: eachindex, eltype, size, keys, (==)
using UUIDs
using JuMP.Containers: DenseAxisArray

include("AmbiguitySet.jl")
include("formulations.jl")
include("market_interface.jl")
include("VolumeMarket.jl")
include("DeterministicSamples.jl")
include("estimated_mean_variance.jl")
include("conditional_mean.jl")
include("robust_mean.jl")
include("dro_mean.jl")
include("exponential_utility.jl")
include("backtest.jl")
include("forecasters.jl")
include("test_utils/testutils.jl")

export AmbiguitySet,
    CenteredAmbiguitySet,
    ambiguityset,
    distribution,
    PieceWiseUtility,
    coefficients,
    intercepts,
    AbstractPortfolioTerm,
    PortfolioRiskMeasure,
    calculate_measure!,
    uid,
    ExpectedReturn,
    WorstCaseReturn,
    Variance,
    SqrtVariance,
    ConditionalExpectedReturn,
    alpha_quantile,
    sample_size,
    ExpectedUtility,
    utility,
    ExponentialUtility,
    risk_aversion,
    CertaintyEquivalent,
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
    # markets
    PortfolioMarket,
    bids,
    set_bids!,
    build_portfolio_model,
    VolumeMarket,
    change_bids!,
    clear_market!,
    calculate_profit,
    market_budget,
    set_market_budget,
    market_volume_fee,
    market_model,
    MarketHistory,
    past_prices,
    VolumeMarketHistory,
    num_days,
    num_assets,
    market_template,
    current_prices,
    risk_free_rate,
    total_profit,
    # backtest
    sequential_backtest_market,
    get_records,
    record!,
    StateRecorder,
    portfolio_weights,
    # forecasters
    AbstractForecaster,
    forecast,
    EmpiricalGaussian,
    EmpiricalSamples

Reexport.@reexport using JuMP

Reexport.@reexport using MathOptInterface: LessThan, EqualTo, GreaterThan

function __init__()
    # `get_test_data` is implemented by the MarketData extension
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        if exc.f === TestUtils.get_test_data
            print(
                io,
                "\n`get_test_data` needs the MarketData package: add it to your environment and run `using MarketData`.",
            )
        end
    end
    return nothing
end

end
