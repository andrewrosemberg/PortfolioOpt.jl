module PortfolioOpt

import Base: size, length
using Distributions
using JuMP
using LinearAlgebra
using LinearAlgebra: dot
using MathOptInterface: LessThan, EqualTo, GreaterThan, constant
using PDMats
using Random
import Reexport
using Statistics

include("formulations.jl")
include("mean_variance.jl")
include("mean_variance_robust.jl")
include("mean_variance_dro.jl")
include("sample_based.jl")
include("sample_based_robust.jl")
include("sample_based_stochastic.jl")
include("simple_rules.jl")
include("utils.jl")
include("forecasts.jl")
include("test_utils/testutils.jl")

export AmbiguitySet,
    AbstractMeanVariance,
    AbstractSampleBased,
    base_model,
    compute_solution,
    MeanVariance,
    mixed_signals_predict_return,
    portfolio_return!,
    portfolio_variance!,
    po_max_conditional_expectation_limit_predicted_return,
    po_max_predicted_return_limit_conditional_expectation,
    po_max_predicted_return_limit_return,
    po_max_return_limit_variance,
    po_max_utility_return,
    po_min_variance_limit_return,
    predicted_portfolio_return!,
    predicted_portfolio_variance!,
    readjust_volumes!,
    RobustBenTal,
    RobustBertsimas,
    MomentUncertainty,
    RobustBetina,
    SampleBased,
    # end-to-end
    mean_variance_noRf_analytical,
    max_sharpe,
    equal_weights

Reexport.@reexport using JuMP

Reexport.@reexport using MathOptInterface: LessThan, EqualTo, GreaterThan

end
