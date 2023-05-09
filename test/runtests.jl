using COSMO
using Distributions
using LinearAlgebra
using MarketData
using PortfolioOpt
using PortfolioOpt.TestUtils
using Random
using Test
using UUIDs

DEFAULT_SOLVER = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

rng = MersenneTwister(1234)

include("./generated_data.jl")

@testset "PortfolioOpt.jl" begin
    include("./VolumeMarket.jl")
    include("./backtest.jl")
    include("./formulations.jl")
    include("./estimated_mean_variance.jl")
    include("./robust_mean.jl")
    include("./dro_mean.jl")
end
