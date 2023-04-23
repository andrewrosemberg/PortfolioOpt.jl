using COSMO
using Distributions
using LinearAlgebra
using MarketData
using PortfolioOpt
using PortfolioOpt.TestUtils
using Test

DEFAULT_SOLVER = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

@testset "PortfolioOpt.jl" begin
    include("./VolumeMarket.jl")
    include("./backtest.jl")
    include("./formulations.jl")
end
