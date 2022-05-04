using COSMO
using MarketData
using PortfolioOpt
using PortfolioOpt.TestUtils
using Test

DEFAULT_SOLVER = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

@testset "PortfolioOpt.jl" begin
    # Write tests here.
end
