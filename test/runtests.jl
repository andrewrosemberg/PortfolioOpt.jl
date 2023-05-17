using COSMO
using Distributions
using LinearAlgebra
using MarketData
using PortfolioOpt
using PortfolioOpt.TestUtils
using Random
using Test
using UUIDs

include("./generated_data.jl")

DEFAULT_SOLVER = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

rng = MersenneTwister(1234)

function util_test_directory(dir, exclude = String[])
    for (root, _, files) in walkdir(dir)
        for file in files
            if endswith(file, ".jl") && !(file in exclude)
                @testset "$(file)" begin
                    @info file
                    Random.seed!(12345)
                    include(joinpath(root, file))
                end
            end
        end
    end
    return
end

@testset "PortfolioOpt.jl" begin
    util_test_directory(".", ["runtests.jl", "generated_data.jl"])
    # util_test_directory(joinpath(dirname(@__DIR__), "docs", "src", "examples"))
end
