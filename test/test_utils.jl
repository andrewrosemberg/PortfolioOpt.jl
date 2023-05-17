@testset "TestUtils" begin
    # read prices
    prices = get_test_data()
    @test size(prices, 2) == 6
    numD, numA = size(prices) # A: Assets    D: Days

    # calculating returns
    returns_series = percentchange(prices)
    returns = values(returns_series)
    Σ, r̄ = mean_variance(returns[:, :])
    @test Σ isa Matrix
    @test r̄ isa Vector

    w = max_sharpe(Σ, r̄, 0.0)
    @test w isa Vector
    @test length(w) == numA
end