# Checks that the conditional mean is between the mean and the conditional mean
# of α=0.0, and that it matches the empirical expected shortfall of the samples.

@testset "conditional_mean" begin
    numA = 3
    d = generate_gaussian_distribution(numA, rng)
    s = DeterministicSamples(rand(rng, d, 100))
    measure = ConditionalExpectedReturn(s)
    measure2 = ConditionalExpectedReturn(s; α=0.0)

    model = Model(DEFAULT_SOLVER)
    w = @variable(model, [1:numA])
    fix.(w, ones(numA)/numA)
    er_pointer = PortfolioOpt.calculate_measure!(measure, w)
    er_pointer2 = PortfolioOpt.calculate_measure!(measure2, w)

    @objective(model, Max, er_pointer + er_pointer2)
    JuMP.optimize!(model)

    @test value(er_pointer) <= dot(mean(s), value.(w))
    @test value(er_pointer) >= value(er_pointer2)

    # empirical conditional mean: average of the worst α * N portfolio returns
    portfolio_returns = sort(rand(s)' * value.(w))
    @test isapprox(value(er_pointer), mean(portfolio_returns[1:5]); atol=1e-6)
    @test isapprox(value(er_pointer2), minimum(portfolio_returns); atol=1e-6)

    @testset "same measure used twice in a model" begin
        model = Model(DEFAULT_SOLVER)
        w = @variable(model, [1:numA])
        fix.(w, ones(numA)/numA)
        p1 = PortfolioOpt.calculate_measure!(ConditionalExpectedReturn(s; α=0.1), w)
        p2 = PortfolioOpt.calculate_measure!(ConditionalExpectedReturn(s; α=0.05), w)
        @objective(model, Max, p1 + p2)
        JuMP.optimize!(model)
        @test value(p2) <= value(p1) + 1e-6
    end
end