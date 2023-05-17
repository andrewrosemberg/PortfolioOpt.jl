# This is a very relaxed tests. It only checks that the conditional mean is
# between the mean and the conditional mean of α=0.0. It does not check that the
# conditional mean is the correct value.
# A check with the analytical solution would be better, but it is not numerically
# stable.

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

    @objective(model, Max, er_pointer)
    JuMP.optimize!(model)

    @test value(er_pointer) <= dot(mean(s), value.(w))
    @test value(er_pointer) >= value(er_pointer2)
end