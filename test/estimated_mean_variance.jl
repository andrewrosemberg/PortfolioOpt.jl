@testset "Estimated Case" begin
    numA = 3
    d = generate_gaussian_distribution(numA, rng)
    
    model = Model(DEFAULT_SOLVER)
    w = @variable(model, [1:numA])
    fix.(w, ones(numA)/numA)

    er = ExpectedReturn(d)
    er_pointer = PortfolioOpt.calculate_measure!(er, w)

    variance = Variance(d)
    var_pointer = PortfolioOpt.calculate_measure!(variance, w)
    var_pointer_rotated_SOC = PortfolioOpt.calculate_measure!(variance, w .+ 0.0)

    sqrtvar = SqrtVariance(d)
    sqrtvar_pointer = PortfolioOpt.calculate_measure!(sqrtvar, w)

    @objective(model, Min, var_pointer + var_pointer_rotated_SOC + sqrtvar_pointer)
    JuMP.optimize!(model)

    @test isapprox(value(er_pointer), mean(d)' * value.(w), atol=1e-4)
    @test isapprox(value(var_pointer), value.(w)' * cov(d) * value.(w), atol=1e-4)
    @test isapprox(value(var_pointer), value(var_pointer_rotated_SOC), atol=1e-2)
    @test isapprox(value(sqrtvar_pointer), sqrt(value(var_pointer)), atol=1e-2)
end
