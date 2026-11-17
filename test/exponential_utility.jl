@testset "ExponentialUtility" begin
    numA = 3
    d = generate_gaussian_distribution(numA, rng)
    a = 2.0
    u = ExponentialUtility(a)
    w0 = ones(numA) / numA

    @testset "constructor" begin
        @test risk_aversion(u) == a
        @test u == ExponentialUtility(2.0)
        @test_throws ArgumentError ExponentialUtility(0.0)
        @test_throws ArgumentError ExponentialUtility(-1.0)
    end

    # value of a measure for the fixed portfolio `w0`
    function fixed_portfolio_value(measure, w0)
        model = Model(DEFAULT_SOLVER)
        w = @variable(model, [1:length(w0)])
        fix.(w, w0)
        pointer = PortfolioOpt.calculate_measure!(measure, w)
        @objective(model, Max, pointer)
        JuMP.optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        return value(pointer)
    end

    @testset "Gaussian" begin
        m = mean(d)' * w0
        v = w0' * cov(d) * w0
        @test isapprox(fixed_portfolio_value(ExpectedUtility(d, u), w0), (1 - exp(-a * m + a^2 * v / 2)) / a; atol=1e-5)
        @test isapprox(fixed_portfolio_value(CertaintyEquivalent(d, u), w0), m - a * v / 2; atol=1e-5)
    end

    @testset "Gaussian mixture" begin
        d2 = MvNormal(mean(d) .- 0.5, cov(d) * 2)
        mix = MixtureModel([d, d2], [0.7, 0.3])
        moment_generating = sum(
            p * exp(-a * mean(c)' * w0 + a^2 * w0' * cov(c) * w0 / 2) for (p, c) in zip([0.7, 0.3], [d, d2])
        )
        @test isapprox(fixed_portfolio_value(ExpectedUtility(mix, u), w0), (1 - moment_generating) / a; atol=1e-5)
        @test isapprox(fixed_portfolio_value(CertaintyEquivalent(mix, u), w0), -log(moment_generating) / a; atol=1e-5)
        # a mixture with a single component is the Gaussian
        @test isapprox(
            fixed_portfolio_value(CertaintyEquivalent(MixtureModel([d], [1.0]), u), w0),
            fixed_portfolio_value(CertaintyEquivalent(d, u), w0); atol=1e-5,
        )
    end

    @testset "DeterministicSamples" begin
        s = DeterministicSamples(rand(rng, d, 50))
        returns = rand(s)' * w0
        @test isapprox(fixed_portfolio_value(ExpectedUtility(s, u), w0), mean(1 .- exp.(-a * returns)) / a; atol=1e-5)
        @test isapprox(fixed_portfolio_value(CertaintyEquivalent(s, u), w0), -log(mean(exp.(-a * returns))) / a; atol=1e-5)
    end

    @testset "optimal portfolio" begin
        # for a Gaussian, maximizing the expected exponential utility (or its certainty
        # equivalent) is the mean-variance problem with risk aversion a
        d_assets = MvNormal([0.10, 0.12, 0.08], [0.04 0.01 0.0; 0.01 0.09 0.02; 0.0 0.02 0.03])
        mean_variance_model = Model(DEFAULT_SOLVER)
        x = @variable(mean_variance_model, [1:numA], lower_bound = 0)
        @constraint(mean_variance_model, sum(x) <= 1)
        @objective(mean_variance_model, Max, mean(d_assets)' * x - a / 2 * x' * cov(d_assets) * x)
        JuMP.optimize!(mean_variance_model)
        w_mean_variance = value.(x)

        for measure in (ExpectedUtility(d_assets, u), CertaintyEquivalent(d_assets, u))
            w = portfolio_weights(PortfolioFormulation(MAX_SENSE, ObjectiveTerm(measure)), numA, DEFAULT_SOLVER)
            @test isapprox(w, w_mean_variance; atol=1e-4)
        end

        # constraint on the certainty equivalent
        ce_min = 0.005
        formulation = PortfolioFormulation(MAX_SENSE,
            ObjectiveTerm(ExpectedReturn(d_assets)),
            RiskConstraint(CertaintyEquivalent(d_assets, ExponentialUtility(20.0)), GreaterThan(ce_min)),
        )
        w = portfolio_weights(formulation, numA, DEFAULT_SOLVER)
        @test isapprox(mean(d_assets)' * w - 10.0 * w' * cov(d_assets) * w, ce_min; atol=1e-5) # binding
        @test sum(w) > 1e-3
    end
end
