@testset "CenteredAmbiguitySet" begin
    numA = 3
    d = generate_gaussian_distribution(numA, rng)
    @testset "MomentUncertainty" begin
        @testset "constructor" begin
            @test MomentUncertainty(d, 0.1, 1.5) isa MomentUncertainty
            @test MomentUncertainty(d; γ1=0.1, γ2=1.5) isa MomentUncertainty
            @test_throws ArgumentError MomentUncertainty(d, -0.1, 1.5)
            @test_throws ArgumentError MomentUncertainty(d, 0.1, 0.1)
        end
        ambiguity_set = MomentUncertainty(d, 0.1, 1.5)
        @testset "distribution" begin
            @test PortfolioOpt.distribution(ambiguity_set) == d
        end
        @testset "ExpectedUtility" begin
            measure = ExpectedUtility(ambiguity_set, PieceWiseUtility())
            model = Model(DEFAULT_SOLVER)
            w = @variable(model, [1:numA])
            fix.(w, ones(numA)/numA)
            er_pointer = PortfolioOpt.calculate_measure!(measure, w)

            @objective(model, Max, er_pointer)
            JuMP.optimize!(model)

            # has to be more concervative than the expected return under EllipticalSet
            @test value(er_pointer) <= mean(d)' * value.(w) - sqrt(value.(w)' * cov(d) * value.(w)) * ambiguity_set.γ1
        end
        @testset "support constructor" begin
            μ = mean(d)
            s = MomentUncertainty(d, 0.1, 1.5, μ .- 1.0, μ .+ 1.0)
            @test s isa MomentUncertainty
            @test s.lower == μ .- 1.0
            @test MomentUncertainty(d; γ1=0.1, γ2=1.5, upper=μ .+ 1.0).upper == μ .+ 1.0
            @test all(isinf, MomentUncertainty(d; γ1=0.1, γ2=1.5).lower)
            # the mean has to be inside the support
            @test_throws ArgumentError MomentUncertainty(d, 0.1, 1.5, μ .+ 1.0, μ .+ 2.0)
            @test_throws ArgumentError MomentUncertainty(d, 0.1, 1.5, μ[1:2], μ .+ 1.0)
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
        w0 = ones(numA) / numA
        μ = mean(d)
        σ = sqrt(w0' * cov(d) * w0)

        @testset "ExpectedReturn" begin
            # worst case mean: r̂'w - sqrt(γ1) σ
            v = fixed_portfolio_value(ExpectedReturn(MomentUncertainty(d, 0.1, 1.5)), w0)
            @test isapprox(v, μ' * w0 - sqrt(0.1) * σ; atol=1e-4)
        end

        @testset "ConditionalExpectedReturn" begin
            for α in (0.05, 0.3), γ2 in (1.0, 1.5)
                # known mean and covariance bounded by γ2 Σ: r̂'w - sqrt(γ2 (1 - α) / α) σ
                v = fixed_portfolio_value(ConditionalExpectedReturn(MomentUncertainty(d, 0.0, γ2); α=α), w0)
                @test isapprox(v, μ' * w0 - sqrt(γ2 * (1 - α) / α) * σ; atol=1e-4)
            end
            # α = 1 is the worst case expected return
            s = MomentUncertainty(d, 0.1, 1.5)
            @test isapprox(
                fixed_portfolio_value(ConditionalExpectedReturn(s; α=1.0), w0),
                fixed_portfolio_value(ExpectedReturn(s), w0); atol=1e-5,
            )
            # the worst case is more conservative when the mean is uncertain
            @test fixed_portfolio_value(ConditionalExpectedReturn(s; α=0.1), w0) <
                fixed_portfolio_value(ConditionalExpectedReturn(MomentUncertainty(d, 0.0, 1.5); α=0.1), w0)
            # the worst case return (α = 0) is unbounded
            model = Model(DEFAULT_SOLVER)
            w = @variable(model, [1:numA])
            @test_throws ArgumentError PortfolioOpt.calculate_measure!(ConditionalExpectedReturn(s; α=0.0), w)
        end

        @testset "bounded support" begin
            α = 0.1
            unbounded = MomentUncertainty(d, 0.0, 1.0)
            v_unbounded = fixed_portfolio_value(ConditionalExpectedReturn(unbounded; α=α), w0)
            # a far away support does not change the worst case
            far = MomentUncertainty(d, 0.0, 1.0, μ .- 1e3, μ .+ 1e3)
            @test isapprox(fixed_portfolio_value(ConditionalExpectedReturn(far; α=α), w0), v_unbounded; atol=1e-3)
            # a tight support makes the worst case less extreme, but not better than the mean
            δ = sqrt.(diag(cov(d)))
            tight = MomentUncertainty(d, 0.0, 1.0, μ .- δ, μ .+ δ)
            v_tight = fixed_portfolio_value(ConditionalExpectedReturn(tight; α=α), w0)
            @test v_tight > v_unbounded + 1e-3
            @test v_tight <= μ' * w0 + 1e-5
            # the worst case CVaR within the box is at least the worst return in the box
            @test v_tight >= min.(w0 .* (μ .- δ), w0 .* (μ .+ δ)) |> sum
            # only one side bounded
            below = MomentUncertainty(d; γ1=0.0, γ2=1.0, lower=μ .- δ)
            v_below = fixed_portfolio_value(ConditionalExpectedReturn(below; α=α), w0)
            @test v_unbounded - 1e-5 <= v_below <= v_tight + 1e-5
            # utility with bounded support
            v_utility = fixed_portfolio_value(ExpectedUtility(tight, PieceWiseUtility()), w0)
            @test isapprox(v_utility, μ' * w0; atol=1e-4)
        end

        @testset "portfolio" begin
            formulation = PortfolioFormulation(MAX_SENSE,
                ObjectiveTerm(ConditionalExpectedReturn(MomentUncertainty(d, 0.05, 1.5); α=0.1)),
            )
            w = portfolio_weights(formulation, numA, DEFAULT_SOLVER)
            @test length(w) == numA
            @test all(w .>= -1e-6)
            @test sum(w) <= 1.0 + 1e-6
        end
    end
    @testset "DuWassersteinBall" begin
        s = DeterministicSamples(rand(rng, d, 100))
        @testset "constructor" begin
            @test DuWassersteinBall(s, 0.1, 1000.0, rand(numA, numA), 2.0) isa DuWassersteinBall
            @test DuWassersteinBall(s; ϵ=0.02,
                norm_cone=Inf,
                Λ=100.0,
                Q=rand(numA, numA)
            ) isa DuWassersteinBall
            @test_throws ArgumentError DuWassersteinBall(s, -0.1, 1000.0, rand(numA, numA), 2.0)
            @test_throws ArgumentError DuWassersteinBall(s, 0.1, -1000.0, rand(numA, numA), 2.0)
            @test_throws ArgumentError DuWassersteinBall(s, 0.1, 1000.0, rand(5, numA), 3.0)
        end
        @testset "distribution" begin
            @test PortfolioOpt.distribution(DuWassersteinBall(s)) == s
        end
        @testset "sample_size" begin
            @test PortfolioOpt.sample_size(DuWassersteinBall(s)) == PortfolioOpt.sample_size(s)
        end
        @testset "ExpectedReturn" begin
            measure = ExpectedReturn(DuWassersteinBall(s))
            model = Model(DEFAULT_SOLVER)
            w = @variable(model, [1:numA])
            fix.(w, ones(numA)/numA)
            er_pointer = PortfolioOpt.calculate_measure!(measure, w)

            @objective(model, Max, er_pointer)
            JuMP.optimize!(model)

            # the worst case expected return over the ball is strictly below the empirical mean
            @test value(er_pointer) < mean(s)' * value.(w) - 1e-4

            # a ball with zero radius only contains the empirical distribution
            model = Model(DEFAULT_SOLVER)
            w = @variable(model, [1:numA])
            fix.(w, ones(numA)/numA)
            er_pointer = PortfolioOpt.calculate_measure!(ExpectedReturn(DuWassersteinBall(s; ϵ=0.0)), w)
            @objective(model, Max, er_pointer)
            JuMP.optimize!(model)
            @test termination_status(model) == MOI.OPTIMAL
            @test isapprox(value(er_pointer), mean(s)' * value.(w); atol=1e-5)
        end
    end
end