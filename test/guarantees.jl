# Named guarantees.
#
# Each testset checks numerically one mathematical claim stated in the documentation
# ("Theory" pages and docstrings). The name of the testset is the name of the claim, so that a
# formal proof can later be attached to the same name. See docs/src/theory/guarantees.md.

@testset "guarantees" begin
    grng = MersenneTwister(2024)
    numA = 4
    num_instances = 3
    tol = 1e-5

    # value of `measure` for the fixed portfolio `w0`, maximizing (or minimizing) over the
    # auxiliary variables of its reformulation
    function fixed_value(measure, w0; sense=MAX_SENSE)
        model = Model(DEFAULT_SOLVER)
        w = @variable(model, [1:length(w0)])
        fix.(w, w0)
        pointer = PortfolioOpt.calculate_measure!(measure, w)
        @objective(model, sense, pointer)
        JuMP.optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        return value(pointer)
    end

    random_portfolio(rng, n) = (x = rand(rng, n); x / sum(x))

    # lower tail mean of the portfolio returns `x` with equal probabilities: the expected return
    # over the worst `α` fraction of outcomes (fractional atom included)
    function sorted_tail_mean(x::AbstractVector, α::Real)
        S = length(x)
        xs = sort(x)
        K = floor(Int, α * S)
        tail = sum(xs[1:K]; init=0.0) / S
        K < S && (tail += (α - K / S) * xs[K + 1])
        return tail / α
    end

    @testset "cvar_ru_lower_bound" begin
        # Rockafellar & Uryasev (2000): for every z, z - E[(z - R)^+] / α <= -CVaR_α(R)
        for _ in 1:num_instances, α in (0.05, 0.1, 0.37)
            x = randn(grng, 50)
            exact = sorted_tail_mean(x, α)
            for z in randn(grng, 20)
                @test z - mean(max.(z .- x, 0.0)) / α <= exact + 1e-12
            end
        end
    end

    @testset "cvar_ru_exact" begin
        # the conic (LP) reformulation equals the sample conditional expected return, also when
        # α S is not an integer
        for _ in 1:num_instances, α in (0.05, 0.1, 0.37, 1.0)
            d = generate_gaussian_distribution(numA, grng)
            s = DeterministicSamples(rand(grng, d, 30))
            w0 = random_portfolio(grng, numA)
            x = rand(s)' * w0
            @test isapprox(fixed_value(ConditionalExpectedReturn(s; α=α), w0), sorted_tail_mean(x, α); atol=tol)
        end
    end

    @testset "worst_case_return_exact" begin
        for _ in 1:num_instances
            d = generate_gaussian_distribution(numA, grng)
            s = DeterministicSamples(rand(grng, d, 30))
            w0 = random_portfolio(grng, numA)
            @test isapprox(fixed_value(WorstCaseReturn(s), w0), minimum(rand(s)' * w0); atol=tol)
            @test isapprox(fixed_value(ConditionalExpectedReturn(s; α=0.0), w0), minimum(rand(s)' * w0); atol=tol)
        end
    end

    @testset "variance_epigraph_exact" begin
        for _ in 1:num_instances
            d = generate_gaussian_distribution(numA, grng)
            w0 = random_portfolio(grng, numA)
            v = w0' * cov(d) * w0
            # affine expressions of the variables use the rotated second order cone epigraph
            model = Model(DEFAULT_SOLVER)
            w = @variable(model, [1:numA])
            fix.(w, w0)
            p = PortfolioOpt.calculate_measure!(Variance(d), w .+ 0.0)
            @objective(model, Min, p)
            JuMP.optimize!(model)
            @test isapprox(value(p), v; rtol=1e-4)
            @test isapprox(fixed_value(SqrtVariance(d), w0; sense=MIN_SENSE), sqrt(v); rtol=1e-4)
        end
    end

    @testset "elliptical_closed_form" begin
        # min over ||Σ^{-1/2}(μ - r̂)|| <= δ of μ'w is r̂'w - δ ||Σ^{1/2} w||
        for _ in 1:num_instances, δ in (0.0, 0.1, 0.5)
            d = generate_gaussian_distribution(numA, grng)
            w0 = random_portfolio(grng, numA)
            @test isapprox(
                fixed_value(ExpectedReturn(EllipticalSet(d, δ)), w0),
                mean(d)' * w0 - δ * sqrt(w0' * cov(d) * w0); atol=tol,
            )
        end
    end

    @testset "budget_set_strong_duality" begin
        # the dual expression equals the primal worst case over the budget uncertainty set,
        # computed by solving the primal LP and by the greedy closed form
        for _ in 1:num_instances, Γ in (0.0, 1.5, 3.0)
            d = generate_gaussian_distribution(numA, grng)
            Δ = rand(grng, numA)
            w0 = randn(grng, numA) # also with short positions
            dual_value = fixed_value(ExpectedReturn(BudgetSet(d, Δ, Γ)), w0)

            primal = Model(DEFAULT_SOLVER)
            @variable(primal, μ[1:numA])
            @variable(primal, 0 <= z[1:numA] <= 1)
            @constraint(primal, μ .<= mean(d) .+ z .* Δ)
            @constraint(primal, μ .>= mean(d) .- z .* Δ)
            @constraint(primal, sum(z) <= Γ)
            @objective(primal, Min, μ' * w0)
            JuMP.optimize!(primal)
            @test isapprox(dual_value, objective_value(primal); atol=tol)

            # greedy: deviate the Γ assets with the largest Δ_i |w_i| (the last one fractionally)
            g = sort(Δ .* abs.(w0); rev=true)
            K = floor(Int, Γ)
            deviation = sum(g[1:min(K, numA)]; init=0.0) + (K < numA ? (Γ - K) * g[K + 1] : 0.0)
            @test isapprox(dual_value, mean(d)' * w0 - deviation; atol=tol)
        end
    end

    @testset "moment_worst_case_le_nominal" begin
        # the nominal distribution belongs to the set: the worst case expected return is at
        # most the nominal mean return, for every portfolio and at the optimum
        for _ in 1:num_instances
            d = generate_gaussian_distribution(numA, grng)
            s = MomentUncertainty(d, 0.1, 1.5)
            w0 = random_portfolio(grng, numA)
            @test fixed_value(ExpectedReturn(s), w0) <= mean(d)' * w0 + tol

            robust = PortfolioFormulation(MAX_SENSE, ObjectiveTerm(ExpectedReturn(s)))
            nominal = PortfolioFormulation(MAX_SENSE, ObjectiveTerm(ExpectedReturn(distribution(s))))
            w_robust = portfolio_weights(robust, numA, DEFAULT_SOLVER)
            w_nominal = portfolio_weights(nominal, numA, DEFAULT_SOLVER)
            robust_optimum = fixed_value(ExpectedReturn(s), w_robust)
            @test robust_optimum <= mean(d)' * w_nominal + tol
            # the robust portfolio is optimal for the worst case
            @test robust_optimum >= fixed_value(ExpectedReturn(s), w_nominal) - tol
        end
    end

    @testset "moment_monotone_in_radius" begin
        # larger γ1 or γ2 means a larger set, hence a smaller (or equal) worst case
        d = generate_gaussian_distribution(numA, grng)
        w0 = random_portfolio(grng, numA)
        values_γ1 = [fixed_value(ConditionalExpectedReturn(MomentUncertainty(d, γ1, 1.2); α=0.1), w0) for γ1 in (0.0, 0.05, 0.2)]
        @test issorted(values_γ1; rev=true, lt=(a, b) -> a < b - tol)
        values_γ2 = [fixed_value(ConditionalExpectedReturn(MomentUncertainty(d, 0.05, γ2); α=0.1), w0) for γ2 in (1.0, 1.5, 3.0)]
        @test issorted(values_γ2; rev=true, lt=(a, b) -> a < b - tol)
    end

    @testset "moment_cvar_le_member_cvar" begin
        # the Gaussian with the nominal moments belongs to the set (unbounded support), so the
        # worst case conditional expected return is at most its Gaussian value
        # μ'w - σ φ(Φ⁻¹(α)) / α
        for _ in 1:num_instances, α in (0.05, 0.2)
            d = generate_gaussian_distribution(numA, grng)
            w0 = random_portfolio(grng, numA)
            σ = sqrt(w0' * cov(d) * w0)
            gaussian = mean(d)' * w0 - σ * pdf(Normal(), quantile(Normal(), α)) / α
            @test fixed_value(ConditionalExpectedReturn(MomentUncertainty(d, 0.0, 1.0); α=α), w0) <= gaussian + tol
        end
    end

    @testset "moment_bounded_support_between" begin
        # restricting the support removes distributions: unbounded <= bounded <= nominal
        d = generate_gaussian_distribution(numA, grng)
        w0 = random_portfolio(grng, numA)
        μ = mean(d)
        δ = 2 * sqrt.(diag(cov(d)))
        for γ1 in (0.0, 0.1)
            unbounded = fixed_value(ConditionalExpectedReturn(MomentUncertainty(d, γ1, 1.5); α=0.1), w0)
            bounded = fixed_value(ConditionalExpectedReturn(MomentUncertainty(d, γ1, 1.5, μ .- δ, μ .+ δ); α=0.1), w0)
            @test unbounded <= bounded + tol
            @test bounded <= μ' * w0 + tol
        end
    end

    @testset "wasserstein_zero_radius_is_empirical" begin
        d = generate_gaussian_distribution(numA, grng)
        s = DeterministicSamples(rand(grng, d, 40))
        w0 = random_portfolio(grng, numA)
        @test isapprox(fixed_value(ExpectedReturn(DuWassersteinBall(s; ϵ=0.0)), w0), mean(s)' * w0; atol=tol)
    end

    @testset "wasserstein_monotone_in_radius" begin
        d = generate_gaussian_distribution(numA, grng)
        s = DeterministicSamples(rand(grng, d, 40))
        w0 = random_portfolio(grng, numA)
        for p in (1.0, 2.0, Inf)
            v = [fixed_value(ExpectedReturn(DuWassersteinBall(s; ϵ=ϵ, norm_cone=p)), w0) for ϵ in (0.0, 0.01, 0.05)]
            @test issorted(v; rev=true, lt=(a, b) -> a < b - tol)
            @test v[end] <= mean(s)' * w0 + tol
        end
    end

    @testset "certainty_equivalent_le_mean" begin
        # Jensen: the certainty equivalent of a concave utility is at most the expected return
        for _ in 1:num_instances, a in (0.5, 5.0)
            d = generate_gaussian_distribution(numA, grng)
            s = DeterministicSamples(rand(grng, d, 30))
            w0 = random_portfolio(grng, numA)
            u = ExponentialUtility(a)
            @test fixed_value(CertaintyEquivalent(d, u), w0) <= mean(d)' * w0 + tol
            @test fixed_value(CertaintyEquivalent(s, u), w0) <= mean(s)' * w0 + tol
            # E[u(R)] <= u(E[R])
            @test fixed_value(ExpectedUtility(s, u), w0) <= (1 - exp(-a * mean(s)' * w0)) / a + tol
        end
    end

    @testset "risk_constraint_feasible" begin
        # the optimal portfolio satisfies the risk constraint evaluated directly (not through the
        # reformulation)
        d = generate_gaussian_distribution(numA, grng)
        d = MvNormal(mean(d) / 10, cov(d) / 100)
        s = DeterministicSamples(rand(grng, d, 50))
        σ_max = 0.5 * sqrt(minimum(diag(cov(d))))
        f = PortfolioFormulation(MAX_SENSE, ObjectiveTerm(ExpectedReturn(d)), RiskConstraint(SqrtVariance(d), LessThan(σ_max)))
        w = portfolio_weights(f, numA, DEFAULT_SOLVER)
        @test sqrt(w' * cov(d) * w) <= σ_max + tol

        f = PortfolioFormulation(MAX_SENSE, ObjectiveTerm(ExpectedReturn(d)), RiskConstraint(Variance(d), LessThan(σ_max^2)))
        w = portfolio_weights(f, numA, DEFAULT_SOLVER)
        @test w' * cov(d) * w <= σ_max^2 + tol

        cvar_min = sorted_tail_mean(rand(s)' * (ones(numA) / numA), 0.1) / 2
        f = PortfolioFormulation(MAX_SENSE, ObjectiveTerm(ExpectedReturn(s)), RiskConstraint(ConditionalExpectedReturn(s; α=0.1), GreaterThan(cvar_min)))
        w = portfolio_weights(f, numA, DEFAULT_SOLVER)
        @test sorted_tail_mean(rand(s)' * w, 0.1) >= cvar_min - tol

        ce_min = 0.0
        f = PortfolioFormulation(MAX_SENSE, ObjectiveTerm(ExpectedReturn(d)), RiskConstraint(CertaintyEquivalent(d, ExponentialUtility(50.0)), GreaterThan(ce_min)))
        w = portfolio_weights(f, numA, DEFAULT_SOLVER)
        @test mean(d)' * w - 25.0 * w' * cov(d) * w >= ce_min - tol
    end

    @testset "cone_regularizer_exact" begin
        # a penalized norm equals the norm of the optimal portfolio
        d = generate_gaussian_distribution(numA, grng)
        reg_uid = uuid1()
        f = PortfolioFormulation(MAX_SENSE, [
            ObjectiveTerm(ExpectedReturn(d)),
            ObjectiveTerm(ConeRegularizer(MOI.SecondOrderCone(numA + 1)), -0.1, reg_uid),
        ])
        market = VolumeMarket(numA)
        model, w, pointers = build_portfolio_model(market, f, DEFAULT_SOLVER; record_measures=true)
        JuMP.optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test isapprox(value(pointers[reg_uid]), norm(value.(w)); atol=tol)
    end
end
