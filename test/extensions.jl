# Extensions defined outside of the package, using only the documented extension points.

# An ambiguity set: box around the mean, |μ_i - r̂_i| ≤ δ
struct TestBoxSet{T<:Real,D<:Sampleable} <: CenteredAmbiguitySet{T,D}
    d::D
    δ::T
end
PortfolioOpt.distribution(s::TestBoxSet) = s.d

# worst case expected return over the box: r̂'w - δ ||w||_1
function PortfolioOpt.calculate_measure!(m::ExpectedReturn{<:TestBoxSet}, w)
    model = JuMP.owner_model(first(w))
    s = ambiguityset(m)
    t = @variable(model)
    @constraint(model, [t; w] in MOI.NormOneCone(length(w) + 1))
    return dot(mean(s), w) - s.δ * t
end

# A risk measure: mean absolute deviation over samples
struct TestMeanAbsoluteDeviation{S<:DeterministicSamples} <: PortfolioRiskMeasure{S}
    ambiguity_set::S
end
function PortfolioOpt.calculate_measure!(m::TestMeanAbsoluteDeviation, w)
    model = JuMP.owner_model(first(w))
    samples = rand(ambiguityset(m))
    r̄ = mean(ambiguityset(m))
    S = size(samples, 2)
    u = @variable(model, [1:S], lower_bound = 0)
    @constraint(model, [s=1:S], u[s] >= dot(samples[:, s] - r̄, w))
    @constraint(model, [s=1:S], u[s] >= -dot(samples[:, s] - r̄, w))
    return sum(u) / S
end

# A term that is not a risk measure
struct TestLinearCost <: AbstractPortfolioTerm
    c::Vector{Float64}
end
PortfolioOpt.calculate_measure!(m::TestLinearCost, w) = dot(m.c, w)

# A market: money not invested earns nothing, no fees
mutable struct TestCashMarket <: PortfolioMarket
    n::Int
    budget::Float64
    bids::Union{Vector{Float64},Nothing}
    returns::Union{Vector{Float64},Nothing}
end
TestCashMarket(n; budget=1.0) = TestCashMarket(n, budget, nothing, nothing)
PortfolioOpt.num_assets(m::TestCashMarket) = m.n
PortfolioOpt.market_budget(m::TestCashMarket) = m.budget
PortfolioOpt.set_market_budget(m::TestCashMarket, v) = (m.budget = v)
PortfolioOpt.bids(m::TestCashMarket) = m.bids
PortfolioOpt.set_bids!(m::TestCashMarket, b::AbstractVector) = (m.bids = collect(Float64, b))
PortfolioOpt.clear_market!(m::TestCashMarket, r::AbstractVector) = (m.returns = r)
PortfolioOpt.total_profit(m::TestCashMarket) = dot(m.bids, m.returns)

# A market history for the new market (dates are integers)
struct TestCashHistory{N} <: MarketHistory{Float64,N}
    market::TestCashMarket
    returns::Matrix{Float64} # T × N
end
TestCashHistory(returns::Matrix{Float64}) = TestCashHistory{size(returns, 2)}(TestCashMarket(size(returns, 2)), returns)
PortfolioOpt.num_assets(::TestCashHistory{N}) where {N} = N
PortfolioOpt.past_prices(h::TestCashHistory, t) = h.returns[1:(t - 1), :]
PortfolioOpt.current_prices(h::TestCashHistory, t) = h.returns[t, :]
PortfolioOpt.market_template(h::TestCashHistory) = h.market
PortfolioOpt.market_template(h::TestCashHistory, t) = h.market

# A forecaster: Gaussian with the covariance shrunk to its diagonal
struct TestDiagonalGaussian <: AbstractForecaster end
PortfolioOpt.forecast(::TestDiagonalGaussian, returns::AbstractMatrix) =
    MvNormal(vec(mean(returns; dims=1)), Diagonal(vec(var(returns; dims=1))))

@testset "extensions" begin
    numA = 3
    d = generate_gaussian_distribution(numA, rng)
    samples = DeterministicSamples(rand(rng, d, 50))

    @testset "type hierarchy" begin
        @test ExpectedReturn(d) isa AbstractPortfolioTerm
        @test ConeRegularizer(MOI.NormOneCone(numA + 1)) isa AbstractPortfolioTerm
        @test VolumeMarket(numA) isa PortfolioMarket
        @test supertype(PortfolioMarket) === Any # standalone interface
    end

    @testset "external ambiguity set" begin
        δ = 0.01
        s = TestBoxSet(d, δ)
        @test length(s) == numA
        @test mean(s) == mean(d)
        model = Model(DEFAULT_SOLVER)
        w = @variable(model, [1:numA])
        fix.(w, [0.5, -0.2, 0.3])
        er = calculate_measure!(ExpectedReturn(s), w)
        @objective(model, Max, er)
        optimize!(model)
        @test isapprox(value(er), dot(mean(d), value.(w)) - δ * 1.0; atol=1e-6)
        # used in a formulation
        f = PortfolioFormulation(MAX_SENSE, ObjectiveTerm(ExpectedReturn(s)), RiskConstraint(SqrtVariance(d), LessThan(0.03)))
        w_box = portfolio_weights(f, numA, DEFAULT_SOLVER)
        @test length(w_box) == numA
        @test sum(w_box) <= 1 + 1e-6
    end

    @testset "external risk measure and term" begin
        mad = TestMeanAbsoluteDeviation(samples)
        cost = TestLinearCost(fill(0.001, numA))
        con = RiskConstraint(mad, LessThan(0.01))
        f = PortfolioFormulation(MAX_SENSE, [ObjectiveTerm(ExpectedReturn(samples)), ObjectiveTerm(cost, -1.0)], con)
        market = VolumeMarket(numA)
        pointers = change_bids!(market, f, DEFAULT_SOLVER; record_measures=true)
        @test length(pointers) == 3
        @test value(pointers[uid(con)]) <= 0.01 + 1e-6
        w = bids(market)
        empirical_mad = mean(abs.((rand(samples) .- mean(samples))' * w))
        @test isapprox(value(pointers[uid(con)]), empirical_mad; atol=1e-6)
    end

    @testset "external market" begin
        market = TestCashMarket(numA; budget=2.0)
        @test length(market) == numA
        f = PortfolioFormulation(MAX_SENSE, ObjectiveTerm(ExpectedReturn(d)), RiskConstraint(SqrtVariance(d), LessThan(0.03)))
        change_bids!(market, f, DEFAULT_SOLVER)
        @test length(bids(market)) == numA
        @test all(bids(market) .>= -1e-6)
        @test sum(bids(market)) <= 2.0 + 1e-6
        # build the model, add a position limit and solve
        model, w, pointers = build_portfolio_model(market, f, DEFAULT_SOLVER)
        @test pointers === nothing
        @test termination_status(model) == MOI.OPTIMIZE_NOT_CALLED
        @constraint(model, w .<= 0.2)
        change_bids!(market, model, w)
        @test all(bids(market) .<= 0.2 + 1e-6)
        # recorders only use the interface
        dates = 1:3
        wealth = PortfolioOpt.WealthRecorder(dates)
        returns = PortfolioOpt.ReturnsRecorder(dates)
        decisions = PortfolioOpt.DecisionRecorder(numA, dates)
        clear_market!(market, [0.01, 0.02, 0.03])
        for r in (wealth, returns, decisions)
            record!(r, market, 1)
        end
        @test get_records(wealth, 1) == 2.0
        @test get_records(returns, 1) ≈ dot(bids(market), [0.01, 0.02, 0.03]) / 2.0
        @test get_records(decisions, 1) == bids(market)
    end

    @testset "external market history" begin
        T = 30
        returns = collect(rand(rng, d, T)') # T × N
        hist = TestCashHistory(returns)
        date_range = 21:T
        recorders, _ = sequential_backtest_market(hist, date_range) do market, past_returns, ext
            @test size(past_returns, 1) == ext[:date] - 1
            change_bids!(market, fill(market_budget(market) / numA, numA))
        end
        wealth = get_records(recorders[:wealth])
        @test wealth[21] == 1.0
        expected = prod(1 .+ vec(mean(returns[date_range, :]; dims=2)))
        @test isapprox(wealth[T + 1], expected; rtol=1e-10)
    end

    @testset "build_portfolio_model" begin
        market = VolumeMarket(numA)
        con = RiskConstraint(SqrtVariance(d), LessThan(0.03))
        f = PortfolioFormulation(MAX_SENSE, ObjectiveTerm(ExpectedReturn(d)), con)
        model, w, pointers = build_portfolio_model(market, f, DEFAULT_SOLVER; record_measures=true)
        @test length(w) == numA
        @test haskey(pointers, uid(con))
        @constraint(model, w[1] == 0.0) # e.g. an asset that cannot be traded
        change_bids!(market, model, w)
        @test isapprox(bids(market)[1], 0.0; atol=1e-6)
        @test value(pointers[uid(con)]) <= 0.03 + 1e-6
    end

    @testset "forecasters" begin
        returns = collect(rand(rng, d, 40)') # T × N
        g = forecast(EmpiricalGaussian(20), returns)
        @test g isa MvNormal
        @test mean(g) ≈ vec(mean(returns[21:40, :]; dims=1))
        @test cov(g) ≈ cov(returns[21:40, :])
        @test mean(forecast(EmpiricalGaussian(), returns)) ≈ vec(mean(returns; dims=1))
        s = forecast(EmpiricalSamples(10), returns)
        @test s isa DeterministicSamples
        @test size(s) == (numA, 10) # n × T
        @test rand(s) == returns[31:40, :]'
        @test size(forecast(EmpiricalSamples(100), returns)) == (numA, 40)
        @test_throws ArgumentError EmpiricalGaussian(1)
        @test_throws ArgumentError EmpiricalSamples(0)
        # TimeArray input
        prices = get_test_data()
        returns_series = percentchange(prices)
        @test mean(forecast(EmpiricalGaussian(30), returns_series)) ≈ vec(mean(values(returns_series)[(end - 29):end, :]; dims=1))
        # external forecaster
        dg = forecast(TestDiagonalGaussian(), returns)
        @test cov(dg) ≈ Diagonal(vec(var(returns; dims=1)))
        @test_throws ArgumentError forecast(TestDiagonalGaussian(), "not returns")
        # forecasts plug into the measures
        f = PortfolioFormulation(MAX_SENSE,
            ObjectiveTerm(ExpectedReturn(forecast(EmpiricalGaussian(20), returns))),
            RiskConstraint(ConditionalExpectedReturn(forecast(EmpiricalSamples(20), returns); α=0.1), GreaterThan(-0.05)),
        )
        @test length(portfolio_weights(f, numA, DEFAULT_SOLVER)) == numA
    end

    @testset "WorstCaseReturn" begin
        model = Model(DEFAULT_SOLVER)
        w = @variable(model, [1:numA])
        fix.(w, ones(numA) / numA)
        wc = calculate_measure!(WorstCaseReturn(samples), w)
        cvar0 = calculate_measure!(ConditionalExpectedReturn(samples; α=0.0), w)
        @objective(model, Max, wc + cvar0)
        optimize!(model)
        expected = minimum(rand(samples)' * value.(w))
        @test isapprox(value(wc), expected; atol=1e-6)
        @test isapprox(value(cvar0), expected; atol=1e-6)
    end

    @testset "primal_cone" begin
        @test PortfolioOpt.primal_cone(1) == MOI.NormOneCone
        @test PortfolioOpt.primal_cone(2.0) == MOI.SecondOrderCone
        @test PortfolioOpt.primal_cone(Inf) == MOI.NormInfinityCone
        @test DuWassersteinBall(samples; norm_cone=1) isa DuWassersteinBall
        @test_throws ArgumentError DuWassersteinBall(samples; norm_cone=3.0)
    end

    @testset "Variance with expressions" begin
        model = Model(DEFAULT_SOLVER)
        w = @variable(model, [1:numA])
        fix.(w, ones(numA) / numA)
        v_aff = calculate_measure!(Variance(d), 2.0 .* w) # Vector{AffExpr}
        v_view = calculate_measure!(Variance(d), view(w, 1:numA)) # SubArray of variables
        @test v_view isa QuadExpr
        @objective(model, Min, v_aff)
        optimize!(model)
        @test isapprox(value(v_aff), 4 * value.(w)' * cov(d) * value.(w); atol=1e-6)
    end

    @testset "no type piracy" begin
        @test !hasmethod(JuMP.owner_model, Tuple{Vector{VariableRef}})
        @test !any(m -> m.module === PortfolioOpt.TestUtils, methods(keys))
    end
end
