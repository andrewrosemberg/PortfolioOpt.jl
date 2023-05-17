@testset "VolumeMarket.jl" begin
    budget = 10.0
    fee = 0.1
    allow_short_selling = true
    rf = 0.05
    volume_bids = [0.2, 0.3, 0.5]
    clearing_prices = [1.0, 2.0, 3.0]

    m = VolumeMarket{Float64, 3}(budget, fee, allow_short_selling, rf, volume_bids, clearing_prices)

    @test market_budget(m) ≈ budget
    @test market_volume_fee(m) ≈ fee
    @test m.allow_short_selling == allow_short_selling
    @test m.risk_free_rate ≈ rf
    @test m.volume_bids == volume_bids
    @test m.clearing_prices == clearing_prices

    # Test the VolumeMarket constructors
    @testset "VolumeMarket constructors" begin
        m2 = VolumeMarket(3, budget=budget, volume_fee=fee, allow_short_selling=allow_short_selling, risk_free_rate=rf)
        @test market_budget(m2) ≈ budget
        @test market_volume_fee(m2) ≈ fee
        @test m2.allow_short_selling == allow_short_selling
        @test m2.risk_free_rate ≈ rf
        @test m2.volume_bids === nothing
        @test m2.clearing_prices === nothing
    end

    # Test the VolumeMarket setters
    @testset "VolumeMarket setters" begin
        set_market_budget(m, 15.0)
        @test market_budget(m) ≈ 15.0

        change_bids!(m, [0.1, 0.2, 0.7])
        @test m.volume_bids ≈ [0.1, 0.2, 0.7]

        clear_market!(m, [3.0, 2.0, 1.0])
        @test m.clearing_prices ≈ [3.0, 2.0, 1.0]
    end

    # Test the calculate_profit and total_profit functions
    @testset "Profit functions" begin
        profit = calculate_profit(m)
        expected_result = (cleared_volumes=[0.1, 0.2, 0.7], clearing_prices=[3.0, 2.0, 1.0], profit=1.4, risk_free_profit=0.7)
        @test all(k->getfield(profit,k) ≈ getfield(expected_result,k), keys(profit))
        @test total_profit(m) ≈ 2.1
    end

    @testset "VolumeMarket interface" begin
        @test risk_free_rate(m) ≈ rf
        @test eltype(m) == Float64
    end

    @testset "VolumeMarket errors" begin
        @test_throws MethodError clear_market!(m)
    end

    @testset "VolumeMarket type stability" begin
        # @inferred VolumeMarket{Float64, 3}(budget, fee, allow_short_selling, rf, volume_bids, clearing_prices)
        # @inferred VolumeMarket(3, budget=budget, volume_fee=fee, allow_short_selling=allow_short_selling, risk_free_rate=rf)
        @inferred set_market_budget(m, 15.0)
        @inferred change_bids!(m, [0.1, 0.2, 0.7])
        @inferred clear_market!(m, [3.0, 2.0, 1.0])
        # @inferred calculate_profit(m)
        @inferred total_profit(m)
        @inferred risk_free_rate(m)
        @inferred eltype(m)
    end

    @testset "market_model and change_bids!" begin
        jump_model, w = market_model(m, DEFAULT_SOLVER; sense=MAX_SENSE)
        @test  jump_model isa JuMP.Model
        @test  w isa Vector{VariableRef}
        @test  length(w) == 3
        change_bids!(m, jump_model, w)
        @test  isapprox(norm(m.volume_bids, 1), 0.0; atol=1e-4)
    end

    @testset "VolumeMarketHistory" begin
        # read prices
        prices = get_test_data();
        numD, numA = size(prices) # A: Assets    D: Days
        # calculating returns
        returns_series = percentchange(prices);
        # creating VolumeMarketHistory
        market = VolumeMarket(numA, budget=100.0, volume_fee=0.1, allow_short_selling=true, risk_free_rate=0.05)
        hist = VolumeMarketHistory(market, returns_series)
        # testing
        @test num_days(hist) == numD - 1
        @test num_assets(hist) == numA
        @test hist.history_clearing_prices == returns_series
        @test keys(hist) == timestamp(returns_series)
        @test all(values(hist.history_risk_free_rates) .== rf)
        @test market_template(hist) == market
        t = timestamp(returns_series)[10]
        @test current_prices(hist, t) == values(returns_series[t])[1,:]
        hist.history_risk_free_rates[t] = 0.0
        @test risk_free_rate(hist, t) == 0.0
        t_1 = timestamp(returns_series)[11]
        @test risk_free_rate(hist, t_1) == rf
        @test market_template(hist, t_1) == market
    end

end