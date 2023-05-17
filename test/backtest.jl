@testset "backtest.jl" begin
    @testset "AbstractRecorder" begin
        dates = Date(2023,1,1):Day(1):Date(2023,1,5)
        t = Date(2023,1,2)
        numA = 3
        budget=10.0
        volume_bids = [0.1, 0.3, 0.7]
        clearing_prices = [3.0, 2.0, 1.0]
        market = VolumeMarket(numA, budget=budget, volume_fee=0.1, allow_short_selling=true, risk_free_rate=0.05)
        @testset "WealthRecorder" begin
            recorder = PortfolioOpt.WealthRecorder(dates)
            @test length(recorder) == length(dates) + 1
            PortfolioOpt.record!(recorder, market, t)
            @test get_records(recorder, t) == budget
        end
        @testset "ReturnsRecorder" begin
            recorder = PortfolioOpt.ReturnsRecorder(dates)
            change_bids!(market, volume_bids)
            clear_market!(market, clearing_prices)
            @test length(recorder) == length(dates)
            PortfolioOpt.record!(recorder, market, t)
            @test get_records(recorder, t) == 0.2045
        end
        @testset "DecisionRecorder" begin
            recorder = PortfolioOpt.DecisionRecorder(numA, dates)
            @test length(recorder) == length(dates)
            PortfolioOpt.record!(recorder, market, t)
            @test get_records(recorder, t) == volume_bids
        end
    end
    @testset "change_bids!" begin
        numA = 3
        d = generate_gaussian_distribution(numA, rng)
        max_std = 0.03
        m = VolumeMarket(numA)
        con = RiskConstraint(SqrtVariance(d), LessThan(max_std))
        obj = ObjectiveTerm(ExpectedReturn(d))
        formulation = PortfolioFormulation(MAX_SENSE, obj, con)
        pointers = change_bids!(m, formulation, DEFAULT_SOLVER; record_measures=true)
        @test length(pointers) == 2
        risk_variable_id = PortfolioOpt.uid(con)
        @test isapprox(value(pointers[risk_variable_id]), max_std)
    end
    @testset "day_backtest_market!" begin
        # read prices
        prices = get_test_data()
        numD, numA = size(prices) # A: Assets    D: Days
        # calculating returns
        returns_series = percentchange(prices)
        # dates for the backtest
        date_range = timestamp(returns_series)[100:102]
        # backtest
        state_recorders = PortfolioOpt.default_state_recorders(numA, date_range)
        PortfolioOpt.day_backtest_market!(
            VolumeMarketHistory(returns_series), date_range[1],
            state_recorders=state_recorders,
            provide_state=true
        ) do market, past_returns, ext
            numD, numA = size(past_returns)
            change_bids!(market, fill(market_budget(market)/numA, numA))
        end
        _record = get_records(state_recorders[:returns])
        @test _record.data isa Vector
        @test isapprox(_record.data[1], sum(values(returns_series[date_range])[1,:]) / numA)
    end
    @testset "sequential_backtest_market" begin
        # read prices
        prices = get_test_data()
        numD, numA = size(prices) # A: Assets    D: Days
        # calculating returns
        returns_series = percentchange(prices)
        # dates for the backtest
        date_range = timestamp(returns_series)[100:102]
        # backtest
        state_recorders = PortfolioOpt.default_state_recorders(numA, date_range)
        sequential_backtest_market(
            VolumeMarketHistory(returns_series), date_range,
            state_recorders=state_recorders,
        ) do market, past_returns, ext
            numD, numA = size(past_returns)
            change_bids!(market, fill(market_budget(market)/numA, numA))
        end
        _record = get_records(state_recorders[:wealth])
        @test _record.data isa Vector
        @test isapprox(_record.data[2], 1.0 + sum(values(returns_series[date_range])[1,:]) / numA)
    end
end
