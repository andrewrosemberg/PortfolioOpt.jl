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
    
end