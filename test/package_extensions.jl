# Integrations with optional packages (weak dependencies), loaded in runtests.jl
using OptimalBids: OptimalBids
using TimeSeries: TimeSeries

@testset "package extensions" begin
    @testset "core dependencies" begin
        # the optional packages are not dependencies of the package
        project = joinpath(pkgdir(PortfolioOpt), "Project.toml")
        deps = keys(Base.parsed_toml(project)["deps"])
        @test !("OptimalBids" in deps)
        @test !("MarketData" in deps)
        @test !("TimeSeries" in deps)
    end

    @testset "loading without the optional packages" begin
        code = """
        using PortfolioOpt
        loaded = Set(string(m) for m in values(Base.loaded_modules))
        println(any(in(loaded), ("OptimalBids", "MarketData", "TimeSeries", "PowerModels", "HTTP")))
        try
            PortfolioOpt.TestUtils.get_test_data()
        catch e
            showerror(stdout, e)
        end
        """
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $code`
        output = read(cmd, String)
        @test startswith(output, "false")
        @test occursin("using MarketData", output)
    end

    @testset "MarketData" begin
        @test Base.get_extension(PortfolioOpt, :PortfolioOptMarketDataExt) !== nothing
        prices = get_test_data()
        @test prices isa TimeSeries.TimeArray
        @test size(prices, 2) == 6
    end

    @testset "TimeSeries" begin
        @test Base.get_extension(PortfolioOpt, :PortfolioOptTimeSeriesExt) !== nothing
        returns_series = percentchange(get_test_data())
        hist = VolumeMarketHistory(returns_series)
        @test num_days(hist) == size(returns_series, 1)
        @test PortfolioOpt._timestamps(returns_series) ==
            TimeSeries.timestamp(returns_series)
    end

    @testset "OptimalBids" begin
        @test Base.get_extension(PortfolioOpt, :PortfolioOptOptimalBidsExt) !== nothing
        # the market functions of OptimalBids accept any PortfolioMarket
        market = VolumeMarket(2; budget=1.0)
        OptimalBids.change_bids!(market, [0.25, 0.75])
        @test bids(market) == [0.25, 0.75]
        OptimalBids.clear_market!(market, [0.1, -0.1])
        @test OptimalBids.calculate_profit(market).profit ≈ 0.25 * 0.1 - 0.75 * 0.1
        @test calculate_profit(market) == OptimalBids.calculate_profit(market)
        # PortfolioOpt and OptimalBids own different functions
        @test change_bids! !== OptimalBids.change_bids!
    end
end
