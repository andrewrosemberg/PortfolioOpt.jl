using COSMO
using Distributions
using Plots
using PortfolioOpt
using PortfolioOpt.TestUtils
plotly()

##################### Read Prices #####################
prices = get_test_data();
numD, numA = size(prices) # A: Assets    D: Days

##################### Calculating returns #####################
returns_series = percentchange(prices);

# market_history = VolumeMarketHistory(returns_series)

##################### backtest Parameters #####################
DEFAULT_SOLVER = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

date_range = timestamp(returns_series)[100:end];

## Strategies parameters
# Look back days
k_back = 60
# results
backtest_results = Dict()

##################### backtest stochastic strategies #####################

backtest_results["markowitz_limit_R"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    # Parameters
    d = MvNormal(r̄, Σ)
    R = 0.0015 / market_budget(market)

    formulation = PortfolioFormulation(MIN_SENSE,
        ObjectiveTerm(SqrtVariance(d)),
        RiskConstraint(ExpectedReturn(d), GreaterThan(R)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

backtest_results["markowitz_limit_var"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    # Parameters
    d = MvNormal(r̄, Σ)
    V = 0.003 / market_budget(market)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(SqrtVariance(d), LessThan(V)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

##################### backtest with RO strategies #####################

backtest_results["normal_limit_soyster"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    # Parameters
    d = MvNormal(r̄, Σ)
    R = -0.05 / market_budget(market)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(ExpectedReturn(BudgetSet(d, maximum(abs.(returns); dims=1)'[:] .- r̄, numA * 1.0), WorstCase), GreaterThan(R)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

backtest_results["normal_limit_bertsimas_30"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    # Parameters
    d = MvNormal(r̄, Σ)
    R = -0.05 / market_budget(market)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(ExpectedReturn(BudgetSet(d, maximum(abs.(returns); dims=1)'[:] .- r̄, numA * 0.3), WorstCase), GreaterThan(R)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

backtest_results["normal_limit_bertsimas_60"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    # Parameters
    d = MvNormal(r̄, Σ)
    R = -0.05 / market_budget(market)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(ExpectedReturn(BudgetSet(d, maximum(abs.(returns); dims=1)'[:] .- r̄, numA * 0.6), WorstCase), GreaterThan(R)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

backtest_results["normal_limit_bental"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    # Parameters
    d = MvNormal(r̄, Σ)
    R = -0.05 / market_budget(market)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(ExpectedReturn(EllipticalSet(d, 0.1), WorstCase), GreaterThan(R)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

##################### backtest with Data-Driven RO strategies #####################

backtest_results["mixed_signals_limit_betina"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    
    # Parameters
    j_robust = 25
    R = -0.015 / market_budget(market)
    num_train = 20
    kst = 14
    klt = 30
    kmom = 20
    Q = 2

    # Predict return 
    r̄ = zeros(numA)
    for i in 1:numA
        r̄[i] = mixed_signals_predict_return(
            returns[:, i], num_train, kst, klt, kmom, Q
        )
    end

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(Dirac(r̄))),
        RiskConstraint(ConditionalExpectedReturn{WorstCase}(Inf, DeterministicSamples(returns'[:,:]), j_robust), GreaterThan(R)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

##################### backtest with Moment DRO strategies #####################

backtest_results["delage_utility"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    # Parameters
    d = MvNormal(r̄, Σ)
    R = -0.015 / market_budget(market)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedUtility(MomentUncertainty(d, 0.05, 1.3), 
            PieceWiseUtility(
                [1.0], [0.0]
            ),
            WorstCase
        ))
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

##################### extra strategies #####################

backtest_results["max_sharpe"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    change_bids!(market, market_budget(market) * max_sharpe(Σ, r̄, risk_free_rate(market)))

    return nothing
end

backtest_results["equal_weights"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    
    change_bids!(market, market_budget(market) * equal_weights(numA))

    return nothing
end

##################### plot results  #####################
using Plots
using Plots.PlotMeasures

plt = plot(;title="Culmulative Wealth",
    xlabel="Time",
    ylabel="Wealth",
    legend=:outertopright,
    left_margin=10mm
);
for (strategy_name, recorders) in backtest_results
    plot!(plt, 
        axes(get_records(recorders[:wealth]), 1), get_records(recorders[:wealth]).data;
        label=strategy_name,
    )
end
plt

plt2 = scatter(;
    title="Mean Vs Std",
    xlabel="σ",
    ylabel="E[r]",
    legend=:outertopright,
    left_margin=10mm
);
for (strategy_name, recorders) in backtest_results
    scatter!(plt2, 
    [std(get_records(recorders[:returns]).data)], [mean(get_records(recorders[:returns]).data)];
        label=strategy_name,
    )
end
plt2