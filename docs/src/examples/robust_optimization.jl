# # Robust Portfolio Optimization

# ## Robust Return as Risk Constraint

using HiGHS
using COSMO
using Distributions
using PortfolioOpt
using PortfolioOpt.TestUtils

# Read Prices 
prices = get_test_data();
numD, numA = size(prices) # A: Assets    D: Days

# Calculating returns
returns_series = percentchange(prices);

# Backtest parameters
DEFAULT_SOLVER = optimizer_with_attributes(
    HiGHS.Optimizer, "presolve" => "on", "time_limit" => 60.0, "log_to_console" => false
)

COSMO_SOLVER = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

date_range = timestamp(returns_series)[100:end];

# ### Backtest Soyster's robust return
backtest_results = Dict()

backtest_results["Soyster"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    numD, numA = size(past_returns)
    returns = values(past_returns)
    k_back = 30
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    d = MvNormal(r̄, Σ)
    R = -0.015 / market_budget(market)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(ExpectedReturn(BudgetSet(d; Δ=maximum(abs.(returns); dims=1)'[:] .- r̄, Γ=numA * 1.0), WorstCase), GreaterThan(R)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

# ### Backtest Bertsimas's robust return

backtest_results["Bertsimas"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    numD, numA = size(past_returns)
    returns = values(past_returns)
    k_back = 30
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    d = MvNormal(r̄, Σ)
    R = -0.015 / market_budget(market)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(ExpectedReturn(BudgetSet(d; Δ=maximum(abs.(returns); dims=1)'[:] .- r̄, Γ=numA * 0.3), WorstCase), GreaterThan(R)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

# ### Backtest BenTal's robust return

backtest_results["Ben-Tal"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    numD, numA = size(past_returns)
    returns = values(past_returns)
    k_back = 30
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    d = MvNormal(r̄, Σ)
    R = -0.015 / market_budget(market)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(ExpectedReturn(EllipticalSet(d, 2.0), WorstCase), GreaterThan(R)),
    )
    
    pointers = change_bids!(market, formulation, COSMO_SOLVER)
    return pointers
end

# ### Backtest Betina's robust return

backtest_results["Betina"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    numD, numA = size(past_returns)
    returns = values(past_returns)
    
    k_back = 30
    j_robust = 25
    R = -0.015 / market_budget(market)

    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    d = MvNormal(r̄, Σ)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(ConditionalExpectedReturn(DeterministicSamples(returns'[:, (end - j_robust):end]); α=0.0), GreaterThan(R)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

# ## Plot

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
