# # Stochastic Portfolio Optimization

# ## Empirical Forecast

using Clarabel
using Distributions
using PortfolioOpt
using PortfolioOpt.TestUtils

# Read Prices 
prices = get_test_data();
numD, numA = size(prices) # A: Assets    D: Days

# Calculating returns 
returns_series = percentchange(prices);

# Backtest Parameters 
DEFAULT_SOLVER = optimizer_with_attributes(
    Clarabel.Optimizer, "verbose" => false, "max_iter" => 900000
)

date_range = timestamp(returns_series)[100:end];

# ### Backtest Markowitz

backtest_results = Dict()
backtest_results["EP_markowitz_limit_var"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    max_std = 0.003 / market_budget(market)
    k_back = 60

    numD, numA = size(past_returns)
    returns = values(past_returns)
    
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    d = MvNormal(r̄, Σ)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(SqrtVariance(d), LessThan(max_std)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

# ### Backtest Sampled CVAR

backtest_results["EP_limit_cvar"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    numD, numA = size(past_returns)
    returns = values(past_returns)
    
    R = -0.001 / market_budget(market)
    d = DeterministicSamples(returns'[:,:])

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(ConditionalExpectedReturn(d), GreaterThan(R)),
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