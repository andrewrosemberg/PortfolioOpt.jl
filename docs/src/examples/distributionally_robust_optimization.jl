# # Distributionally Robust Portfolio Optimization

# ## distributionally Robust Return as Objective Term

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

# ### Backtest Delage's robust return
backtest_results = Dict()

backtest_results["Delage"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    k_back = 30
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    # Parameters
    d = MvNormal(r̄, Σ)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedUtility(MomentUncertainty(d, 0.05, 1.3), 
            PieceWiseUtility(
                [1.0], [0.0]
            ),
            WorstCase
        ))
    )
    
    pointers = change_bids!(market, formulation, COSMO_SOLVER)
    return pointers
end

# ### Backtest NingNing Du's Wasserstein robust return (under light tail assumption)

backtest_results["Wasserstein_light_tail"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    
    # Parameters
    j_robust = numD
    δ = 0.1
    ϵ = log(1/δ)/j_robust
    
    d = DeterministicSamples(returns'[:,:])
    s = DuWassersteinBall(d; ϵ=ϵ)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(s))
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
