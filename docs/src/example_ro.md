# Example Markowitz

## Empirical Forecast

Example of backtest with Mean-Variance strategy with a simple empirical forecaster.

```@example Backtest
using COSMO
using Distributions
using Plots
using PortfolioOpt
using PortfolioOpt.TestUtils

# Read Prices 
prices = get_test_data();
numD, numA = size(prices) # A: Assets    D: Days

# Calculating returns 
returns_series = percentchange(prices);

# Backtest Parameters 
DEFAULT_SOLVER = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

date_range = timestamp(returns_series)[100:end];

# Backtest
backtest_results = Dict()
backtest_results["EP_markowitz_limit_var"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Parameters
    max_std = 0.003 / market_budget(market)
    k_back = 60

    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    
    # Forecast
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    d = MvNormal(r̄, Σ)

    # PO Formulation
    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(SqrtVariance(d), LessThan(max_std)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

```