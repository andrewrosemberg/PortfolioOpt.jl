# Examples

## Backtest Example

Simple example of backtest with an available strategy.

```@example MeanVariance
using COSMO
using Plots
using PortfolioOpt
using PortfolioOpt.TestUtils: 
    backtest_market, get_test_data, mean_variance, 
    percentchange, timestamp, rename!

prices = get_test_data()
numD, numA = size(prices) # A: Assets    D: Days
returns_series = percentchange(prices)

solver = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

start_date = timestamp(returns_series)[100]

wealth_strategy, returns_strategy =
    backtest_market(returns_series; start_date=start_date
    ) do past_returns, current_wealth, risk_free_return

        # Prep data provided by the backtest pipeline
        numD, numA = size(past_returns)
        returns = values(past_returns)
        # calculate mean and variance for the past 60 days
        Σ, r̄ = mean_variance(returns[(end - 60):end, :])

        # Parameters
        # maximum acceptable normalized variance for our portfolio
        max_risk = 0.8
        formulation = MeanVariance(;
            predicted_mean = r̄,
            predicted_covariance = Σ,
        )

        # Build PO model
        model = po_max_return_limit_variance(formulation, max_risk; rf = risk_free_return)

        # Optimize model and retrieve solution
        x = compute_solution(model, solver)

        # return invested portfolio in used currency
        return x * current_wealth
end

plt = plot(
    wealth_strategy;
    title="Culmulative Wealth",
    xlabel="Time",
    ylabel="Wealth",
    legend=:outertopright,
)
```
