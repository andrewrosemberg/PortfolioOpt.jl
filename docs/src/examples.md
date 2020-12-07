# Examples

## Backtest Example

```julia
using COSMO
using PortfolioOpt
using PortfolioOpt.TestUtils: 
    backtest_po, compute_solution_backtest, get_test_data, mean_variance, base_model, 
    percentchange, timestamp, rename!

prices = get_test_data()
numD, numA = size(prices) # A: Assets    D: Days
returns_series = percentchange(prices)

solver = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

start_date = timestamp(returns_series)[100]

wealth_strategy, returns_strategy =
    backtest_po(returns_series; start_date=start_date) 
        do past_returns, current_wealth, risk_free_return

        # Prep data provided by the backtest pipeline
        numD, numA = size(past_returns)
        returns = values(past_returns)
        # calculate mean and variance for the past 60 days
        Σ, r̄ = mean_variance(returns[(end - 60):end, :])

        # Parameters
        # maximum acceptable normalized variance for our portfolio
        max_risk = 0.8
        formulation = MeanVariance(;
            predicted_mean = r̄_s,
            predicted_covariance = Σ,
        )

        # Build model 
        # creates jump model with portfolio weights variable w
        model, w = base_model(numA; allow_borrow=false)
        # modifies the problem to add fromulation variable and constraints
        po_max_return_limit_variance!(model, w, formulation, max_risk; rf = rf)

        # Optimize model and retrieve solution (x = optimal w value)
        x = compute_solution_backtest(model, w, solver)

        # return invested portfolio in used currency
        return x * current_wealth
end

```

## Plot Results
```
using Plots

plot(
    wealth_strategy;
    title="Culmulative Wealth",
    xlabel="Time",
    ylabel="Wealth",
    legend=:outertopright,
)
```
![](https://github.com/andrewrosemberg/PortfolioOpt/blob/master/docs/src/assets/cumwealth.png?raw=true)