# Example Betina

## Empirical Forecast

Example of backtest with Mean-Variance strategy with a simple empirical forecaster.

```@example Backtest
using COSMO
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
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

date_range = timestamp(returns_series)[100:end];

# Backtest
backtest_results = Dict()

backtest_results["EP_limit_betina"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    
    # Parameters
    k_back = 30
    j_robust = 25
    R = -0.015 / market_budget(market)

    # Forecast
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    d = MvNormal(r̄, Σ)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(ConditionalExpectedReturn{WorstCase}(Inf, DeterministicSamples(returns'[:,:]), j_robust), GreaterThan(R)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

```

## Mixed Signals Forecast

```@example Backtest

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

    # Forecast
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

```


## Xgboost Forecast

### Train trees

```@example Backtest
using XGBoost

# Forecast
k_back = 30
dates_for_training = timestamp(returns_series)[1:99]
prep_x(returns, k_back) = reshape(returns[end-k_back+1:end, :], 1, :)

bst_forecasters = Array{Any}(undef, numA)
for i in 1:numA
    x_train = Matrix(vcat([prep_x(values(returns_series[dates_for_training[1:t-1]]), k_back) for t = k_back+1:length(dates_for_training)]...))
    y_train = values(returns_series[k_back+1:length(dates_for_training)])[:,i]

    num_round = 2
    bst_forecasters[i] = xgboost(x_train, num_round, label = y_train, eta = 1, max_depth = 2)
end

pred = predict(bst_forecasters[2], prep_x(values(returns_series[timestamp(returns_series)[1:100]]), k_back))

```

### backtest

```@example Backtest

backtest_results["xgboost_limit_betina"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    
    # Parameters
    j_robust = 25
    R = -0.015 / market_budget(market)
    k_back = 30

    # Forecast
    r̄ = zeros(numA)
    for i in 1:numA
        r̄[i] = predict(bst_forecasters[i], prep_x(values(past_returns), k_back))[1]
    end

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(Dirac(r̄))),
        RiskConstraint(ConditionalExpectedReturn{WorstCase}(Inf, DeterministicSamples(returns'[:,:]), j_robust), GreaterThan(R)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

```

## Plot

```@example Backtest
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
```