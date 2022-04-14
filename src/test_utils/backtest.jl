"""
    backtest_po(strategy_logic::Function, returns_series::TimeArray) -> TimeArray, Array{Float64}

Simple backtest functionality for strategies that return an array of invested money per asset.

API:
```julia
wealth_strategy, returns_strategy = backtest_po(returns_series; start_date=start_date) 
    do past_returns, current_wealth, risk_free_return
    # ... Strategy definition ...
    return investment_decision
end
```

Arguments:
 - `strategy_logic::Function`: Function that represents investment strategy.
 - `returns_series::TimeArray{Float64,2,Dates.Date,Array{Float64,2}}`: Asset return series.

Optional Keywork Arguments:
 - `start_date::Date=timestamp(returns_series)[1]`: Start date of the backtest.
 - `end_date::Date=timestamp(returns_series)[end]`: End date of the backtest.
 - `initial_wealth::Real=1.0`: Initial available wealth to be invested.
 - `risk_free_returns::TimeArray=TimeArray(timestamp(returns_series), fill(0.0, size(returns_series, 1)))`: Risk-free returns of money not invested per period.
"""
function backtest_po(
    strategy_logic::Function,
    returns_series::TimeArray{Float64,2,Dates.Date,Array{Float64,2}};
    start_date::Date=timestamp(returns_series)[1],
    end_date::Date=timestamp(returns_series)[end],
    initial_wealth::Real=1.0,
    risk_free_returns::TimeArray=TimeArray(
        timestamp(returns_series), fill(0.0, size(returns_series, 1))
    ),
)
    start_date_previous = timestamp(to(returns_series, start_date))[end - 1]
    time_stamps = timestamp(to(from(returns_series, start_date_previous), end_date))
    T = length(time_stamps) - 1
    numD, numA = size(returns_series)
    # save returns
    strategy_returns = Array{Float64}(undef, T)
    # wealth at the beginning of the period
    wealth = Array{Float64}(undef, T + 1)
    wealth[1] = initial_wealth
    for iter in 1:T
        t = time_stamps[iter + 1]
        portfolio_volumes = readjust_volumes!(
            strategy_logic(
                to(returns_series, time_stamps[iter]),
                wealth[iter],
                values(risk_free_returns[t])[1],
            );
            current_wealth = wealth[iter],
        )
        step_return = sum(portfolio_volumes * values(returns_series[t]))
        wealth[iter + 1] = wealth[iter] + step_return
        strategy_returns[iter] = step_return / wealth[iter]
    end
    # wealth at the end of each period
    return TimeArray((datetime=time_stamps, wealth=wealth); timestamp=:datetime),
    strategy_returns
end
