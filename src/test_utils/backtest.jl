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
    market_history::MarketHistory;
    date_range=second(eachindex(market_history)):step(eachindex(market_history)):last(eachindex(market_history)),
)
    numD = length(date_range)
    
    # save returns
    strategy_returns = Array{Float64}(undef, numD)
    # wealth at the beginning of the period
    wealth = Array{Float64}(undef, numD + 1)
    wealth[1] = initial_wealth

    for (iter, date) in enumerate(date_range)
        market =  market_template(market_history, date)
        strategy_logic(
            market,
            past_prices(market_history, date)
        )

        clear_market!(market, current_prices(market_history, date))

        step_return = total_profit(market)
        wealth[iter + 1] = wealth[iter] + step_return
        strategy_returns[iter] = step_return / wealth[iter]
    end
    # wealth at the end of each period
    return wealth, strategy_returns
end
