"""
    backtest_market(strategy_logic::Function, market_history::VolumeMarketHistory) -> Vector{Real}, Vector{Real}

Simple backtest functionality for strategies that return an array of invested money per asset.

API:
```julia
wealth_strategy, returns_strategy = backtest_market(market_history; date_range=date_range) 
    do market, past_returns
    # ... Strategy definition ...
    return investment_decision
end
```

Arguments:
 - `strategy_logic::Function`: Function that represents the investment strategy.
 - `market_history::MarketHistory`: Market history.

Optional Keywork Arguments:
 - `date_range`: Dates to be simulated with corresponding entries in the market history.
"""
function backtest_market(
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
    return Dict(date_range .=> wealth), Dict(date_range .=> strategy_returns)
end
