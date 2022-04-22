function day_backtest_market!(
    strategy_logic::Function,
    market_history::MarketHistory,
    date;
    recorders=
)
    market =  market_template(market_history, date)
    strategy_logic(
        market,
        past_prices(market_history, date)
    )

    clear_market!(market, current_prices(market_history, date))

    set_market_budget(market, market_budget(market) + total_profit(market))

    return nothing
end

"""
    sequential_backtest_market(strategy_logic::Function, market_history::VolumeMarketHistory) -> Vector{Real}, Vector{Real}

Simple sequential backtest functionality for strategies that return an array of invested money per asset. Ideal for statefull strategies.

API:
```julia
wealth_strategy, returns_strategy = sequential_backtest_market(market_history; date_range=date_range) 
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
function sequential_backtest_market(
    strategy_logic::Function,
    market_history::MarketHistory;
    date_range=keys(market_history)[2:end],
)
    @assert issorted(date_range)
    numD = length(date_range)
    T = eltype(market_history)

    strategy_returns = Array{T}(undef, numD)
    wealth = Array{T}(undef, numD + 1)

    wealth[1] = market_template(market_history).budget

    for (iter, date) in enumerate(date_range)
        day_backtest_market!(strategy_logic, market_history, date)

        wealth[iter + 1] = market_budget(market)
        strategy_returns[iter] = total_profit(market) / wealth[iter]
    end
    
    wealth_dict = OrderedDict(date_range .=> wealth[1:end-1])
    date_step = date_range[end] - date_range[end-1]
    wealth_dict[date_range[end]+date_step] = wealth[end]
    
    return wealth_dict, OrderedDict(date_range .=> strategy_returns)
end
