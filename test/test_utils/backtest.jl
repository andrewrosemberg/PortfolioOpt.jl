using JuMP
using MarketData
using LinearAlgebra

"""Ajust volumes to be feasible under current wealth"""
function reajust_volumes(w_values, max_wealth)
    w_values_ajusted = deepcopy(w_values)
    if norm(w_values, 1) > max_wealth
        w_values_ajusted = w_values / norm(w_values, 1)
    end
    return w_values_ajusted
end

"""Simple backtest logic"""
function backtest_po(
    strategy_logic::Function, 
    returns_series; 
    start_date = timestamp(returns_series)[1], 
    end_date = timestamp(returns_series)[end], 
    initial_wealth = 1.0, 
    risk_free_returns = TimeArray(timestamp(returns_series), fill(0.0, size(returns_series,1)))
)
    start_date_previous = timestamp(to(returns_series, start_date))[end-1]
    time_stamps = timestamp(to(from(returns_series, start_date_previous), end_date))
    T = length(time_stamps)-1
    numD,numA = size(returns_series)
    # save returns
    strategy_returns = Array{Float64}(undef, T)
    # wealth at the beginning of the period
    wealth = Array{Float64}(undef, T+1)
    wealth[1] = initial_wealth
    for iter = 1:T
        t = time_stamps[iter+1]
        portfolio_volumes = reajust_volumes(
            strategy_logic(to(returns_series, time_stamps[iter]), wealth[iter], values(risk_free_returns[t])[1]),
            wealth[iter]
        )
        step_return = sum(portfolio_volumes*values(returns_series[t]))
        wealth[iter+1] = wealth[iter] + step_return
        strategy_returns[iter] = step_return/wealth[iter]
    end
    # wealth at the end of each period
    return TimeArray((datetime=time_stamps, wealth=wealth), timestamp=:datetime), strategy_returns
end