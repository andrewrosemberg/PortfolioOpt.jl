"""
Get test data from MarketData.
"""
function get_test_data(;
    start_date=Date(2009, 9, 1), end_date=start_date + Year(1) + Month(3)
)
    df_AAPL = rename(to(from(AAPL[:Close], start_date), end_date), :AAPL)
    df_BA = rename(to(from(BA[:Close], start_date), end_date), :BA)
    df_DELL = rename(to(from(DELL[:Close], start_date), end_date), :DELL)
    df_CAT = rename(to(from(CAT[:Close], start_date), end_date), :CAT)
    df_EBAY = rename(to(from(EBAY[:Close], start_date), end_date), :EBAY)
    df_F = rename(to(from(F[:Close], start_date), end_date), :F)
    df = merge(df_AAPL, df_BA, df_DELL, df_CAT, df_EBAY, df_F)
    return df
end

"""
Mean and Variance of returns
"""
function mean_variance(returns)
    r̄ = mean(returns; dims=1)'
    Σ = cov(returns)
    return Σ, r̄
end

"""
simulate returns normal
"""
function returns_montecarlo(Σ, r̄, numS)
    d = MvNormal(r̄, Σ)
    r = rand(d, numS)
    P = pdf(d, r)
    return r', P / sum(P)
end

"""
Basic solution
"""
function compute_solution_backtest(
    model::JuMP.Model, w, solver; max_wealth=1
)
    set_optimizer(model, solver)
    optimize!(model)
    status = termination_status(model)
    status !== MOI.OPTIMAL && @warn "Did not find an optimal solution: status=$status"
    w_values = value.(w)
    w_values = reajust_volumes(w_values, max_wealth)
    return w_values
end
