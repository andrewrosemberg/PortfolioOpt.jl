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

"""Basic solution """
function compute_solution_backtest(
    model::JuMP.Model, w; solver=DEFAULT_SOLVER, max_wealth=1
)
    set_optimizer(model, solver)
    optimize!(model)
    status = termination_status(model)
    status !== MOI.OPTIMAL && @warn "Did not find an optimal solution: status=$status"
    w_values = value.(w)
    w_values = reajust_volumes(w_values, max_wealth)
    return w_values
end
