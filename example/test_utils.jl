"""
Mean and Variance of returns
"""
function mean_variance(returns)
    r̄ = mean(returns,1)'
    Σ = cov(returns)
    return Σ,r̄
end

function returns_montecarlo(Σ,r̄, numS)
    d = MvNormal(r̄, Σ)
    r = rand(d, numS)
    P = pdf(d,r)
    return r',P/sum(P)
end

function compute_solution(model::JuMP.Model, w)
    optimize!(model)
    w_values = value(w)
    r = sum(r̄'w)
    return w, objective_value(model), r
end

function compute_solution_dual(model::JuMP.Model, w)
    optimize!(model)
    w_values = value(w)
    return w, objective_value(model), value(model[:E])
end

function compute_solution_stoc(model::JuMP.Model, w)
    optimize!(model)
    w_values = value(w)
    Cvar = -objective_value(model)
    r = sum(r̄'w)
    q1_α = value(z)
    return w, r, Cvar, q1_α
end

function compute_solution_stoc_2(model::JuMP.Model, w)
    optimize!(model)
    w_values = value(w)
    r = objective_value(model)
    q1_α = value(z)
    return w, r, q1_α
end