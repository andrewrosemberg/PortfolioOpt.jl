using Logging
using PortfolioOpt
using JuMP

"""
    worst_case_return(decision::Array{Float64,1}, formulation::PortfolioOpt.RobustBertsimas, solver)

Returns worst case return (WCR) in Bertsimas's uncertainty set ([`RobustBertsimas`](@ref)) for a defined decision:

$(PortfolioOpt._portfolio_return_latex_RobustBertsimas_primal())

"""
function worst_case_return(decision::Array{Float64,1}, formulation::PortfolioOpt.RobustBertsimas, solver)
    r̄ = formulation.predicted_mean
    numA = formulation.number_of_assets
    Δ = formulation.uncertainty_delta
    Λ = formulation.bertsimas_budget

    model = Model(solver)
    @variable(model, μ[i=1:numA])
    @variable(model, 0.0 <= z[i=1:numA] <= 1)
    @constraint(model, [i=1:numA], μ[i] <= r̄[i] + z[i] * Δ[i])
    @constraint(model, [i=1:numA], μ[i] >= r̄[i] - z[i] * Δ[i])
    @constraint(model, sum(z) <= Λ)
    @objective(model, Min, decision'μ)

    optimize!(model)
    status = termination_status(model)
    status !== MOI.OPTIMAL && @warn "Did not find an optimal solution: status=$status"

    return objective_value(model)
end

function best_case_return(decision::Array{Float64,1}, formulation::PortfolioOpt.RobustBertsimas, solver)
    r̄ = formulation.predicted_mean
    numA = formulation.number_of_assets
    Δ = formulation.uncertainty_delta
    Λ = formulation.bertsimas_budget

    model = Model(solver)
    @variable(model, μ[i=1:numA])
    @variable(model, 0.0 <= z[i=1:numA] <= 1)
    @constraint(model, [i=1:numA], μ[i] <= r̄[i] + z[i] * Δ[i])
    @constraint(model, [i=1:numA], μ[i] >= r̄[i] - z[i] * Δ[i])
    @constraint(model, sum(z) <= Λ)
    @objective(model, Max, decision'μ)

    optimize!(model)
    status = termination_status(model)
    status !== MOI.OPTIMAL && @warn "Did not find an optimal solution: status=$status"

    return objective_value(model)
end

"""
    worst_case_return(decision::Array{Float64,1}, formulation::PortfolioOpt.RobustBenTal, solver)

Returns worst case return (WCR) in BenTal's uncertainty set ([`RobustBenTal`](@ref)) for a defined decision:

$(PortfolioOpt._portfolio_return_latex_RobustBenTal_primal())

"""
function worst_case_return(decision::Array{Float64,1}, formulation::PortfolioOpt.RobustBenTal, solver)
    r̄ = formulation.predicted_mean
    sqrt_Σ_inv = inv(sqrt(formulation.predicted_covariance))
    numA = formulation.number_of_assets
    δ = formulation.uncertainty_delta

    model = Model(solver)
    @variable(model, μ[i=1:numA])
    @constraint(model, [δ; sqrt_Σ_inv * (μ - r̄)] in JuMP.SecondOrderCone())
    @objective(model, Min, decision'μ)

    optimize!(model)
    status = termination_status(model)
    status !== MOI.OPTIMAL && @warn "Did not find an optimal solution: status=$status"

    return objective_value(model)
end

function best_case_return(decision::Array{Float64,1}, formulation::PortfolioOpt.RobustBenTal, solver)
    r̄ = formulation.predicted_mean
    sqrt_Σ_inv = inv(sqrt(formulation.predicted_covariance))
    numA = formulation.number_of_assets
    δ = formulation.uncertainty_delta

    model = Model(solver)
    @variable(model, μ[i=1:numA])
    @constraint(model, [δ; sqrt_Σ_inv * (μ - r̄)] in JuMP.SecondOrderCone())
    @objective(model, Max, decision'μ)

    optimize!(model)
    status = termination_status(model)
    status !== MOI.OPTIMAL && @warn "Did not find an optimal solution: status=$status"

    return objective_value(model)
end

################################### To be Updated #####################################
function returns_montecarlo(Σ, r̄, numS)
    d = MvNormal(r̄, Σ)
    r = rand(d, numS)
    P = pdf(d, r)
    return r', P / sum(P)
end

function compute_solution_stoc(model::JuMP.Model, w; solver=DEFAULT_SOLVER)
    set_optimizer(model, solver)
    optimize!(model)
    status = termination_status(model)
    status !== MOI.OPTIMAL && @warn "Did not find an optimal solution: status=$status"

    w_values = value.(w)
    if sum(w_values) > 1.0
        w_values = w_values / sum(w_values)
    end
    Cvar = -objective_value(model)
    r = sum(r̄'w_values)
    q1_α = value.(model[:z])
    return w_values, r, Cvar, q1_α
end

function compute_solution_stoc_2(model::JuMP.Model, w; solver=DEFAULT_SOLVER)
    set_optimizer(model, solver)
    optimize!(model)
    status = termination_status(model)
    status !== MOI.OPTIMAL && @warn "Did not find an optimal solution: status=$status"

    w_values = value.(w)
    if sum(w_values) > 1.0
        w_values = w_values / sum(w_values)
    end
    r = objective_value(model)
    q1_α = value.(model[:z])
    return w_values, r, q1_α
end
