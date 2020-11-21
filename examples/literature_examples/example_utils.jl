using COSMO
using Logging

DEFAULT_SOLVER = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

## Prep data
function compute_solution(model::JuMP.Model, w; solver=DEFAULT_SOLVER)
    set_optimizer(model, solver)
    optimize!(model)
    status = termination_status(model)
    status !== MOI.OPTIMAL && @warn "Did not find an optimal solution: status=$status"

    w_values = value.(w)
    if sum(w_values) > 1.0
        w_values = w_values / sum(w_values)
    end
    r = sum(r̄'w_values)
    return w_values, objective_value(model), r
end

function compute_solution_dual(model::JuMP.Model, w; solver=DEFAULT_SOLVER)
    set_optimizer(model, solver)
    optimize!(model)
    status = termination_status(model)
    status !== MOI.OPTIMAL && @warn "Did not find an optimal solution: status=$status"

    w_values = value.(w)
    if sum(w_values) > 1.0
        w_values = w_values / sum(w_values)
    end
    return w_values, objective_value(model), value(model[:E])
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
