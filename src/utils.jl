"""
    create_sum_invested_variable(model::JuMP.Model, w)

Creates decision variable representing the invested money in variable-return assets: ``||w||``.

Arguments:
 - `model::JuMP.Model`: JuMP upper level portfolio optimization model.
 - `w`: portfolio optimization investment variable ("weights").
"""
function create_sum_invested_variable(model::JuMP.Model, w)
    if !haskey(object_dictionary(model), :sum_invested)
        sum_invested = @variable(model, sum_invested)
        @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
        return sum_invested
    else
        return model[:sum_invested]
    end
end

"""
    base_model(numA) -> model

Creates a JuMP model with generic PO variable and constraints:
    - Investment vector of variables `w` (portfolio weights if `current_wealth = 1`).
    - Invested monney should be lower current wealth.

Returns the model (which has a reference to the decision variable named `w`).

Arguments:
 - `numA::Integer`: number of assets in portfolio.

Optional Keywork Arguments:
 - `current_wealth::Real = 1.0`: Current available wealth to be invested.
 - `allow_borrow::Bool=false`: Short variables work as borrowing money.
"""
function base_model(numA::Integer; current_wealth::Real=1.0, allow_borrow::Bool=false)
    model = Model()
    w = @variable(model, w[i=1:numA])
    @variable(model, sum_invested)
    if allow_borrow
        @constraint(model, sum_invested == sum(w))
    else
        @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    end
    @constraint(model, sum_invested <= current_wealth)
    return model
end

"""
    readjust_volumes(decision_values::Array{Float64,1}; current_wealth::Real = 1.0)

Ajust volumes to be feasible under current wealth.

Arguments:
 - `decision_values::Array{Float64,1}`: Decided values to be invested.

Optional Keywork Arguments:
 - `current_wealth::Real = 1.0`: Current available wealth to be invested.
"""
function readjust_volumes!(decision_values::Array{Float64,1}; current_wealth::Real = 1.0)
    decision_values_ajusted = deepcopy(decision_values)
    if norm(decision_values, 1) > current_wealth
        decision_values_ajusted = decision_values / norm(decision_values, 1)
    end
    return decision_values_ajusted
end

"""
    compute_solution(model, solver) -> Array{Float64,1}

Solves optimization model using provides solver and returns solutions ajusted to current wealth.

Arguments:
 - `model::JuMP.Model`: JuMP portfolio optimization model.
 - `solver`: Appropriate solver for the provided model.

Optional Keywork Arguments:
 - `current_wealth::Real = 1.0`: Current available wealth to be invested.
 - `decision_variable = model[:w]`: Portfolio optimization investment variable reference.
"""
function compute_solution(
    model::JuMP.Model, solver; 
    current_wealth::Real = 1.0,
    decision_variable = model[:w]
)
    set_optimizer(model, solver)
    optimize!(model)
    status = termination_status(model)
    status !== MOI.OPTIMAL && @warn "Did not find an optimal solution: status=$status"
    decision_values = value.(decision_variable)
    decision_values = readjust_volumes!(decision_values; current_wealth = current_wealth)
    return decision_values
end
