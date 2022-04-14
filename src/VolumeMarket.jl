import OptimalBids: Market, change_bids!, clear_market!, calculate_profit

mutable struct VolumeMarket{T<:Real} <: OptimalBids.Market
    budget::T
    volume_fee::T
    allow_short_selling::Bool
    risk_free_rate::T
    volume_bids::Union{Vector{T},Nothing}
    clearing_prices::Union{Vector{T},Nothing}
end

function VolumeMarket(budget::T, volume_fee::T, allow_short_selling::Bool, risk_free_rate::T) where {T<:Real}
    VolumeMarket{T}(budget, volume_fee, allow_short_selling, risk_free_rate, nothing, nothing)
end

function VolumeMarket(; budget::T=1.0, volume_fee::T=0.0, allow_short_selling::Bool=false, risk_free_rate::T=0.0) where {T<:Real}
    VolumeMarket(budget, volume_fee, allow_short_selling, risk_free_rate)
end

function change_bids!(market::VolumeMarket{T}, new_bids::Vector{T}) where {T<:Real}
    total_volume = norm(new_bids, 1)
    if  total_volume > market.budget
        market.volume_bids = new_bids / total_volume
    else
        market.volume_bids = new_bids
    end

    return nothing
end

function clear_market!(market::VolumeMarket)
    @warn "Prices have not yet been set for this market. 
    This function can only be dispatched passing cleared prices (`clearing_prices`)"
    throw(MethodError(clear_market!, (market)))
end

function clear_market!(market::VolumeMarket{T}, clearing_prices::Vector{T}) where {T<:Real}
    market.clearing_prices = clearing_prices
end

function calculate_profit(market::VolumeMarket)
    return (;
        cleared_volumes=market.volume_bids,
        clearing_prices=market.clearing_prices,
        profit=dot(cleared_volumes, clearing_prices),
        risk_free_profit=market.risk_free_rate * (market.budget - norm(cleared_volumes, 1))
    )
end

"""
    market_model(market::VolumeMarket, numA::Integer, optimizer_factory::Function) -> JuMP.Model, Vector{VariableRef}

Creates a JuMP model with appropriate PO variable and constraints:
    - Investment vector of variables `w` (portfolio weights if `budget = 1`).
    - Invested monney should be lower budget.

Returns the model and the reference to the vector of decision variables (length numA).

Aditional arguments:
 - `numA::Integer`: number of assets in portfolio.
 - `optimizer_factory`: callable with zero arguments and return an empty `MathOptInterface.AbstractOptimizer``.
"""
function market_model(market::VolumeMarket, numA::Int, optimizer_factory::Function; 
    sense::MOI.OptimizationSense=MOI.MAX_SENSE, model::JuMP.Model=Model(optimizer_factory)
)
    w = @variable(model, [1:numA])
    sum_invested = @variable(model)
    if !market.allow_short_selling
        set_lower_bound.(w, 0.0)
        @constraint(model, sum_invested == sum(w))
    else
        @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    end
    @constraint(model, sum_invested <= current_wealth)
    @objective(model, sense, sum_invested * market.risk_free_rate)
    return model, w
end

"""
    change_bids!(market::VolumeMarket, model::JuMP.Model, decision_variables)

Solves optimization model and set new bids to optimal values of the `decision_variables`.
"""
function change_bids!(market::VolumeMarket, model::JuMP.Model, decision_variables)
    optimize!(model)
    status = termination_status(model)
    status !== MOI.OPTIMAL && @warn "Did not find an optimal solution: status=$status"
    new_bids = value.(decision_variables)
    change_bids!(market, new_bids)

    return nothing
end

JuMP.owner_model(decision_variables::Vector{VariableRef}) = owner_model(first(decision_variables))
JuMP.owner_model(decision_variables::AffExpr) = owner_model(first(keys(decision_variables.terms)))
JuMP.owner_model(decision_variables::Vector{AffExpr}) = owner_model(first(decision_variables))
