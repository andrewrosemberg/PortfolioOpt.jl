mutable struct VolumeMarket{T<:Real,N} <: OptimalBids.Market
    budget::T
    volume_fee::T
    allow_short_selling::Bool
    risk_free_rate::T
    volume_bids::Union{Vector{T},Nothing}
    clearing_prices::Union{Vector{T},Nothing}
end

function VolumeMarket(N::Int, budget::T, volume_fee::T, allow_short_selling::Bool, risk_free_rate::T) where {T<:Real}
    VolumeMarket{T,N}(budget, volume_fee, allow_short_selling, risk_free_rate, nothing, nothing)
end

function VolumeMarket(N::Int; budget::T=1.0, volume_fee::T=0.0, allow_short_selling::Bool=false, risk_free_rate::T=0.0) where {T<:Real}
    VolumeMarket(budget, volume_fee, allow_short_selling, risk_free_rate, N)
end

function change_bids!(market::VolumeMarket{T, N}, new_bids::Vector{T}) where {T<:Real,N}
    @assert length(new_bids) == N
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

function clear_market!(market::VolumeMarket{T, N}, clearing_prices::Vector{T}) where {T<:Real,N}
    @assert length(clearing_prices) == N
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

function total_profit(market::VolumeMarket)
    profits = calculate_profit(market)
    return sum(profits.profit) + risk_free_profit
end

function length(::VolumeMarket{T, N}) where {T, N}
    return N
end

"""
    market_model(market::VolumeMarket{T,N}, optimizer_factory::Function) -> JuMP.Model, Vector{VariableRef}

Creates a JuMP model with appropriate PO variable and constraints:
    - Investment vector of variables `w` (portfolio weights if `budget = 1`).
    - Invested monney should be lower budget.

Returns the model and the reference to the vector of decision variables (length N).

Aditional arguments:
 - `optimizer_factory`: callable with zero arguments and return an empty `MathOptInterface.AbstractOptimizer``.
"""
function market_model(market::VolumeMarket{T,N}, optimizer_factory::Function; 
    sense::MOI.OptimizationSense=MAX_SENSE, model::JuMP.Model=Model(optimizer_factory)
) where {T,N}
    w = @variable(model, [1:N])
    sum_invested = @variable(model)
    if !market.allow_short_selling
        set_lower_bound.(w, 0.0)
        @constraint(model, sum_invested == sum(w))
    else
        @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    end
    @constraint(model, sum_invested <= current_wealth)
    if sense === MAX_SENSE
        @objective(model, sense, sum_invested * market.risk_free_rate)
    else
        @objective(model, sense, 0.0)
    end
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

abstract type MarketHistory end

eachindex(hist::MarketHistory) = eachindex(hist.history_clearing_prices)
past_prices(hist::MarketHistory, t) = hist.history_clearing_prices[first(eachindex(market_history)):step(eachindex(market_history)):t]
current_prices(hist::MarketHistory, t) = hist.history_clearing_prices[t]

struct VolumeMarketHistory <: MarketHistory
    market::VolumeMarket
    history_clearing_prices::Any
    history_risk_free_rates::Any
end

function VolumeMarketHistory(market::VolumeMarket, history_clearing_prices)
    rf = Dict(eachindex(market_history) .=> market.risk_free_rate)
    return VolumeMarketHistory(market, history_clearing_prices, rf)
end

risk_free_rate(hist::VolumeMarket, t) = hist.history_risk_free_rates[t]

market_template(hist::VolumeMarket) = hist.market
function market_template(hist, t)
    market.risk_free_rate = risk_free_rate(hist, t)
    return hist.market
end
