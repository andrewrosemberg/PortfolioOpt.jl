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
    VolumeMarket(N, budget, volume_fee, allow_short_selling, risk_free_rate)
end

risk_free_rate(market::VolumeMarket) = market.risk_free_rate

market_budget(market::VolumeMarket) = market.budget
function set_market_budget(market::VolumeMarket{T,N}, val::T) where {T,N}
    market.budget = val
end
market_volume_fee(market::VolumeMarket) = market.volume_fee

function eltype(::VolumeMarket{T,N}) where {T,N}
    return T
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
        profit=dot(market.volume_bids, market.clearing_prices),
        risk_free_profit=market.risk_free_rate * (market.budget - norm(market.volume_bids, 1))
    )
end

function total_profit(market::VolumeMarket)
    profits = calculate_profit(market)
    return sum(profits.profit) + profits.risk_free_profit
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
function market_model(market::VolumeMarket{T,N}, optimizer_factory::Any; 
    sense::MOI.OptimizationSense, model::JuMP.Model=Model(optimizer_factory)
) where {T,N}
    w = @variable(model, [1:N])
    sum_invested = @variable(model)
    if !market.allow_short_selling
        set_lower_bound.(w, 0.0)
        @constraint(model, sum_invested == sum(w))
    else
        @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    end
    @constraint(model, sum_invested <= market_budget(market))
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
    if status !== MOI.OPTIMAL
        @warn "Did not find an optimal solution: status=$status"
        change_bids!(market, zeros(length(decision_variables)))
    else
        new_bids = value.(decision_variables)
        change_bids!(market, new_bids)
    end

    return nothing
end

JuMP.owner_model(decision_variables::Vector{VariableRef}) = owner_model(first(decision_variables))
JuMP.owner_model(decision_variables::AffExpr) = owner_model(first(keys(decision_variables.terms)))
JuMP.owner_model(decision_variables::Vector{AffExpr}) = owner_model(first(decision_variables))

abstract type MarketHistory{T<:Real,N} end

function eltype(::MarketHistory{T,N}) where {T,N}
    return T
end

struct VolumeMarketHistory{T<:Real,N} <: MarketHistory{T,N}
    market::VolumeMarket{T,N}
    history_clearing_prices::Any
    history_risk_free_rates::Any
    timestamp::Any

    function VolumeMarketHistory{T,N}(market::VolumeMarket{T,N}, history_clearing_prices, history_risk_free_rates, timestamp) where {T,N}
        @assert eltype(values(history_clearing_prices)) === T
        @assert eltype(values(history_risk_free_rates)) === T
        @assert issetequal(keys(history_risk_free_rates), timestamp)
        @assert issetequal(keys(history_clearing_prices), timestamp)

        return new(market, history_clearing_prices, history_risk_free_rates, timestamp)
    end
end

function VolumeMarketHistory(market::VolumeMarket{T,N}, history_clearing_prices) where {T<:Real,N}
    rf = Dict(keys(history_clearing_prices) .=> market.risk_free_rate)
    return VolumeMarketHistory{T,N}(market, history_clearing_prices, rf, sort(unique(keys(history_clearing_prices))))
end

function VolumeMarketHistory(history_clearing_prices; kwards...)
    return VolumeMarketHistory(VolumeMarket(length(values(first(history_clearing_prices))); kwards...), history_clearing_prices)
end

number_assets(::VolumeMarketHistory{T,N}) where {T,N} = N 

timestamp(hist::VolumeMarketHistory) = hist.timestamp
keys(hist::VolumeMarketHistory) = timestamp(hist)
function past_prices(hist::VolumeMarketHistory, t)
    timestamps = timestamp(hist)
    idx = findfirst(x -> x == t, timestamps)
    hist.history_clearing_prices[timestamps[1:idx-1]]
end
current_prices(hist::VolumeMarketHistory, t) = values(hist.history_clearing_prices[t])[1,:]
risk_free_rate(hist::VolumeMarketHistory, t) = first(values(hist.history_risk_free_rates[t]))
function size(hist::VolumeMarketHistory{T,N}) where {T,N}
    return (length(eachindex(hist.history_risk_free_rates)), N)
end

market_template(hist::VolumeMarketHistory) = hist.market

function market_template(hist, t)
    market = hist.market
    market.risk_free_rate = risk_free_rate(hist, t)
    return market
end
