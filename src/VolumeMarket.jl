import OptimalBids: Market, build_market, change_bids!, clear_market!, calculate_profit

mutable struct VolumeMarket <: OptimalBids.Market
    
end

"""
    build_market(::Type{Market}, params...) -> Market
Builds market of type Market using provided parameters (`params`).
"""
function build_market(typ::Type{M}, params...) where {M<:Market}
    throw(MethodError(build_market, (typ, params...)))
end

"""
    change_bids!(market::Market, new_bids::Vector)
Changes strategic agent's bids in the market to `new_bids`.
"""
function change_bids!(market::M, new_bids::Vector) where {M<:Market}
    throw(MethodError(change_bids!, (market, new_bids)))
end

"""
    clear_market!(market::Market)
Clears the market.
"""
function clear_market!(market::M) where {M<:Market}
    throw(MethodError(clear_market!, (market)))
end

"""
    calculate_profit(market::Market) -> NamedTuple{(:cleared_volumes, :clearing_prices, :profit), Tuple{Vector{Int}, Vector{Int}, Vector{Int}}}
Retrieves strategic agent's cleared volumes and prices from the market and calculates per bid profit.
"""
function calculate_profit(market::M) where {M<:Market}
    throw(MethodError(calculate_profit, (market)))
end
