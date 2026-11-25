"""
    get_test_data(; start_date=Date(2009, 9, 1), end_date=start_date + Year(1) + Month(3))

Daily closing prices of 6 assets (AAPL, BA, DELL, CAT, EBAY, F) between `start_date` and
`end_date`, as a `TimeArray`. The data ships with the MarketData package, which has to be loaded
(`using MarketData`) for this function to be available (package extension).
"""
function get_test_data end

"""
    mean_variance(returns::Array{T, 2}; digits::Union{Nothing,Int}=nothing) where {T<:Real}

Mean and Variance of returns
"""
function mean_variance(
    returns::Array{T,2}; digits::Union{Nothing,Int}=nothing
) where {T<:Real}
    r̄ = mean(returns; dims=1)'[:, 1]
    Σ = cov(returns)
    if !isnothing(digits)
        return round.(Σ; digits=digits), round.(r̄; digits=digits)
    end
    return Σ, r̄
end

"""
    max_sharpe(Σ::Array{T, 2}, r̄::Array{T, 1}, rf::T) where {T<:Real}

Maximize sharp coefficient allocation.
"""
function max_sharpe(Σ::Array{T,2}, r̄::Array{T,1}, rf::T) where {T<:Real}
    one = ones(size(r̄, 1))
    invΣ = pinv(Σ, 1E-25)
    v = invΣ * (r̄ .- one * rf)
    return v
end
