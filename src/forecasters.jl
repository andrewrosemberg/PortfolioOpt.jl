"""
    AbstractForecaster

Supertype of the methods that turn past returns into a distribution or an ambiguity set, that can
then be used in a risk measure (e.g. `ExpectedReturn(forecast(f, returns))`).

A new forecaster has to implement [`forecast`](@ref)`(f, returns::AbstractMatrix)`.
"""
abstract type AbstractForecaster end

"""
    forecast(f::AbstractForecaster, returns::AbstractMatrix) -> AmbiguitySet

Returns the distribution (or ambiguity set) of the next period returns given the past `returns`,
a `T × N` matrix with one row per period and one column per asset (as `values(past_returns)` in a
backtest). Any other argument is converted with `values` (e.g. a `TimeArray`).
"""
function forecast end

function forecast(f::AbstractForecaster, returns)
    returns isa AbstractMatrix && throw(MethodError(forecast, (f, returns)))
    returns_matrix = values(returns)
    returns_matrix isa AbstractMatrix || throw(ArgumentError(
        "returns must be a T × N matrix (or a table whose `values` is one), got $(typeof(returns))"
    ))
    return forecast(f, returns_matrix)
end

_lookback_rows(returns::AbstractMatrix, ::Nothing) = returns
function _lookback_rows(returns::AbstractMatrix, lookback::Integer)
    T = size(returns, 1)
    return returns[max(1, T - lookback + 1):T, :]
end

"""
    EmpiricalGaussian(lookback=nothing)

Gaussian distribution (`MvNormal`) with the sample mean and covariance of the last `lookback`
periods (all periods if `nothing`).
"""
struct EmpiricalGaussian <: AbstractForecaster
    lookback::Union{Int,Nothing}
    function EmpiricalGaussian(lookback::Union{Integer,Nothing}=nothing)
        isnothing(lookback) || lookback >= 2 || throw(ArgumentError("lookback must be >= 2"))
        return new(lookback)
    end
end

function forecast(f::EmpiricalGaussian, returns::AbstractMatrix)
    r = _lookback_rows(returns, f.lookback)
    μ = vec(Statistics.mean(r; dims=1))
    Σ = Statistics.cov(r)
    return MvNormal(μ, Matrix(Symmetric(Σ)))
end

"""
    EmpiricalSamples(lookback=nothing)

Empirical distribution ([`DeterministicSamples`](@ref)) of the last `lookback` periods (all
periods if `nothing`). Note that `DeterministicSamples` stores one sample per column (`N × T`).
"""
struct EmpiricalSamples <: AbstractForecaster
    lookback::Union{Int,Nothing}
    function EmpiricalSamples(lookback::Union{Integer,Nothing}=nothing)
        isnothing(lookback) || lookback >= 1 || throw(ArgumentError("lookback must be >= 1"))
        return new(lookback)
    end
end

function forecast(f::EmpiricalSamples, returns::AbstractMatrix{T}) where {T<:Real}
    r = _lookback_rows(returns, f.lookback)
    return DeterministicSamples(Matrix{float(T)}(permutedims(r)))
end
