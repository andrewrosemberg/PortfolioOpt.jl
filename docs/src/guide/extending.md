# Extending PortfolioOpt

PortfolioOpt is built around a few abstract types and functions that can be extended from your
own code, without changing the package:

| To add | Subtype | Implement |
|:--|:--|:--|
| a term of the objective or of a constraint | [`AbstractPortfolioTerm`](@ref) | [`calculate_measure!`](@ref) |
| a risk measure over a distribution or ambiguity set | [`PortfolioRiskMeasure`](@ref) | [`calculate_measure!`](@ref) |
| an ambiguity set | [`CenteredAmbiguitySet`](@ref) | [`distribution`](@ref), and [`calculate_measure!`](@ref) for the measures that use it |
| a way to estimate the distribution of returns | [`AbstractForecaster`](@ref) | [`forecast`](@ref) |
| a market | [`PortfolioMarket`](@ref) | [`num_assets`](@ref), [`market_budget`](@ref), [`set_market_budget`](@ref), [`bids`](@ref), [`set_bids!`](@ref), [`clear_market!`](@ref), [`total_profit`](@ref), and optionally [`market_model`](@ref) |
| a history of prices for backtests | [`MarketHistory`](@ref) | [`past_prices`](@ref), `current_prices`, `market_template`, [`num_assets`](@ref) |
| a quantity recorded in backtests | [`StateRecorder`](@ref) | [`record!`](@ref), `get_records` |

Remember to extend the functions of the package (`PortfolioOpt.calculate_measure!(...) = ...`),
not to define new functions with the same name.

## A new ambiguity set

The worst case expected return over a box around the mean,
``\{\mu : |\mu_i - \hat{r}_i| \leq \delta\}``, is ``\hat{r}'w - \delta \|w\|_1``. To use it, subtype
[`CenteredAmbiguitySet`](@ref) and implement [`calculate_measure!`](@ref) for
[`ExpectedReturn`](@ref) over the new set. The function receives the investment variables `w`,
adds whatever variables and constraints it needs to their model, and returns the value of the
measure as a JuMP expression.

```@example extending
using PortfolioOpt, Distributions, Clarabel, LinearAlgebra

struct BoxSet{T<:Real,D<:Sampleable} <: CenteredAmbiguitySet{T,D}
    d::D
    δ::T
end

PortfolioOpt.distribution(s::BoxSet) = s.d

function PortfolioOpt.calculate_measure!(m::ExpectedReturn{<:BoxSet}, w)
    model = JuMP.owner_model(first(w))
    s = ambiguityset(m)
    t = @variable(model)
    @constraint(model, [t; w] in MOI.NormOneCone(length(w) + 1))
    return dot(mean(s), w) - s.δ * t
end
nothing # hide
```

It can now be used like the ambiguity sets of the package:

```@example extending
d = MvNormal([0.010, 0.012, 0.008], Diagonal([0.02, 0.03, 0.01] .^ 2))
optimizer = optimizer_with_attributes(Clarabel.Optimizer, "verbose" => false)

formulation = PortfolioFormulation(MAX_SENSE,
    ObjectiveTerm(ExpectedReturn(BoxSet(d, 0.005))),
    RiskConstraint(SqrtVariance(d), LessThan(0.015)),
)
w = portfolio_weights(formulation, 3, optimizer)
```

A new risk measure is the same: subtype [`PortfolioRiskMeasure`](@ref) (or
[`AbstractPortfolioTerm`](@ref) if it does not depend on a distribution) and implement
[`calculate_measure!`](@ref). Both can be used in an [`ObjectiveTerm`](@ref) and in a
[`RiskConstraint`](@ref).

## A new market

A [`PortfolioMarket`](@ref) holds the money available and the current investment. The default
[`market_model`](@ref) is a long only portfolio within the budget; implement it to change the
feasible set (e.g. short selling or fees, as in [`VolumeMarket`](@ref)).

```@example extending
mutable struct CashMarket <: PortfolioMarket
    n::Int
    budget::Float64
    bids::Union{Vector{Float64},Nothing}
    returns::Union{Vector{Float64},Nothing}
end
CashMarket(n; budget=1.0) = CashMarket(n, budget, nothing, nothing)

PortfolioOpt.num_assets(m::CashMarket) = m.n
PortfolioOpt.market_budget(m::CashMarket) = m.budget
PortfolioOpt.set_market_budget(m::CashMarket, v) = (m.budget = v)
PortfolioOpt.bids(m::CashMarket) = m.bids
PortfolioOpt.set_bids!(m::CashMarket, b::AbstractVector) = (m.bids = collect(Float64, b))
PortfolioOpt.clear_market!(m::CashMarket, r::AbstractVector) = (m.returns = r)
PortfolioOpt.total_profit(m::CashMarket) = dot(m.bids, m.returns)
nothing # hide
```

The formulations, the recorders and [`sequential_backtest_market`](@ref) only use these functions,
so they work with the new market. [`build_portfolio_model`](@ref) returns the model before it is
solved, which is useful to add constraints that are not part of the formulation, such as position
limits:

```@example extending
market = CashMarket(3)
model, w, _ = build_portfolio_model(market, formulation, optimizer)
@constraint(model, w .<= 0.4 * market_budget(market))
change_bids!(market, model, w)
bids(market)
```

## Extending from another package

The interfaces above have no dependencies beyond the core of PortfolioOpt (JuMP, MathOptInterface
and Distributions), so a package that adds richer market types or stronger formulations only needs
to depend on PortfolioOpt: subtype [`PortfolioMarket`](@ref), [`PortfolioRiskMeasure`](@ref),
[`CenteredAmbiguitySet`](@ref) or [`AbstractForecaster`](@ref), and add methods to the functions of
PortfolioOpt as in the examples on this page. Integrations with optional packages (TimeSeries,
MarketData, OptimalBids) are package extensions, loaded only when those packages are.

## Forecasts from past returns

A forecaster turns a `T × N` matrix of past returns (one row per period) into a distribution or
an ambiguity set. [`EmpiricalGaussian`](@ref) and [`EmpiricalSamples`](@ref) are included, and a
new one only needs [`forecast`](@ref):

```@example extending
using Random
Random.seed!(1)

struct DiagonalGaussian <: AbstractForecaster end
PortfolioOpt.forecast(::DiagonalGaussian, returns::AbstractMatrix) =
    MvNormal(vec(mean(returns; dims=1)), Diagonal(vec(var(returns; dims=1))))

past_returns = rand(d, 60)' # T × N
samples = forecast(EmpiricalSamples(30), past_returns) # N × 30 samples
formulation = PortfolioFormulation(MAX_SENSE,
    ObjectiveTerm(ExpectedReturn(forecast(DiagonalGaussian(), past_returns))),
    RiskConstraint(ConditionalExpectedReturn(samples; α=0.1), GreaterThan(-0.05)),
)
portfolio_weights(formulation, 3, optimizer)
```
