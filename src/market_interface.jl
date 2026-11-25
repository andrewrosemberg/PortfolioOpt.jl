"""
    PortfolioMarket

Supertype of markets in which the decision is the vector of amounts invested in each asset
(the bids). [`VolumeMarket`](@ref) is the implementation shipped with the package.

A new market type has to implement:
 - [`num_assets`](@ref)`(market)`: number of assets;
 - [`market_budget`](@ref)`(market)` and [`set_market_budget`](@ref)`(market, value)`: money available;
 - [`bids`](@ref)`(market)` and [`set_bids!`](@ref)`(market, bids)`: current investment decision;
 - [`clear_market!`](@ref)`(market, prices)`: sets the realized returns of the assets;
 - [`total_profit`](@ref)`(market)`: profit of the current bids after the market is cleared.

and can optionally implement [`market_model`](@ref) (the default is a long only portfolio within
the budget). With these, [`build_portfolio_model`](@ref), `change_bids!(market, formulation, optimizer)`,
the recorders and [`sequential_backtest_market`](@ref) work with the new market.

The interface does not depend on other packages. When `OptimalBids` is loaded, its
`change_bids!`, `clear_market!` and `calculate_profit` functions also accept any
`PortfolioMarket` (package extension).
"""
abstract type PortfolioMarket end

"""
    change_bids!(market::PortfolioMarket, new_bids::AbstractVector)
    change_bids!(market::PortfolioMarket, model::JuMP.Model, decision_variables)
    change_bids!(market::PortfolioMarket, formulation::PortfolioFormulation, optimizer_factory; record_measures=false)

Changes the investment decision (bids) of `market`: to `new_bids`, to the optimal value of
`decision_variables` after solving `model`, or to the optimal portfolio of `formulation`.
"""
function change_bids! end

"""
    clear_market!(market::PortfolioMarket, clearing_prices)

Clears `market` with the realized returns (`clearing_prices`) of the assets, so that the profit
of the current bids can be computed with [`total_profit`](@ref).
"""
function clear_market! end

"""
    calculate_profit(market::PortfolioMarket) -> NamedTuple

Details of the profit of the current bids after the market was cleared (e.g. for
[`VolumeMarket`](@ref): cleared volumes, clearing prices, profit of the assets and of the money
not invested).
"""
function calculate_profit end

"""
    num_assets(market::PortfolioMarket) -> Int

Number of assets traded in `market`.
"""
function num_assets end

"""
    market_budget(market::PortfolioMarket)

Money available to invest in `market`.
"""
function market_budget end

"""
    set_market_budget(market::PortfolioMarket, value)

Sets the money available to invest in `market` (used by the backtest to reinvest the profits).
"""
function set_market_budget end

"""
    bids(market::PortfolioMarket) -> AbstractVector

Current investment decision of `market` (amount invested per asset), or `nothing` if no bids
were set.
"""
function bids end

"""
    set_bids!(market::PortfolioMarket, new_bids::AbstractVector)

Sets the investment decision of `market`. `change_bids!(market, new_bids)` calls this function
for any [`PortfolioMarket`](@ref).
"""
function set_bids! end

"""
    total_profit(market::PortfolioMarket)

Total profit of the current bids after the market was cleared with `clear_market!`.
"""
function total_profit end

Base.length(market::PortfolioMarket) = num_assets(market)

function change_bids!(market::PortfolioMarket, new_bids::AbstractVector)
    return (set_bids!(market, new_bids); nothing)
end

"""
    market_model(market::PortfolioMarket, optimizer_factory; sense, model=Model(optimizer_factory)) -> (model, w)

Creates (or fills `model` with) the variables and constraints of an investment in `market` and
returns the model and the vector `w` of amounts invested per asset (length `num_assets(market)`).
Any objective added here (e.g. the risk free return of the money that is not invested) must use
`sense`; the formulation terms are added on top of it by [`portfolio_model!`](@ref).

The default method creates a long only portfolio, `w ≥ 0` and `sum(w) ≤ market_budget(market)`,
with no objective. Implement it for a new market type to change the feasible set.
"""
function market_model(
    market::PortfolioMarket,
    optimizer_factory::Any;
    sense::MOI.OptimizationSense=MAX_SENSE,
    model::JuMP.Model=Model(optimizer_factory),
)
    w = @variable(model, [1:num_assets(market)], lower_bound=0.0)
    @constraint(model, sum(w) <= market_budget(market))
    return model, w
end

"""
    build_portfolio_model(market::PortfolioMarket, formulation::PortfolioFormulation, optimizer_factory;
        record_measures=false, model=Model(optimizer_factory)) -> (model, w, pointers)

Builds, without solving, the JuMP model of `formulation` in `market`: [`market_model`](@ref)
followed by [`portfolio_model!`](@ref). Returns the model, the investment variables `w` and the
pointers to the value of each term (`nothing` unless `record_measures=true`).

Use it to add constraints before solving, then solve and set the bids with
`change_bids!(market, model, w)`:

```julia
model, w, _ = build_portfolio_model(market, formulation, optimizer)
@constraint(model, w .<= 0.3 * market_budget(market)) # position limits
change_bids!(market, model, w)
```
"""
function build_portfolio_model(
    market::PortfolioMarket,
    formulation::PortfolioFormulation,
    optimizer_factory::Any;
    record_measures::Bool=false,
    model::JuMP.Model=Model(optimizer_factory),
)
    model, w = market_model(
        market, optimizer_factory; sense=sense(formulation), model=model
    )
    pointers = portfolio_model!(model, formulation, w; record_measures=record_measures)
    return model, w, pointers
end

"""
    change_bids!(market::PortfolioMarket, model::JuMP.Model, decision_variables)

Solves `model` and sets the bids of `market` to the optimal value of `decision_variables`. If no
optimal solution is found, a warning is emitted and the bids are set to zero.
"""
function change_bids!(market::PortfolioMarket, model::JuMP.Model, decision_variables)
    optimize!(model)
    status = termination_status(model)
    if status !== MOI.OPTIMAL
        @warn "Did not find an optimal solution: status=$status"
        set_bids!(market, zeros(length(decision_variables)))
    else
        set_bids!(market, value.(decision_variables))
    end

    return nothing
end

"""
    change_bids!(market::PortfolioMarket, formulation::PortfolioFormulation, optimizer_factory; record_measures=false)

Builds the model of `formulation` in `market` (see [`build_portfolio_model`](@ref)), solves it and
sets the bids. Returns the pointers to the value of each term (`nothing` unless
`record_measures=true`, or if no optimal solution was found).
"""
function change_bids!(
    market::PortfolioMarket,
    formulation::PortfolioFormulation,
    optimizer_factory::Any;
    record_measures::Bool=false,
)
    model, decision_variables, pointers = build_portfolio_model(
        market, formulation, optimizer_factory; record_measures=record_measures
    )
    change_bids!(market, model, decision_variables)

    status = termination_status(model)
    if status !== MOI.OPTIMAL
        return nothing
    else
        return pointers
    end
end

# The model that owns a vector (or a scalar) of JuMP variables or expressions.
_owner_model(x::JuMP.AbstractVariableRef) = JuMP.owner_model(x)
function _owner_model(x::JuMP.GenericAffExpr)
    for v in keys(x.terms)
        return JuMP.owner_model(v)
    end
    return nothing
end
function _owner_model(x::JuMP.GenericQuadExpr)
    model = _owner_model(x.aff)
    model === nothing || return model
    for pair in keys(x.terms)
        return JuMP.owner_model(pair.a)
    end
    return nothing
end
_owner_model(x::JuMP.AbstractJuMPScalar) = JuMP.owner_model(x)
function _owner_model(x::AbstractArray)
    for xi in x
        model = _owner_model(xi)
        model === nothing || return model
    end
    return throw(
        ArgumentError("could not find the JuMP model: no element depends on a variable")
    )
end
