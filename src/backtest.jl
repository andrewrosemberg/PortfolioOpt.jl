abstract type AbstractRecorder end

"""
    StateRecorder <: AbstractRecorder

Records a quantity of the market at each date of a backtest. A new recorder has to implement
[`record!`](@ref) and `get_records(recorder)`.
"""
abstract type StateRecorder <: AbstractRecorder end

"""
    record!(recorder::StateRecorder, market::PortfolioMarket, date)

Stores the state of `market` (after it was cleared) for `date` in `recorder`. Called by the
backtest for each date. The recorders of the package only use the [`PortfolioMarket`](@ref)
interface, so they work with any market type.
"""
function record! end

mutable struct WealthRecorder{T<:Real} <: StateRecorder
    record::DenseAxisArray{T,1}

    function WealthRecorder(dates; zero::T=0.0) where {T}
        date_step = dates[end] - dates[end-1]
        new{T}(DenseAxisArray{T}(undef, vcat(dates,dates[end]+date_step)))
    end
end

function record!(recorder::WealthRecorder, market::PortfolioMarket, date)
    recorder.record[date] = market_budget(market)
    return nothing
end

get_records(recorder::StateRecorder) = recorder.record
get_records(recorder::StateRecorder, t) = get_records(recorder)[t]
length(recorder::AbstractRecorder) = size(get_records(recorder), 1)
mutable struct ReturnsRecorder{T<:Real} <: StateRecorder
    record::DenseAxisArray{T,1}

    function ReturnsRecorder(dates; zero::T=0.0) where {T}
        new{T}(DenseAxisArray{T}(undef, dates))
    end
end

function record!(recorder::ReturnsRecorder, market::PortfolioMarket, date)
    recorder.record[date] = total_profit(market) / market_budget(market)
    return nothing
end

mutable struct DecisionRecorder{T<:Real} <: StateRecorder
    record::DenseAxisArray{T}

    function DecisionRecorder(numA::Int, dates; zero::T=0.0) where {T}
        new{T}(DenseAxisArray{T}(undef, dates, 1:numA))
    end
end

function record!(recorder::DecisionRecorder, market::PortfolioMarket, date)
    recorder.record[date, :] = bids(market)
    return nothing
end

get_records(recorder::DecisionRecorder, t) = get_records(recorder)[t, :]

default_state_recorders(numA::Int, date_range) = Dict(
    :wealth => WealthRecorder(date_range),
    :returns => ReturnsRecorder(date_range),
    :decisions => DecisionRecorder(numA, date_range)
)

"""
    portfolio_weights(formulation::PortfolioFormulation, num_assets::Integer, optimizer_factory;
        budget=1.0, volume_fee=0.0, allow_short_selling=false, risk_free_rate=0.0
    ) -> Vector{Float64}

Solves `formulation` on a fresh [`VolumeMarket`](@ref) with `num_assets` assets and returns the
optimal amount invested in each asset (the portfolio weights when `budget = 1.0`). Whatever is not
invested is held in the risk-free asset.

The keyword arguments are passed to the `VolumeMarket` constructor. If the solver does not find an
optimal solution, a warning is emitted and a vector of zeros is returned (see `change_bids!`).

Example (`returns` is a `T × N` matrix of past returns, one row per period):
```julia
d = MvNormal(vec(mean(returns; dims=1)), cov(returns))
formulation = PortfolioFormulation(MAX_SENSE,
    ObjectiveTerm(ExpectedReturn(d)), RiskConstraint(SqrtVariance(d), LessThan(0.02)),
)
w = portfolio_weights(formulation, size(returns, 2), Clarabel.Optimizer)
```
"""
function portfolio_weights(
    formulation::PortfolioFormulation,
    num_assets::Integer,
    optimizer_factory::Any;
    budget::Real=1.0,
    volume_fee::Real=0.0,
    allow_short_selling::Bool=false,
    risk_free_rate::Real=0.0,
)
    market = VolumeMarket(num_assets;
        budget=float(budget),
        volume_fee=float(volume_fee),
        allow_short_selling=allow_short_selling,
        risk_free_rate=float(risk_free_rate),
    )
    change_bids!(market, formulation, optimizer_factory)
    return copy(bids(market))
end


function day_backtest_market!(
    strategy_logic::Function,
    market_history::MarketHistory,
    date;
    state_recorders::Dict{Symbol,StateRecorder}=Dict(),
    optimization_recorder::Union{DenseAxisArray,Nothing}=nothing,
    provide_state::Bool=false
)
    market =  market_template(market_history, date)
    pointers = if provide_state
        strategy_logic(
            market,
            past_prices(market_history, date),
            Dict(:state_recorders => state_recorders, :date => date)
        )
    else
        strategy_logic(
            market,
            past_prices(market_history, date)
        )
    end

    clear_market!(market, current_prices(market_history, date))

    for recorder in values(state_recorders)
        record!(recorder, market, date)
    end

    if !isnothing(pointers)
        for (uid, pointer) in pointers
            optimization_recorder[date,uid] = value(pointer)
        end
    end

    set_market_budget(market, market_budget(market) + total_profit(market))

    return nothing
end

"""
    sequential_backtest_market(strategy_logic::Function, market_history::VolumeMarketHistory) -> Vector{Real}, Vector{Real}

Simple sequential backtest functionality for strategies that return an array of invested money per asset. Ideal for statefull strategies.

API:
```julia
wealth_strategy, returns_strategy = sequential_backtest_market(market_history; date_range=date_range) 
    do market, past_returns
    # ... Strategy definition ...
    return investment_decision
end
```

Arguments:
 - `strategy_logic::Function`: Function that represents the investment strategy.
 - `market_history::MarketHistory`: Market history.

Optional Keywork Arguments:
 - `date_range`: Dates to be simulated with corresponding entries in the market history.
"""
function sequential_backtest_market(
    strategy_logic::Function,
    market_history::MarketHistory,
    date_range;
    optimization_recorder::Union{DenseAxisArray,Nothing}=nothing,
    state_recorders::Dict{Symbol,StateRecorder}=default_state_recorders(num_assets(market_history), date_range)
)
    @assert issorted(date_range)

    @showprogress 1 "Computing..." for date in date_range
        day_backtest_market!(strategy_logic, market_history, date; 
            state_recorders=state_recorders, optimization_recorder=optimization_recorder, provide_state=true
        )
    end

    if haskey(state_recorders, :wealth)
        date_step = date_range[end] - date_range[end-1]
        record!(state_recorders[:wealth], market_template(market_history), date_range[end]+date_step)
    end
    
    return state_recorders, optimization_recorder
end
