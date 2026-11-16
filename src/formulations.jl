abstract type ConcaveUtilityFunction end

struct PieceWiseUtility{T<:Real} <: ConcaveUtilityFunction
    c::Vector{T}
    b::Vector{T}

    function PieceWiseUtility(c::Vector{T}=[1.0], b::Vector{T}=[0.0]) where T
        @assert length(c) == length(b)
        return new{T}(c, b)
    end
end

coefficients(u::PieceWiseUtility) = u.c
intercepts(u::PieceWiseUtility) = u.b

==(a::PieceWiseUtility, b::PieceWiseUtility) = coefficients(a) == coefficients(b) && intercepts(a) == intercepts(b)

# TODO: Implement other useful utility functions

"""
    AbstractPortfolioTerm

Supertype of everything that can be evaluated on the portfolio decision variables, i.e. that can be
used in an [`ObjectiveTerm`](@ref) or a [`RiskConstraint`](@ref). Subtypes include
[`PortfolioRiskMeasure`](@ref) and [`ConeRegularizer`](@ref).

To add a new term, subtype it and implement [`calculate_measure!`](@ref):

```julia
struct MyTerm <: AbstractPortfolioTerm end
PortfolioOpt.calculate_measure!(::MyTerm, w) = ... # JuMP expression or variable
```
"""
abstract type AbstractPortfolioTerm end

"""
    calculate_measure!(term::AbstractPortfolioTerm, w) -> JuMP expression

Adds to the model that owns the decision variables `w` whatever auxiliary variables and
constraints are needed to represent `term`, and returns a scalar JuMP expression (or variable)
with its value for the portfolio `w`. The model is obtained from `w` (e.g. with
`JuMP.owner_model(first(w))`).

This is the main extension point of the package: implement it for a new
[`AbstractPortfolioTerm`](@ref), or for an existing measure with a new ambiguity set (for example
`calculate_measure!(m::ExpectedReturn{<:MySet}, w)`).
"""
function calculate_measure! end

"""
    PortfolioRiskMeasure{S<:AmbiguitySet} <: AbstractPortfolioTerm

Supertype of the risk (or return) measures that are evaluated over an ambiguity set or
distribution `S`, e.g. [`ExpectedReturn`](@ref), [`SqrtVariance`](@ref) and
[`ConditionalExpectedReturn`](@ref). Subtypes should implement [`ambiguityset`](@ref) and
[`calculate_measure!`](@ref).
"""
abstract type PortfolioRiskMeasure{S<:AmbiguitySet} <: AbstractPortfolioTerm end

"""
    ambiguityset(m::PortfolioRiskMeasure)

Returns the ambiguity set (or distribution) over which the measure `m` is evaluated.
"""
ambiguityset(m::PortfolioRiskMeasure) = m.ambiguity_set

"""
    ExpectedReturn(s::AmbiguitySet)

Expected return of the portfolio. For a distribution this is the mean return; for an ambiguity
set it is the worst case expected return over the set.
"""
struct ExpectedReturn{S<:AmbiguitySet} <: PortfolioRiskMeasure{S}
    ambiguity_set::S
end

ambiguityset(m::ExpectedReturn) = m.ambiguity_set

"""
    Variance(s::AmbiguitySet)

Variance of the portfolio return.
"""
struct Variance{S<:AmbiguitySet} <: PortfolioRiskMeasure{S}
    ambiguity_set::S
end

ambiguityset(m::Variance) = m.ambiguity_set

"""
    SqrtVariance(s::AmbiguitySet)

Standard deviation of the portfolio return.
"""
struct SqrtVariance{S<:AmbiguitySet} <: PortfolioRiskMeasure{S}
    ambiguity_set::S
end

ambiguityset(m::SqrtVariance) = m.ambiguity_set

"""
    WorstCaseReturn(s::AmbiguitySet)

Worst case return of the portfolio over the support of `s`. For [`DeterministicSamples`](@ref) it
is the smallest return over the samples. `ConditionalExpectedReturn` with `α = 0` is equivalent.
"""
struct WorstCaseReturn{S<:AmbiguitySet} <: PortfolioRiskMeasure{S}
    ambiguity_set::S
end

ambiguityset(m::WorstCaseReturn) = m.ambiguity_set

"""
    ConditionalExpectedReturn(s::AmbiguitySet; α=0.05)

Expected return of the portfolio over its worst `α` fraction of outcomes (minus the conditional
value at risk). `α = 0` gives the worst case return ([`WorstCaseReturn`](@ref)) and `α = 1` the
expected return.
"""
struct ConditionalExpectedReturn{α,S<:AmbiguitySet} <: PortfolioRiskMeasure{S}
    ambiguity_set::S
end

# TODO: deprecate
function ConditionalExpectedReturn(α::T, ambiguity_set::S) where {T<:Real,S<:AmbiguitySet}
    return ConditionalExpectedReturn{α,S}(ambiguity_set)
end

function ConditionalExpectedReturn(ambiguity_set::S; α::T=0.05) where {T<:Real,S<:AmbiguitySet}
    return ConditionalExpectedReturn{α,S}(ambiguity_set)
end

ambiguityset(m::ConditionalExpectedReturn) = m.ambiguity_set

function alpha_quantile(::ConditionalExpectedReturn{α,S}) where {α,S}
    return α
end

"""
    ExpectedUtility(s::AmbiguitySet, utility::ConcaveUtilityFunction)

Expected (worst case, for an ambiguity set) utility of the portfolio return.
"""
struct ExpectedUtility{C<:ConcaveUtilityFunction,S<:AmbiguitySet} <: PortfolioRiskMeasure{S}
    ambiguity_set::S
    utility::C
end

ambiguityset(m::ExpectedUtility) = m.ambiguity_set
utility(m::ExpectedUtility) = m.utility

"""
    RiskConstraint(measure::AbstractPortfolioTerm, set::Union{EqualTo,GreaterThan,LessThan}, uid=uuid1())

Constrains the value of `measure` (see [`calculate_measure!`](@ref)) to `set`, e.g.
`RiskConstraint(SqrtVariance(d), LessThan(0.02))`.
"""
struct RiskConstraint{C<:Union{EqualTo, GreaterThan, LessThan}}
    risk_measure::AbstractPortfolioTerm
    constraint_type::C
    uid::UUID

    function RiskConstraint(risk_measure::AbstractPortfolioTerm, constraint_type::C, uid::UUID=uuid1()) where {C}
        return new{C}(risk_measure, constraint_type, uid)
    end
end

"""
    uid(term::Union{ObjectiveTerm,RiskConstraint}) -> UUID

Identifier of an objective term or risk constraint. It is the key of the term in the pointers
returned by [`portfolio_model!`](@ref) (and [`build_portfolio_model`](@ref)) when
`record_measures=true`.
"""
uid(c::RiskConstraint) = c.uid
risk_measure(c::RiskConstraint) = c.risk_measure
constant(c::RiskConstraint) = constant(c.constraint_type)

function constraint!(model::JuMP.Model, r::RiskConstraint{LessThan{T}}, decision_variables) where {T}
    pointer_term = calculate_measure!(risk_measure(r), decision_variables)
    @constraint(model, pointer_term <= constant(r))
    return pointer_term
end

function constraint!(model::JuMP.Model, r::RiskConstraint{GreaterThan{T}}, decision_variables) where {T}
    pointer_term = calculate_measure!(risk_measure(r), decision_variables)
    @constraint(model, pointer_term >= constant(r))
    return pointer_term
end

function constraint!(model::JuMP.Model, r::RiskConstraint{EqualTo{T}}, decision_variables) where {T}
    pointer_term = calculate_measure!(risk_measure(r), decision_variables)
    @constraint(model, pointer_term == constant(r))
    return pointer_term
end

"""
    ConeRegularizer(norm_cone, weight_matrix=I)

Norm of the weighted portfolio, `t` such that `[t; weight_matrix * w] in norm_cone`, e.g.
`ConeRegularizer(MOI.NormOneCone(n + 1))` for an L1 penalty.
"""
struct ConeRegularizer <: AbstractPortfolioTerm
    norm_cone::Union{AbstractVectorSet, MOI.AbstractVectorSet}
    weight_matrix::Union{Matrix,UniformScaling}

    function ConeRegularizer(norm_cone, weight_matrix=I)
        return new(norm_cone, weight_matrix)
    end
end

cone(m::ConeRegularizer) = m.norm_cone
weights(m::ConeRegularizer) = m.weight_matrix

function calculate_measure!(m::ConeRegularizer, w)
    model = _owner_model(w)
    norm_value = @variable(model)
    @constraint(model, [norm_value; weights(m) * w] in cone(m))
    return norm_value
end

"""
    ObjectiveTerm(term::AbstractPortfolioTerm, weight=1.0, uid=uuid1())

Adds `weight` times the value of `term` (see [`calculate_measure!`](@ref)) to the objective.
"""
struct ObjectiveTerm{T<:Real}
    term::AbstractPortfolioTerm
    weight::T
    uid::UUID

    function ObjectiveTerm(term::AbstractPortfolioTerm, weight::T=1.0, uid::UUID=uuid1()) where {T}
        return new{T}(term, weight, uid)
    end
end

uid(c::ObjectiveTerm) = c.uid
term(o::ObjectiveTerm) = o.term
weight(o::ObjectiveTerm) = o.weight

"""
    PortfolioFormulation(sense, objective_terms, risk_constraints=RiskConstraint[])

Portfolio optimization problem: the objective is the weighted sum of the `objective_terms`
(maximized or minimized according to `sense`), subject to the `risk_constraints`.
"""
struct PortfolioFormulation
    sense::MOI.OptimizationSense
    objective_terms::Vector{<:ObjectiveTerm}
    risk_constraints::Vector{<:RiskConstraint}

    function PortfolioFormulation(sense::MOI.OptimizationSense,
        objective_terms::Vector{<:ObjectiveTerm}, 
        risk_constraints::Vector{<:RiskConstraint}=Vector{RiskConstraint}(),
    )
        return new(sense, objective_terms, risk_constraints)
    end
end

PortfolioFormulation(sense::MOI.OptimizationSense, obj::ObjectiveTerm{T}, risk::RiskConstraint{C}) where {T,C} = PortfolioFormulation(sense, [obj], [risk])
PortfolioFormulation(sense::MOI.OptimizationSense, obj::ObjectiveTerm{T}, risk_constraints::Vector{<:RiskConstraint}) where {T} = PortfolioFormulation(sense, [obj], risk_constraints)
PortfolioFormulation(sense::MOI.OptimizationSense, obj::Vector{<:ObjectiveTerm}, risk::RiskConstraint) = PortfolioFormulation(sense, obj, [risk])
PortfolioFormulation(sense::MOI.OptimizationSense, obj::ObjectiveTerm{T}) where {T} = PortfolioFormulation(sense, [obj])
sense(formulation::PortfolioFormulation) = formulation.sense

"""
    portfolio_model!(model::JuMP.Model, formulation::PortfolioFormulation, w; record_measures=false)

Adds the objective terms and risk constraints of `formulation` to `model`, where `w` are the
investment decision variables (e.g. created by [`market_model`](@ref)). The objective terms are
added to the current objective of `model`.

If `record_measures=true`, returns a `Dict` from the [`uid`](@ref) of each term to the JuMP
expression of its value; otherwise returns `nothing`.
"""
function portfolio_model!(model::JuMP.Model, formulation::PortfolioFormulation, decision_variables; record_measures::Bool=false)
    pointers = if record_measures
        Dict()
    else
        nothing
    end

    # objective
    obj = objective_function(model)
    for obj_term in formulation.objective_terms
        pointer_term = calculate_measure!(term(obj_term), decision_variables)
        obj += pointer_term .* weight(obj_term)

        if record_measures
            pointers[uid(obj_term)] = pointer_term
        end
    end

    drop_zeros!(obj)
    set_objective(model, sense(formulation), obj)

    # risk constraints
    for risk_constraint in formulation.risk_constraints
        pointer_term = constraint!(model, risk_constraint, decision_variables)
        
        if record_measures
            pointers[uid(risk_constraint)] = pointer_term
        end
    end

    return pointers
end