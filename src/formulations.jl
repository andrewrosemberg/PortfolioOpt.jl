abstract type ConcaveUtilityFunction end

struct PieceWiseUtility{T<:Real} <: ConcaveUtilityFunction
    c::Vector{T}
    b::Vector{T}

    function PieceWiseUtility{T}(c, b) where {T}
        @assert length(c) == length(b)
        return PieceWiseUtility(c, b)
    end
end

coeficients(u::PieceWiseUtility) = u.c
intercepts(u::PieceWiseUtility) = u.b

# TODO: Implement other useful utility functions

abstract type Robustness end
abstract type EstimatedCase <: Robustness end
abstract type WorstCase <: Robustness end

abstract type PortfolioStatisticalMeasure{S<:AmbiguitySet,R<:Robustness} end

struct ExpectedReturn{S<:AmbiguitySet,R<:Robustness} <: PortfolioStatisticalMeasure{S,R}
    ambiguity_set::S
end

function ExpectedReturn(ambiguity_set::S, R::Type{<:Robustness}=EstimatedCase) where {S<:AmbiguitySet}
    return ExpectedReturn{S,R}(ambiguity_set)
end
ambiguityset(m::ExpectedReturn) = m.ambiguity_set

struct Variance{S<:AmbiguitySet,R<:Robustness} <: PortfolioStatisticalMeasure{S,R}
    ambiguity_set::S
end

function Variance(ambiguity_set::S, R::Type{<:Robustness}=EstimatedCase) where {S<:AmbiguitySet}
    return Variance{S,R}(ambiguity_set)
end
ambiguityset(m::Variance) = m.ambiguity_set

struct SqrtVariance{S<:AmbiguitySet,R<:Robustness} <: PortfolioStatisticalMeasure{S,R}
    ambiguity_set::S
end

function SqrtVariance(ambiguity_set::S, R::Type{<:Robustness}=EstimatedCase) where {S<:AmbiguitySet}
    return SqrtVariance{S,R}(ambiguity_set)
end

ambiguityset(m::SqrtVariance) = m.ambiguity_set
struct ConditionalExpectedReturn{α,N<:Union{Int,Nothing},S<:AmbiguitySet,R<:Robustness} <: PortfolioStatisticalMeasure{S,R}
    ambiguity_set::S
    num_samples::N
end

function ConditionalExpectedReturn{R}(α::T, ambiguity_set::S, num_samples::N) where {T<:Real, N<:Union{Int,Nothing},S<:AmbiguitySet,R<:Robustness}
    return ConditionalExpectedReturn{α,N,S,R}(ambiguity_set, num_samples)
end
ambiguityset(m::ConditionalExpectedReturn) = m.ambiguity_set
sample_size(m::ConditionalExpectedReturn) = m.num_samples
function alpha_quantile(::ConditionalExpectedReturn{α,N,S,R}) where {α,N,S,R}
    return α
end

struct ExpectedUtility{C<:ConcaveUtilityFunction,S<:AmbiguitySet,R<:Robustness} <: PortfolioStatisticalMeasure{S,R}
    ambiguity_set::S
    utility::C
end
ambiguityset(m::ExpectedUtility) = m.ambiguity_set
utility(m::ExpectedUtility) = m.utility

struct RiskConstraint{C<:Union{EqualTo, GreaterThan, LessThan}}
    risk_measure::PortfolioStatisticalMeasure
    constraint_type::C
    uid::UUID

    function RiskConstraint(risk_measure::PortfolioStatisticalMeasure, constraint_type::C, uid::UUID=uuid1()) where {C}
        return new{C}(risk_measure, constraint_type, uid)
    end
end

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

struct ConeRegularizer{T<:Real}
    weight_matrix::Array{T,2}
    norm_cone::AbstractVectorSet
end

cone(m::ConeRegularizer) = m.norm_cone
weights(m::ConeRegularizer) = m.weight_matrix

function calculate_measure!(m::ConeRegularizer, w)
    model = owner_model(w)
    norm_value = @variable(model)
    @constraint(model, [norm_value; weights(m) * w] in cone(m)())
    return norm_value
end

struct ObjectiveTerm{T<:Real}
    term::Union{PortfolioStatisticalMeasure,ConeRegularizer{T}}
    weight::T
    uid::UUID

    function ObjectiveTerm(term::Union{PortfolioStatisticalMeasure,ConeRegularizer{T}}, weight::T=1.0, uid::UUID=uuid1()) where {T}
        return new{T}(term, weight, uid)
    end
end

uid(c::ObjectiveTerm) = c.uid
term(o::ObjectiveTerm) = o.term
weight(o::ObjectiveTerm) = o.weight

struct PortfolioFormulation
    objective_terms::Vector{ObjectiveTerm}
    risk_constraints::Vector{RiskConstraint}
    sense::MOI.OptimizationSense
end

PortfolioFormulation(O::ObjectiveTerm, R::RiskConstraint, sense::MOI.OptimizationSense) = PortfolioFormulation([O], [R], sense)
sense(formulation::PortfolioFormulation) = formulation.sense

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