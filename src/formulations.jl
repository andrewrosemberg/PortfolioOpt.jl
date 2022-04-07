"""
    CenteredDROSet

Defines the ambiguity related to the distribution of a random variable, often used in
Robust Optimization (RO) and Distributionally Robust Optimization (DRO) problems.
It represents a bounded infinite set of distributions.
"""
abstract type CenteredDROSet{T<:Real, D<:ContinuousMultivariateSampleable} end

Base.length(s::CenteredDROSet) = length(distribution(s))

distribution(s::CenteredDROSet) = s.d

mean(s::CenteredDROSet) = mean(distribution(s))
cov(s::CenteredDROSet) = cov(distribution(s))

const ContinuousMultivariateSampleable = Sampleable{Multivariate, Continuous}

distribution(d::ContinuousMultivariateSampleable) = d

"""
    AmbiguitySet

Alias for Union{DROSet, ContinuousMultivariateSampleable}
"""
AmbiguitySet = Union{CenteredDROSet, ContinuousMultivariateSampleable}

abstract type ConcaveUtilityFunction end

struct PieceWiseUtility{T<:Real} <: ConcaveUtilityFunction
    coeficients::Array{T,1}
    intercepts::Array{T,1}
end

# TODO: Implement other useful utility functions

@enum Robustness begin
    EstimatedCase = 1
    WorstCase = 2
end

abstract type PortfolioStatisticalMeasure{S<:AmbiguitySet,R<:Robustness} end
struct ExpectedReturn{S<:AmbiguitySet,R<:Robustness} <: PortfolioStatisticalMeasure{S,R}
    ambiguity_set::S
end
struct Variance{S<:AmbiguitySet,R<:Robustness} <: PortfolioStatisticalMeasure{S,R}
    ambiguity_set::S
end

struct SqrtVariance{S<:AmbiguitySet,R<:Robustness} <: PortfolioStatisticalMeasure{S,R}
    ambiguity_set::S
end
struct ConditionalExpectedReturn{T<:Real,N<:Union{Int,Nothing},S<:AmbiguitySet,R<:Robustness} <: PortfolioStatisticalMeasure{S,R}
    ambiguity_set::S
    α::T
    num_samples::N
end
struct ExpectedUtility{C<:ConcaveUtilityFunction,S<:AmbiguitySet,R<:Robustness} <: PortfolioStatisticalMeasure{S,R}
    ambiguity_set::S
    utility::C
end

struct RiskConstraint{C<:Union{EqualTo, GreaterThan, LessThan}}
    measure::PortfolioStatisticalMeasure
    type::C
end
struct PortfolioFormulation
    objective::NamedTuple{(:measures, :weights), Tuple{Vector{PortfolioStatisticalMeasure}, Vector{Float64}}}
    risk_constraints::Vector{RiskConstraint}
end