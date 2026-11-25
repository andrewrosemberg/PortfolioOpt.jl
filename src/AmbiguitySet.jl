const ContinuousMultivariateSampleable = Sampleable{Multivariate,Continuous}

"""
    distribution(s::AmbiguitySet) -> Sampleable

Nominal (center) distribution of an ambiguity set; a distribution is its own nominal distribution.
New [`CenteredAmbiguitySet`](@ref) types should implement it: `length`, `mean` and `cov` of the set
default to those of its distribution.
"""
distribution(d::Sampleable) = d

"""
    CenteredAmbiguitySet

Defines the ambiguity related to the distribution of a random variable, often used in
Robust Optimization (RO) and Distributionally Robust Optimization (DRO) problems.
It represents a bounded infinite set of distributions.
"""
abstract type CenteredAmbiguitySet{T<:Real,D<:Sampleable} end

Base.length(s::CenteredAmbiguitySet) = length(distribution(s))

mean(s::CenteredAmbiguitySet) = mean(distribution(s))
cov(s::CenteredAmbiguitySet) = cov(distribution(s))

"""
    AmbiguitySet

Alias for Union{CenteredAmbiguitySet, ContinuousMultivariateSampleable}
"""
const AmbiguitySet = Union{CenteredAmbiguitySet,Sampleable}
