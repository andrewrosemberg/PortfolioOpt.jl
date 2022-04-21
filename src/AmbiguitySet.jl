"""
    CenteredAmbiguitySet

Defines the ambiguity related to the distribution of a random variable, often used in
Robust Optimization (RO) and Distributionally Robust Optimization (DRO) problems.
It represents a bounded infinite set of distributions.
"""
abstract type CenteredAmbiguitySet{T<:Real, D<:ContinuousMultivariateSampleable} end

Base.length(s::CenteredAmbiguitySet) = length(distribution(s))

mean(s::CenteredAmbiguitySet) = mean(distribution(s))
cov(s::CenteredAmbiguitySet) = cov(distribution(s))

const ContinuousMultivariateSampleable = Sampleable{Multivariate, Continuous}

distribution(d::ContinuousMultivariateSampleable) = d

"""
    AmbiguitySet

Alias for Union{CenteredAmbiguitySet, ContinuousMultivariateSampleable}
"""
AmbiguitySet = Union{CenteredAmbiguitySet, ContinuousMultivariateSampleable}