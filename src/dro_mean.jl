
"""
MomentUncertainty <: CenteredAmbiguitySet

```math
\\left\\{ r  \\; \\middle| \\begin{array}{ll}
s.t.  \\quad (\\mathbb{E} [r] - \\hat{r}) ' \\Sigma^{-1} (\\mathbb{E} [r] - \\hat{r}) \\leq \\gamma_1 \\\\
\\quad \\quad \\mathbb{E} [ (r - \\hat{r}) ' (r - \\hat{r}) ] \\leq \\gamma_2 \\Sigma \\\\
\\end{array}
\\right\\} \\\\
```

Atributes:
- `d::Sampleable{Multivariate, Continous}`: The parent distribution with an uncertain mean
- `γ1::Float64`: Uniform uncertainty around the mean (has to be greater than 0). (default: std(dist) / 5)
- `γ2::Float64`: Uncertainty around the covariance (has to be greater than 1). (default: 3.0)

References:
- Delage paper on moment uncertainty (implemented): https://www.researchgate.net/publication/220244490_Distributionally_Robust_Optimization_Under_Moment_Uncertainty_with_Application_to_Data-Driven_Problems

"""
struct MomentUncertainty{T<:Real, D<:ContinuousMultivariateSampleable} <: CenteredAmbiguitySet{T,D}
    d::D
    γ1::T
    γ2::T

    # Inner constructor for validating arguments
    function MomentUncertainty{T, D}(
        d::D, γ1::T, γ2::T
    ) where {T<:Real, D<:ContinuousMultivariateSampleable}
        γ1 >= 0 || throw(ArgumentError("γ1 must be >= 0"))
        γ2 >= 1 || throw(ArgumentError("γ2 must be >= 1"))
        return new{T, D}(d, γ1, γ2)
    end
end

# Default outer constructor
function MomentUncertainty(
    d::D, γ1::T, γ2::T
) where {T<:Real, D<:ContinuousMultivariateSampleable}
    MomentUncertainty{T, D}(d, γ1, γ2)
end

distribution(s::MomentUncertainty) = s.d

"""
    calculate_measure!(measure::ExpectedReturn{MomentUncertainty,WorstCase}, w)

Returns worst case utility return (WCR) under distribution uncertainty defined by MomentUncertainty ambiguity set ([`MomentUncertainty`](@ref)).

Arguments:
 - `w`: portfolio optimization investment variable ("weights").
 - `s::MomentUncertainty`: Struct containing atributes of MomentUncertainty ambiguity set.
"""
function calculate_measure!(m::ExpectedUtility{U,S,R}, w) where {U<:PieceWiseUtility,S<:MomentUncertainty,R<:WorstCase}
    model = owner_model(w)
    s = ambiguityset(m)
    utility_function = utility(m)
    
    # parameters
    means = Vector(Distributions.mean(s.d))
    Σ = Matrix(Distributions.cov(s.d))
    n = length(s)
    a = coefficients(utility_function)
    b = intercepts(utility_function)
    γ1 = s.γ1
    γ2 = s.γ2
    K = length(a)

    # dual variables
    @variable(model, P[i=1:n, j=1:n])
    @variable(model, p[i=1:n])
    @variable(model, s)
    @variable(model, Q[i=1:n, j=1:n])
    @variable(model, q[i=1:n])
    @variable(model, r)

    # constraints: from duality
    @constraint(model, p .== -q / 2 - Q * means)
    @constraint(model, [[P p]; [p' s]] in PSDCone())
    for k in 1:K
        @constraint(
            model, [[Q (q / 2 + a[k] * w / 2)]; [(q' / 2 + a[k] * w' / 2) (r + b[k])]] in PSDCone()
        )
    end

    return -(γ2 * dot(Σ, Q) - first(means'Q * means) + r + dot(Σ, P) - 2 * dot(means, p) + γ1 * s)
end
