
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
- `γ1::Float64`: Uniform uncertainty around the mean (has to be greater than 0).
- `γ2::Float64`: Uncertainty around the covariance (has to be greater than 1).

References:
- Delage paper on moment uncertainty: https://www.researchgate.net/publication/220244490_Distributionally_Robust_Optimization_Under_Moment_Uncertainty_with_Application_to_Data-Driven_Problems

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

MomentUncertainty(d::Sampleable; γ1, γ2) = MomentUncertainty(d, γ1, γ2)

distribution(s::MomentUncertainty) = s.d

"""
    calculate_measure!(m::ExpectedUtility{U,S,R}, w) where {U<:PieceWiseUtility,S<:MomentUncertainty,R<:WorstCase}

Returns worst case utility return (WCR) under distribution uncertainty defined by MomentUncertainty ambiguity set ([`MomentUncertainty`](@ref)).

Arguments:
 - `w`: portfolio optimization investment variable ("weights").
 - `m::ExpectedUtility`: Struct containing information about the utility and ambiguity set.
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

"""
    DuWassersteinBall <: CenteredAmbiguitySet

```math
\\left\\{ r  \\; \\middle| \\begin{array}{ll}
s.t.  \\quad d_w(P, \\hat{P}_N) \\leq \\epsilon \\\\
\\quad \\quad || \\xi || \\leq \\Lambda \\\\
\\quad \\quad [\\xi; 0] + [0_{m x 1}; \\Lambda] \\in K \\\\
\\quad \\quad K = {[\\omega, \\pi] \\in R^m x R: \\pi \\geq ||\\omega||^*} \\\\
\\end{array}
\\right\\} \\\\
```

Atributes:
- `d::ContinuousMultivariateSampleable`: Samples from the parent distribution
- `ϵ::Float64`: Wasserstein distance from sampled distribution (has to be greater than 0). (default: 0.01)
- `Λ::Float64`: Uncertainty around sampled values (has to be greater than 0). (default: maximum(d))

References:
- NingNing paper on Wasserstein DRO (Corollary 1-3): https://ieeexplore.ieee.org/abstract/document/9311154
"""
struct DuWassersteinBall{T<:Real, D<:ContinuousMultivariateSampleable} <: CenteredAmbiguitySet{T,D}
    d::D
    ϵ::T
    Λ::T
    Q::Array{T,2}
    norm_cone::Real

    # Inner constructor for validating arguments
    function DuWassersteinBall{T, D}(
        d::D, ϵ::T, Λ::T, Q::Array{T,2}, norm_cone::Real
    ) where {T<:Real, D<:ContinuousMultivariateSampleable}
        length(d) == size(Q,1) == size(Q,2) || throw(ArgumentError(
            "Distribution ($(length(d))) and Q ($(size(Q,2))) must have coherent dimensions (m and mxm)"
        ))
        ϵ >= 0 || throw(ArgumentError("ϵ must be >= 0"))
        Λ >= 0 || throw(ArgumentError("Λ must be >= 0"))
        return new{T, D}(d, ϵ, Λ, Q, norm_cone)
    end
end

# Default outer constructor
function DuWassersteinBall(
    d::D, ϵ::T, Λ::T, Q::Array{T,2}, norm_cone::Real
) where {T<:Real, D<:ContinuousMultivariateSampleable}
    DuWassersteinBall{T, D}(d, ϵ, Λ, Q, norm_cone)
end

# Kwarg constructor with defaults
function DuWassersteinBall(
    d::ContinuousMultivariateSampleable;
    ϵ=0.01,
    norm_cone=Inf,
    Λ=default_DuWassersteinBall_lambda(d, norm_cone),
    Q=Matrix(I(length(d))* 1.0)
)
    return DuWassersteinBall(d, ϵ, Λ, Q, norm_cone)
end

distribution(s::DuWassersteinBall) = s.d

default_DuWassersteinBall_lambda(d::Sampleable, norm_cone::Real; num_samples::Int=20, rng::AbstractRNG=MersenneTwister(123)) = maximum(
    norm.(eachrow(rand(rng, d, num_samples)), norm_cone)
)

# Serves as a mapping from the norm's power to the appropriate cones
const primal_cone = Dict(
    "Inf" => MOI.NormInfinityCone,
    "1.0" => MOI.NormOneCone,
    "2.0" => MOI.SecondOrderCone
)

"""
objective_function!(model, f, ambiguity_set, fee_rates, samples)

"""
function calculate_measure!(measure::ConditionalExpectedReturn{1.0,S,R}, w) where {S<:DuWassersteinBall,R}
    model = owner_model(w)
    ambiguity_set = ambiguityset(measure)

    # parameters
    N = sample_size(ambiguity_set)
    ξ = rand(distribution(ambiguity_set), N)

    m = length(ambiguity_set)

    ϵ = ambiguity_set.ϵ
    Λ = ambiguity_set.Λ
    Q = ambiguity_set.Q
    Q_inv = pinv(Q)
    K = primal_cone[string(ambiguity_set.norm_cone)]

    λ = @variable(model)
    e = @variable(model)
    s = @variable(model, [1:N])
    ν = @variable(model, [i=1:N, j=1:m])
    τ = @variable(model, [i=1:N])

    @constraint(model, [i=1:N], - dot(w, ξ[:, i])
        + ν[i, :]' * Q_inv * ξ[:, i] + Λ * τ[i] <= s[i]
    )

    @constraint(model, [i=1:N], [λ; - Q_inv * ν[i, :] + w] in K(m + 1))

    @constraint(model, [i=1:N], [τ[i]; ν[i, :]] in MOI.dual_set(K(m + 1)))

    return - (λ * ϵ + sum(s) / N)
end
