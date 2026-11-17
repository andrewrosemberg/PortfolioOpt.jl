"""
    MomentUncertainty <: CenteredAmbiguitySet

    MomentUncertainty(d, γ1, γ2)
    MomentUncertainty(d, γ1, γ2, lower, upper)
    MomentUncertainty(d; γ1, γ2, lower=nothing, upper=nothing)

Distributions of the returns ``r`` whose first two moments are close to those of the nominal
distribution `d` (with mean ``\\hat{r}`` and covariance ``\\Sigma``), and, optionally, whose
support is contained in the box ``[\\underline{r}, \\overline{r}]``:

```math
\\left\\{ P  \\; \\middle| \\begin{array}{ll}
s.t.  \\quad (\\mathbb{E}_P [r] - \\hat{r}) ' \\Sigma^{-1} (\\mathbb{E}_P [r] - \\hat{r}) \\leq \\gamma_1 \\\\
\\quad \\quad \\mathbb{E}_P [ (r - \\hat{r}) (r - \\hat{r})' ] \\preceq \\gamma_2 \\Sigma \\\\
\\quad \\quad P(\\underline{r} \\leq r \\leq \\overline{r}) = 1 \\\\
\\end{array}
\\right\\} \\\\
```

Atributes:
- `d::Sampleable{Multivariate, Continous}`: The nominal distribution (only its mean and covariance are used).
- `γ1::Float64`: Uncertainty around the mean (has to be greater than or equal to 0).
- `γ2::Float64`: Uncertainty around the covariance (has to be greater than or equal to 1).
- `lower::Vector`, `upper::Vector`: Bounds of the support of the returns (default: `-Inf` and
  `Inf`, i.e. unbounded support). Infinite entries are not bounded. The mean of `d` has to be
  inside the box.

The worst case measures implemented over this set are `ExpectedUtility` with a
`PieceWiseUtility`, `ExpectedReturn` and `ConditionalExpectedReturn` (minus the worst
case CVaR), all reformulated as semidefinite programs (SDP).

With unbounded support the reformulations are exact (Delage & Ye, 2010). With a bounded support,
the constraint "for every return in the box" is handled with one nonnegative multiplier per finite
bound (as in Yang et al., 2014), which gives a conservative approximation: the value is a lower
bound of the exact worst case and is never smaller than the value with unbounded support.

References:
- Delage, E., & Ye, Y. (2010). Distributionally robust optimization under moment uncertainty with
  application to data-driven problems. Operations Research, 58(3), 595-612.
  https://doi.org/10.1287/opre.1090.0741
- Yang, L., Li, Y., Zhou, Z., & Chen, K. (2014). Distributionally robust return-risk optimization
  models and their applications. Journal of Applied Mathematics, 2014, 784715.
  https://doi.org/10.1155/2014/784715
"""
struct MomentUncertainty{T<:Real, D<:ContinuousMultivariateSampleable} <: CenteredAmbiguitySet{T,D}
    d::D
    γ1::T
    γ2::T
    lower::Vector{T}
    upper::Vector{T}

    # Inner constructor for validating arguments
    function MomentUncertainty{T, D}(
        d::D, γ1::T, γ2::T, lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real}
    ) where {T<:Real, D<:ContinuousMultivariateSampleable}
        γ1 >= 0 || throw(ArgumentError("γ1 must be >= 0"))
        γ2 >= 1 || throw(ArgumentError("γ2 must be >= 1"))
        n = length(d)
        length(lower) == n || throw(ArgumentError(
            "Distribution ($(n)) and lower ($(length(lower))) must have the same length"
        ))
        length(upper) == n || throw(ArgumentError(
            "Distribution ($(n)) and upper ($(length(upper))) must have the same length"
        ))
        μ = Vector(Distributions.mean(d))
        all(lower .<= μ .<= upper) || throw(ArgumentError(
            "the mean of the distribution must be inside the support [lower, upper]"
        ))
        return new{T, D}(d, γ1, γ2, convert(Vector{T}, lower), convert(Vector{T}, upper))
    end
end

# unbounded support
function MomentUncertainty{T, D}(d::D, γ1::T, γ2::T) where {T<:Real, D<:ContinuousMultivariateSampleable}
    n = length(d)
    return MomentUncertainty{T, D}(d, γ1, γ2, fill(typemin(T), n), fill(typemax(T), n))
end

# Default outer constructors
function MomentUncertainty(
    d::D, γ1::T, γ2::T
) where {T<:Real, D<:ContinuousMultivariateSampleable}
    MomentUncertainty{T, D}(d, γ1, γ2)
end

function MomentUncertainty(
    d::D, γ1::T, γ2::T, lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real}
) where {T<:Real, D<:ContinuousMultivariateSampleable}
    MomentUncertainty{T, D}(d, γ1, γ2, lower, upper)
end

function MomentUncertainty(d::Sampleable; γ1, γ2, lower=nothing, upper=nothing)
    γ1, γ2 = promote(float(γ1), float(γ2))
    T = typeof(γ1)
    n = length(d)
    lower = isnothing(lower) ? fill(typemin(T), n) : lower
    upper = isnothing(upper) ? fill(typemax(T), n) : upper
    return MomentUncertainty(d, γ1, γ2, lower, upper)
end

distribution(s::MomentUncertainty) = s.d

# Worst case, over the distributions in `s`, of the expected value of the concave piecewise linear
# function min_k (a_k r'w + b_k) of the returns `r` (the intercepts `b` can be JuMP expressions).
# Adds the SDP dual of Delage & Ye (2010) to the model of `w`, with one nonnegative multiplier per
# finite bound of the support (Yang et al. 2014), and returns its value.
function _worst_case_piecewise!(s::MomentUncertainty, w, a::AbstractVector, b::AbstractVector)
    @assert length(a) == length(b)
    model = _owner_model(w)

    # parameters
    means = Vector(Distributions.mean(s.d))
    Σ = Matrix(Distributions.cov(s.d))
    n = length(s)
    γ1 = s.γ1
    γ2 = s.γ2
    bounded_below = findall(isfinite, s.lower)
    bounded_above = findall(isfinite, s.upper)

    # dual variables
    P = @variable(model, [i=1:n, j=1:n])
    p = @variable(model, [i=1:n])
    t = @variable(model)
    Q = @variable(model, [i=1:n, j=1:n])
    q = @variable(model, [i=1:n])
    r = @variable(model)

    # constraints: from duality
    @constraint(model, p .== -q / 2 - Q * means)
    @constraint(model, [[P p]; [p' t]] in PSDCone())
    for k in eachindex(a)
        # r'Q r + r'(q + a_k w) + r + b_k >= 0 for every return r in the support
        linear = q + a[k] * w
        constant = r + b[k]
        if !isempty(bounded_below)
            λ = @variable(model, [eachindex(bounded_below)], lower_bound = 0)
            linear[bounded_below] -= λ
            constant += dot(λ, s.lower[bounded_below])
        end
        if !isempty(bounded_above)
            λ = @variable(model, [eachindex(bounded_above)], lower_bound = 0)
            linear[bounded_above] += λ
            constant -= dot(λ, s.upper[bounded_above])
        end
        @constraint(model, [[Q (linear / 2)]; [(linear' / 2) constant]] in PSDCone())
    end

    return -(γ2 * dot(Σ, Q) - first(means'Q * means) + r + dot(Σ, P) - 2 * dot(means, p) + γ1 * t)
end

"""
    calculate_measure!(m::ExpectedUtility{U,S}, w) where {U<:PieceWiseUtility,S<:MomentUncertainty}

Returns the worst case expected utility under distribution uncertainty defined by the
[`MomentUncertainty`](@ref) ambiguity set (Delage & Ye, 2010).

Arguments:
 - `w`: portfolio optimization investment variable ("weights").
 - `m::ExpectedUtility`: Struct containing information about the utility and ambiguity set.
"""
function calculate_measure!(m::ExpectedUtility{U,S}, w) where {U<:PieceWiseUtility,S<:MomentUncertainty}
    u = utility(m)
    return _worst_case_piecewise!(ambiguityset(m), w, coefficients(u), intercepts(u))
end

"""
    calculate_measure!(m::ExpectedReturn{S}, w) where {S<:MomentUncertainty}

Returns the worst case expected return over the [`MomentUncertainty`](@ref) ambiguity set. With
unbounded support it is ``\\hat{r}'w - \\sqrt{\\gamma_1 w' \\Sigma w}``.
"""
function calculate_measure!(m::ExpectedReturn{S}, w) where {S<:MomentUncertainty}
    return _worst_case_piecewise!(ambiguityset(m), w, [1.0], [0.0])
end

"""
    calculate_measure!(m::ConditionalExpectedReturn{α,S}, w) where {α,S<:MomentUncertainty}

Returns the worst case, over the [`MomentUncertainty`](@ref) ambiguity set, of the expected return
of the portfolio over its worst `α` fraction of outcomes, i.e. minus the worst case CVaR
(Yang et al., 2014):

```math
\\max_{z} \\quad z - \\frac{1}{\\alpha} \\sup_{P} \\mathbb{E}_P [ (z - r'w)^+ ] \\\\
```

where the inner supremum is the Delage & Ye (2010) SDP dual of a piecewise linear function.
`α` has to be in `(0, 1]` (`α = 1` gives the worst case expected return). With unbounded support,
`γ1 = 0` and `γ2 = 1` it equals ``\\hat{r}'w - \\sqrt{(1 - \\alpha) / \\alpha} \\sqrt{w' \\Sigma w}``.
"""
function calculate_measure!(m::ConditionalExpectedReturn{α,S}, w) where {α,S<:MomentUncertainty}
    0 < α <= 1 || throw(ArgumentError(
        "α must be in (0, 1] for MomentUncertainty (the worst case return is unbounded), got $(α)"
    ))
    s = ambiguityset(m)
    isone(α) && return _worst_case_piecewise!(s, w, [1.0], [0.0])
    model = _owner_model(w)
    z = @variable(model)
    # min(0, r'w - z) = -(z - r'w)^+
    worst_case_shortfall = _worst_case_piecewise!(s, w, [0.0, 1.0], [zero(AffExpr), -z])
    return z + worst_case_shortfall / α
end

"""
    primal_cone(::Val{p}) -> Type{<:MOI.AbstractVectorSet}

Cone `{(t, x) : t ≥ ||x||_p}` of the `p` norm used by [`DuWassersteinBall`](@ref). Implemented for
`p = 1.0`, `2.0` and `Inf`; add a method for `Val{p}` (with `p` a `Float64`) to support another norm.
"""
primal_cone(::Val{Inf}) = MOI.NormInfinityCone
primal_cone(::Val{1.0}) = MOI.NormOneCone
primal_cone(::Val{2.0}) = MOI.SecondOrderCone
primal_cone(p::Real) = primal_cone(Val(Float64(p)))

_has_primal_cone(p::Real) = applicable(primal_cone, Val(Float64(p)))

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
- `d::DeterministicSamples`: Samples from the parent distribution
- `ϵ::Float64`: Wasserstein distance from sampled distribution (has to be greater than 0). (default: 0.01)
- `Λ::Float64`: Uncertainty around sampled values (has to be greater than 0). (default: maximum(d))

References:
- NingNing paper on Wasserstein DRO (Corollary 1-3): https://ieeexplore.ieee.org/abstract/document/9311154
"""
struct DuWassersteinBall{T<:Real, D<:DeterministicSamples} <: CenteredAmbiguitySet{T,D}
    d::D
    ϵ::T
    Λ::T
    Q::Array{T,2}
    norm_cone::Real

    # Inner constructor for validating arguments
    function DuWassersteinBall{T, D}(
        d::D, ϵ::T, Λ::T, Q::Array{T,2}, norm_cone::Real
    ) where {T<:Real, D<:DeterministicSamples}
        length(d) == size(Q,1) == size(Q,2) || throw(ArgumentError(
            "Distribution ($(length(d))) and Q ($(size(Q,2))) must have coherent dimensions (m and mxm)"
        ))
        ϵ >= 0 || throw(ArgumentError("ϵ must be >= 0"))
        Λ >= 0 || throw(ArgumentError("Λ must be >= 0"))
        _has_primal_cone(norm_cone) || throw(ArgumentError(
            "no cone for the norm $(norm_cone): implement `PortfolioOpt.primal_cone(::Val{$(Float64(norm_cone))})`"
        ))
        return new{T, D}(d, ϵ, Λ, Q, norm_cone)
    end
end

# Default outer constructor
function DuWassersteinBall(
    d::D, ϵ::T, Λ::T, Q::Array{T,2}, norm_cone::Real
) where {T<:Real, D<:DeterministicSamples}
    DuWassersteinBall{T, D}(d, ϵ, Λ, Q, norm_cone)
end

# Kwarg constructor with defaults
function DuWassersteinBall(
    d::S;
    ϵ=0.01,
    norm_cone=Inf,
    Λ=default_DuWassersteinBall_lambda(d, norm_cone),
    Q=Matrix(I(length(d))* 1.0)
) where {S<:DeterministicSamples}
    return DuWassersteinBall(d, ϵ, Λ, Q, norm_cone)
end

distribution(s::DuWassersteinBall) = s.d
sample_size(s::DuWassersteinBall) = sample_size(distribution(s))

# samples are stored column-wise, so the support bound is the largest norm of a sample (column)
default_DuWassersteinBall_lambda(d::Sampleable, norm_cone::Real; num_samples::Int=20, rng::AbstractRNG=MersenneTwister(123)) = maximum(
    norm.(eachcol(rand(rng, d, num_samples)), norm_cone)
)
# the support bound has to hold for every sample, otherwise the worst case problem is unbounded
default_DuWassersteinBall_lambda(d::DeterministicSamples, norm_cone::Real) = maximum(
    norm.(eachcol(rand(d)), norm_cone)
)

"""
    calculate_measure!(measure::ExpectedReturn{S}, w) where {S<:DuWassersteinBall}

"""
function calculate_measure!(measure::ExpectedReturn{S}, w) where {S<:DuWassersteinBall}
    model = _owner_model(w)
    ambiguity_set = ambiguityset(measure)

    # parameters
    N = sample_size(ambiguity_set)
    ξ = rand(distribution(ambiguity_set))

    m = length(ambiguity_set)

    ϵ = ambiguity_set.ϵ
    Λ = ambiguity_set.Λ
    Q = ambiguity_set.Q
    Q_inv = pinv(Q)
    K = primal_cone(ambiguity_set.norm_cone)

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
