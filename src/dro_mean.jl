
"""
    MomentUncertainty <: CenteredAmbiguitySet

```math
\\left\\{ r  \\; \\middle| \\begin{array}{ll}
s.t.  \\quad (\\mathbb{E} [r] - \\hat{r}) ' \\Sigma^{-1} (\\mathbb{E} [r] - \\hat{r}) \\leq \\gamma_1 \\\\
\\quad \\quad \\mathbb{E} [ (r - \\hat{r}) ' (r - \\hat{r}) ] \\leq \\gamma_2 \\Sigma \\\\
\\quad \\quad \\underline{\\xi} \\leq \\xi \\leq \\bar{\\xi} \\\\
\\end{array}
\\right\\} \\\\
```

Atributes:
- `d::Sampleable{Multivariate, Continous}`: The parent distribution with an uncertain mean
- `γ1::Float64`: Uniform uncertainty around the mean (has to be greater than 0).
- `γ2::Float64`: Uncertainty around the covariance (has to be greater than 1).
- `ξ̄::Vector{T}`: Suport upper limits
- `ξ̲::Vector{T}`: Suport lower limits


References:
- Delage paper on moment uncertainty (implemented): https://www.researchgate.net/publication/220244490_Distributionally_Robust_Optimization_Under_Moment_Uncertainty_with_Application_to_Data-Driven_Problems
- Li Yang paper on moment uncertainty and CVAR: https://www.hindawi.com/journals/jam/2014/784715/

"""
struct MomentUncertainty{T<:Real, D<:ContinuousMultivariateSampleable} <: CenteredAmbiguitySet{T,D}
    d::D
    γ1::T
    γ2::T
    ξ̲::Vector{T}
    ξ̄::Vector{T}

    # Inner constructor for validating arguments
    function MomentUncertainty{T, D}(
        d::D, γ1::T, γ2::T, ξ̲::Vector{T}, ξ̄::Vector{T}
    ) where {T<:Real, D<:ContinuousMultivariateSampleable}
        length(d) == length(ξ̄) || throw(ArgumentError(
            "Distribution ($(length(d))) and ξ̄ ($(length(ξ̄))) are not the same length"
        ))
        length(d) == length(ξ̲) || throw(ArgumentError(
            "Distribution ($(length(d))) and ξ̄ ($(length(ξ̲))) are not the same length"
        ))
        means = Vector(mean(d))
        all(ξ̄ .>= means) || throw(ArgumentError("ξ̄ must be >= mean(d)"))
        all(ξ̲ .<= means) || throw(ArgumentError("ξ̲ must be <= mean(d)"))

        γ1 >= 0 || throw(ArgumentError("γ1 must be >= 0"))
        γ2 >= 1 || throw(ArgumentError("γ2 must be >= 1"))
        return new{T, D}(d, γ1, γ2, ξ̲, ξ̄)
    end
end

# Default outer constructor
function MomentUncertainty(
    d::D, γ1::T, γ2::T, ξ̲::Vector{T}, ξ̄::Vector{T}
) where {T<:Real, D<:ContinuousMultivariateSampleable}
    MomentUncertainty{T, D}(d, γ1, γ2, ξ̲, ξ̄)
end

# Kwarg constructor with defaults
function MomentUncertainty(
    d::ContinuousMultivariateSampleable;
    γ1=0.1, γ2=3.0, 
    ξ̲=(Vector(mean(d)) .- sqrt.(var(d))), ξ̄=(Vector(mean(d)) .+ sqrt.(var(d)))
)
    return MomentUncertainty(d, γ1, γ2, ξ̲, ξ̄)
end


distribution(s::MomentUncertainty) = s.d

"""
    calculate_measure!(measure::ExpectedReturn{MomentUncertainty,WorstCase}, w)

Returns worst case utility return (WCR) under distribution uncertainty defined by MomentUncertainty ambiguity set ([`MomentUncertainty`](@ref)).

Arguments:
 - `w`: portfolio optimization investment variable ("weights").
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
    P = @variable(model, [i=1:n, j=1:n])
    p = @variable(model, [i=1:n])
    s = @variable(model)
    Q = @variable(model, [i=1:n, j=1:n])
    q = @variable(model, [i=1:n])
    r = @variable(model)

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
    calculate_measure!(measure::ExpectedReturn{MomentUncertainty,WorstCase}, w)

Returns worst case return (WCR) under distribution uncertainty defined by MomentUncertainty ambiguity set ([`MomentUncertainty`](@ref)).

Arguments:
 - `w`: portfolio optimization investment variable ("weights").
"""
function calculate_measure!(m::ConditionalExpectedReturn{1.0,T,S,R}, w) where {S<:MomentUncertainty,T,R}
    model = owner_model(w)
    s = ambiguityset(m)
    
    # parameters
    means = Vector(Distributions.mean(s.d))
    Σ = Matrix(Distributions.cov(s.d))
    n = length(s)
    γ1 = s.γ1
    γ2 = s.γ2
    ξ̄ = s.ξ̄
    ξ̲ = s.ξ̲

    # dual variables
    @variable(model, P1[i=1:n, j=1:n])
    @variable(model, p1[i=1:n])
    @variable(model, s1)
    @variable(model, Q1[i=1:n, j=1:n])
    @variable(model, q1[i=1:n])
    @variable(model, r1)

    @variable(model, r3 >= 0)
    @variable(model, λ1[i=1:n] >= 0)
    @variable(model, λ2[i=1:n] >= 0)

    # constraints: from duality
    @constraint(model, p1 .== -q1 / 2 - Q1 * means)
    @constraint(model, [[P1 p1]; [p1' s1]] in PSDCone())

    @constraint(
        model, [[(r1 + dot(λ1, ξ̲) - dot(λ2, ξ̄) - r3) ((q1 + w + λ2 - λ1)' / 2)]; [((q1 + w + λ2 - λ1) / 2) Q1] ] in PSDCone()
    )

    return -(γ2 * dot(Σ, Q1) - first(means'Q1 * means) + r1 + dot(Σ, P1) - 2 * dot(means, p1) + γ1 * s1)

end

"""
    calculate_measure!(measure::ExpectedReturn{MomentUncertainty,WorstCase}, w)

Returns worst case CVAR under distribution uncertainty defined by MomentUncertainty ambiguity set ([`MomentUncertainty`](@ref)).

Arguments:
 - `w`: portfolio optimization investment variable ("weights").
"""
function calculate_measure!(m::ConditionalExpectedReturn{β,T,S,R}, w) where {S<:MomentUncertainty,β,T,R}
    model = owner_model(w)
    s = ambiguityset(m)
    
    # parameters
    means = Vector(Distributions.mean(s.d))
    Σ = Matrix(Distributions.cov(s.d))
    n = length(s)
    γ1 = s.γ1
    γ2 = s.γ2
    ξ̄ = s.ξ̄
    ξ̲ = s.ξ̲

    # dual variables
    @variable(model, P2[i=1:n, j=1:n])
    @variable(model, p2[i=1:n])
    @variable(model, s2)
    @variable(model, Q2[i=1:n, j=1:n])
    @variable(model, q2[i=1:n])
    @variable(model, r2)

    @variable(model, r4)
    @variable(model, r5)
    @variable(model, α)
    @variable(model, λ3[i=1:n] >= 0)
    @variable(model, λ4[i=1:n] >= 0)
    @variable(model, λ5[i=1:n] >= 0)
    @variable(model, λ6[i=1:n] >= 0)

    # constraints: from duality
    @constraint(model, p2 .== -q2 / 2 - Q2 * means)
    @constraint(model, [[P2 p2]; [p2' s2]] in PSDCone())

    @constraint(
        model, [[(r2 + dot(λ3, ξ̲) - dot(λ4, ξ̄) - r4) ((q2 + λ4 - λ3)' / 2)]; [((q2 + λ4 - λ3) / 2) Q2]] in PSDCone()
    )

    @constraint(
        model, [[(r2 + dot(λ5, ξ̲) - dot(λ6, ξ̄) - r5) ((q2 + (w ./ (1 - β)) + λ6 - λ5)' / 2)]; [((q2 + (w ./ (1 - β)) + λ6 - λ5) / 2) Q2]] in PSDCone()
    )

    @constraint(model, r4 >= α)

    @constraint(model, r5 >= (1 - (1/(1 - β))) *  α)

    return γ2 * dot(Σ, Q2) - first(means'Q2 * means) + r2 + dot(Σ, P2) - 2 * dot(means, p2) + γ1 * s2
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
    calculate_measure!(measure::ConditionalExpectedReturn{1.0,T,S,R}, w) where {S<:DuWassersteinBall,T,R}

Returns worst case return (WCR) under distribution uncertainty defined by DuWassersteinBall.
"""
function calculate_measure!(measure::ConditionalExpectedReturn{1.0,T,S,R}, w) where {S<:DuWassersteinBall,T,R}
    model = owner_model(w)
    ambiguity_set = ambiguityset(measure)

    # parameters
    N = sample_size(measure)
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
