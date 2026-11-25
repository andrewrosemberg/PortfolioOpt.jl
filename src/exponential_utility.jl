"""
    ExponentialUtility(a)

Exponential (constant absolute risk aversion, CARA) utility of the portfolio return `x` with risk
aversion `a > 0`:

```math
u(x) = \\frac{1 - e^{-a x}}{a}
```

It is a positive affine transformation of the classic ``-e^{-a x}``, so the optimal portfolio is
the same, and it tends to `x` (risk neutral) as `a → 0`. Use it in [`ExpectedUtility`](@ref) or
[`CertaintyEquivalent`](@ref). Both are represented with exponential cones (e.g. solved by
Clarabel, SCS, Mosek) and are implemented for:
 - a Gaussian distribution (`MvNormal`), where ``\\mathbb{E}[e^{-a w'r}] = e^{-a \\mu'w + a^2 w' \\Sigma w / 2}``;
 - a Gaussian mixture (`MixtureModel` of `MvNormal`s), the sum of the above weighted by the
   component probabilities (Luxenberg & Boyd, 2022);
 - samples ([`DeterministicSamples`](@ref)).

References:
- Luxenberg, E., & Boyd, S. (2024). Portfolio construction with Gaussian mixture returns and
  exponential utility via convex optimization. Optimization and Engineering, 25, 555-574.
  https://doi.org/10.1007/s11081-023-09814-y
"""
struct ExponentialUtility{T<:Real} <: ConcaveUtilityFunction
    a::T

    function ExponentialUtility(a::T) where {T<:Real}
        a > 0 || throw(ArgumentError("the risk aversion must be > 0, got $(a)"))
        return new{T}(a)
    end
end

"""
    risk_aversion(u::ExponentialUtility)

Risk aversion coefficient `a` of the exponential utility.
"""
risk_aversion(u::ExponentialUtility) = u.a

==(a::ExponentialUtility, b::ExponentialUtility) = risk_aversion(a) == risk_aversion(b)

"""
    CertaintyEquivalent(s::AmbiguitySet, utility::ConcaveUtilityFunction)

Certainty equivalent of the portfolio return `R`: the sure return with the same utility,
``u^{-1}(\\mathbb{E}[u(R)])``. Unlike [`ExpectedUtility`](@ref) it is in units of return. For an
[`ExponentialUtility`](@ref) with risk aversion `a` it is minus the entropic risk measure,

```math
CE(R) = -\\frac{1}{a} \\log \\mathbb{E} [e^{-a R}]
```

which equals ``\\mu'w - \\frac{a}{2} w' \\Sigma w`` for a Gaussian distribution. It is concave in the
portfolio, so maximize it or constrain it from below (`GreaterThan`).
"""
struct CertaintyEquivalent{C<:ConcaveUtilityFunction,S<:AmbiguitySet} <:
       PortfolioRiskMeasure{S}
    ambiguity_set::S
    utility::C
end

ambiguityset(m::CertaintyEquivalent) = m.ambiguity_set
utility(m::CertaintyEquivalent) = m.utility

const GaussianMixture = MixtureModel{Multivariate,Continuous,<:AbstractMvNormal}
const ExponentialUtilityDistribution = Union{
    AbstractMvNormal,GaussianMixture,DeterministicSamples
}

# epigraph `v ≥ w'Σw` of the portfolio variance
function _variance_epigraph!(model::JuMP.Model, Σ::AbstractMatrix, w)
    sqrt_Σ = collect(PDMat(Matrix(Symmetric(Σ))).chol.U)
    v = @variable(model)
    @constraint(model, [v; 0.5; sqrt_Σ * w] in JuMP.RotatedSecondOrderCone())
    return v
end

# Probabilities `π` and expressions `e` such that E[exp(-a r'w)] ≤ Σ_k π_k exp(e_k), with
# equality at the optimum (the expressions may use auxiliary epigraph variables).
function _exponential_terms!(model::JuMP.Model, d::AbstractMvNormal, w, a::Real)
    v = _variance_epigraph!(model, cov(d), w)
    return [1.0], [-a * dot(mean(d), w) + a^2 / 2 * v]
end

function _exponential_terms!(model::JuMP.Model, d::GaussianMixture, w, a::Real)
    π = probs(d)
    e = [only(_exponential_terms!(model, c, w, a)[2]) for c in components(d)]
    return π, e
end

function _exponential_terms!(model::JuMP.Model, d::DeterministicSamples, w, a::Real)
    samples = rand(d)
    e = [-a * dot(samples[:, s], w) for s in 1:size(samples, 2)]
    return samples_probability(d), e
end

"""
    calculate_measure!(m::ExpectedUtility{<:ExponentialUtility}, w)

Expected [`ExponentialUtility`](@ref) of the portfolio return,
``(1 - \\mathbb{E}[e^{-a w'r}]) / a``, with one exponential cone per Gaussian component or sample.
"""
function calculate_measure!(
    m::ExpectedUtility{U,S}, w
) where {U<:ExponentialUtility,S<:ExponentialUtilityDistribution}
    model = _owner_model(w)
    a = risk_aversion(utility(m))
    π, e = _exponential_terms!(model, ambiguityset(m), w, a)
    t = @variable(model, [eachindex(e)])
    # t_k ≥ exp(e_k)
    @constraint(model, [k in eachindex(e)], [e[k]; 1.0; t[k]] in MOI.ExponentialCone())
    return (1 - dot(π, t)) / a
end

"""
    calculate_measure!(m::CertaintyEquivalent{<:ExponentialUtility}, w)

Certainty equivalent of the [`ExponentialUtility`](@ref), ``-\\log(\\mathbb{E}[e^{-a w'r}]) / a``,
represented as ``\\theta`` such that ``\\sum_k \\pi_k e^{e_k + a \\theta} \\leq 1``.
"""
function calculate_measure!(
    m::CertaintyEquivalent{U,S}, w
) where {U<:ExponentialUtility,S<:ExponentialUtilityDistribution}
    model = _owner_model(w)
    a = risk_aversion(utility(m))
    π, e = _exponential_terms!(model, ambiguityset(m), w, a)
    θ = @variable(model)
    t = @variable(model, [eachindex(e)])
    # t_k ≥ exp(e_k + a θ)
    @constraint(
        model, [k in eachindex(e)], [e[k] + a * θ; 1.0; t[k]] in MOI.ExponentialCone()
    )
    @constraint(model, dot(π, t) <= 1)
    return θ
end
