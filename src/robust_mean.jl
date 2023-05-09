Distributions.rand(rng::AbstractRNG, d::Dirac, n::Int) = hcat(fill(d.value, n)...)

"""
    BudgetSet{T<:Real, D<:ContinuousMultivariateSampleable} <: CenteredAmbiguitySet{T,D}

Bertsimas's uncertainty set:

```math
\\left\\{ \\mu \\; \\middle| \\begin{array}{ll}
s.t.  \\quad \\mu_i \\leq \\hat{r}_i + z_i \\Delta_i \\quad \\forall i = 1:\\mathcal{N} \\\\
\\quad \\quad \\mu_i \\geq \\hat{r}_i - z_i \\Delta_i  \\quad \\forall i = 1:\\mathcal{N} \\\\
\\quad \\quad z_i \\geq 0 \\quad \\forall i = 1:\\mathcal{N} \\\\
\\quad \\quad z_i \\leq 1 \\quad \\forall i = 1:\\mathcal{N} \\\\
\\quad \\quad \\sum_{i}^{\\mathcal{N}} z_i \\leq \\Gamma \\\\
\\end{array}
\\right\\} \\\\
```

Attributes:
- `d::Sampleable{Multivariate, Continous}`: The parent distribution with an uncertain mean
- `Δ::Array{Float64,1}`: Uncertainty around mean. (default: std(d) / 5)
- `Γ::Float64`: Number of assets in worst case. (default: 0.1 * length(d))

References:
- Bertsimas, D. e Sim, M. (2004). The price of robustness. Operations research, 52(1):35–53.

"""
struct BudgetSet{T<:Real, D<:Sampleable} <: CenteredAmbiguitySet{T,D}
    d::D
    Δ::Vector{T}
    Γ::T
    # Inner constructor for validating arguments
    function BudgetSet{T, D}(d::D, Δ::Vector{T}, Γ::T) where {T<:Real, D<:Sampleable}
        length(d) == length(Δ) || throw(ArgumentError(
            "Distribution ($(length(d))) and Δ ($(length(Δ))) are not the same length"
        ))
        all(>=(0), Δ) || throw(ArgumentError("All uncertainty deltas must be >= 0"))
        Γ >= 0 || throw(ArgumentError("Budget must be >= 0"))
        Γ <= length(d) || @warn "Budget should not exceed the distribution length"
        return new{T, D}(d, Δ, Γ)
    end
end

distribution(s::BudgetSet) = s.d

# Default outer constructor
BudgetSet(d::D, Δ::Vector{T}, Γ::T) where {T<:Real, D<:Sampleable} = BudgetSet{T, D}(d, Δ, Γ)

# Kwarg constructor with defaults
function BudgetSet(
    d::Sampleable;
    Δ=default_budgetset_delta(d),
    Γ=default_budgetset_budget(d),
)
    return BudgetSet(d, Δ, Γ)
end

# Default values (no partuicular reason for these defaults)
default_budgetset_delta(d::AbstractMvNormal) = sqrt.(var(d)) ./ 5
default_budgetset_budget(d::Sampleable) = length(d) * 1.0

"""
    calculate_measure!(measure::ExpectedReturn{BudgetSet,WorstCase}, w)

Returns worst case return (WCR) in BudgetSet's uncertainty set ([`BudgetSet`](@ref)).

For further reading:
- BudgetSet' paper: BudgetSet, D. e Sim, M. (2004). The price of robustness. Operations research, 52(1):35–53.
- Original implementation from: https://github.com/andrewrosemberg/PortfolioOpt.jl.

WCR is defined by the following primal problem: 

```math
\\begin{aligned}
\\min_{\\mu, z} \\quad & \\mu ' w \\\\
s.t. \\quad & \\mu_i \\leq \\hat{r}_i + z_i \\Delta_i \\quad \\forall i = 1:\\mathcal{N} \\quad &: \\pi^-_i \\\\
& \\mu_i \\geq \\hat{r}_i - z_i \\Delta_i  \\quad \\forall i = 1:\\mathcal{N} \\quad &: \\pi^+_i \\\\
& z_i \\geq 0 \\quad \\forall i = 1:\\mathcal{N} \\\\
& z_i \\leq 1 \\quad \\forall i = 1:\\mathcal{N} \\quad &: \\theta_i \\\\
& \\sum_{i}^{\\mathcal{N}} z_i \\leq \\Gamma \\quad : \\lambda \\\\
\\end{aligned}
```

Which is equivalent to the following dual problem:

```math
\\begin{aligned}
\\max_{\\lambda, \\pi^-, \\pi^+, \\theta} \\quad  \\sum_{i}^{\\mathcal{N}} (\\hat{r}_i (\\pi^+_i - \\pi^-_i) - \\theta_i ) - \\Gamma \\lambda\\\\
s.t. \\quad &  w_i = \\pi^+_i - \\pi^-_i  \\quad \\forall i = 1:\\mathcal{N} \\\\
&  \\Delta_i (\\pi^+_i + \\pi^-_i) - \\theta_i \\leq \\lambda \\quad \\forall i = 1:\\mathcal{N} \\\\
& \\lambda \\geq 0 , \\; \\pi^- \\geq 0 , \\; \\pi^+ \\geq 0 , \\; \\theta \\geq 0 \\\\
\\end{aligned}
```

To avoid solving an optimization problem we enforece the dual constraints in 
the upper level problem and return the objective expression (a lower bound of the optimum).

Arguments:
 - `w`: portfolio optimization investment variable ("weights").
 - `s::BudgetSet`: Struct containing atributes of BudgetSet's uncertainty set.
"""
function calculate_measure!(measure::ExpectedReturn{S,WorstCase}, w)  where {S<:BudgetSet}
    model = owner_model(w)
    s = ambiguityset(measure)

    # parameters
    means = Vector(Distributions.mean(s.d))
    n = length(s)
    Δ = s.Δ
    Γ = s.Γ
    # dual variables
    @variable(model, λ >= 0)
    @variable(model, π_neg[i=1:n] >= 0)
    @variable(model, π_pos[i=1:n] >= 0)
    @variable(model, θ[i=1:n] >= 0)
    # constraints: from duality
    @constraints(
        model,
        begin
            constrain_dual1[i=1:n], w[i] == π_pos[i] - π_neg[i]
        end
    )
    @constraints(
        model,
        begin
            constrain_dual2[i=1:n], Δ[i] * (π_pos[i] + π_neg[i]) - θ[i] <= λ
        end
    )

    return return dot(means, w) - sum(θ) - λ * Γ
end

"""
    EllipticalSet <: CenteredAmbiguitySet{T,D}

```math
\\left\\{ \\mu \\; \\middle| \\begin{array}{ll}
s.t.  \\quad \\sqrt{(\\hat{r} - \\mu) ' \\Sigma^{-1} (\\hat{r} - \\mu)} \\leq \\delta \\\\
\\end{array}
\\right\\} \\\\
```

Atributes:
- `d::Sampleable{Multivariate, Continous}`: The parent distribution with an uncertain mean
- `Δ::Array{Float64,1}`: Uniform uncertainty around mean. (default: 0.025)

References:
- Ben-Tal, A. e Nemirovski, A. (2000). Robust solutions of linear programming problems contaminated with uncertain data. Mathematical programming, 88(3):411–424.

"""
struct EllipticalSet{T<:Real, D<:ContinuousMultivariateSampleable} <: CenteredAmbiguitySet{T,D}
    d::D
    Δ::T

    # Inner constructor for validating arguments
    function EllipticalSet{T, D}(d::D, Δ::T) where {T<:Real, D<:ContinuousMultivariateSampleable}
        Δ >= 0 || throw(ArgumentError("Uncertainty delta must be >= 0"))
        return new{T, D}(d, Δ)
    end
end

# Default outer constructor
EllipticalSet(d::D, Δ::T) where {T<:Real, D<:ContinuousMultivariateSampleable} = EllipticalSet{T, D}(d, Δ)

EllipticalSet(d::Sampleable; Δ) = EllipticalSet(d, Δ)

distribution(s::EllipticalSet) = s.d

"""
    calculate_measure!(measure::ExpectedReturn{EllipticalSet,WorstCase}, w)

Returns worst case return (WCR) in EllipticalSet's uncertainty set ([`EllipticalSet`](@ref)).

WCR is defined by the following primal problem: 

```math
\\begin{aligned}
\\min_{\\mu} \\quad & \\mu ' w \\\\
s.t. \\quad & ||Σ^{-\\frac{1}{2}} (\\mu - \\hat{r}) || \\leq \\delta \\quad &: \\theta \\\\
\\end{aligned}
```

Which is equivalent to the following dual problem:

```math
\\begin{aligned}
\\max_{\\theta} \\quad &  w ' \\hat{r} - \\theta \\delta \\\\
s.t. \\quad & ||Σ^{\\frac{1}{2}}  w || \\leq \\theta \\\\
\\end{aligned}
```

To avoid solving an optimization problem we enforece the dual constraints in 
the upper level problem and return the objective expression (a lower bound of the optimum).

Arguments:
 - `w`: portfolio optimization investment variable ("weights").
 - `s:: EllipticalSet `: Struct containing atributes of EllipticalSet's uncertainty set.
"""
function calculate_measure!(measure::ExpectedReturn{S,WorstCase}, w)   where {S<:EllipticalSet}
    model = owner_model(w)
    s = ambiguityset(measure)

    # parameters
    means = Vector(Distributions.mean(s.d))
    Σ = PDMat(Symmetric(cov(s.d)))
    sqrt_Σ = collect(Σ.chol.U)

    Δ = s.Δ
    # dual variables
    @variable(model, θ)
    # constraints: from duality
    @constraint(model, [θ; sqrt_Σ * w] in JuMP.SecondOrderCone())

    return dot(w, means) - θ * Δ
end
