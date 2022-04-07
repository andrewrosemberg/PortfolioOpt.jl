"""
    BudgetSet <: DROSet

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
struct BudgetSet{T<:Real, D<:ContinuousMultivariateSampleable} <: DROSet
    d::D
    Δ::Vector{T}
    Γ::T
    # Inner constructor for validating arguments
    function BudgetSet{T, D}(d::D, Δ::Vector{T}, Γ::T) where {T<:Real, D<:ContinuousMultivariateSampleable}
        length(d) == length(Δ) || throw(ArgumentError(
            "Distribution ($(length(d))) and Δ ($(length(Δ))) are not the same length"
        ))
        all(>=(0), Δ) || throw(ArgumentError("All uncertainty deltas must be >= 0"))
        Γ >= 0 || throw(ArgumentError("Budget must be >= 0"))
        Γ <= length(d) || @warn "Budget should not exceed the distribution length"
        return new{T, D}(d, Δ, Γ)
    end
end

# Default outer constructor
BudgetSet(d::D, Δ::Vector{T}, Γ::T) where {T<:Real, D<:ContinuousMultivariateSampleable} = BudgetSet{T, D}(d, Δ, Γ)

function _portfolio_return_latex_RobustBertsimas_primal()
    return """
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
        """
end

function _portfolio_return_latex_RobustBertsimas_dual()
    return """
        ```math
        \\begin{aligned}
        \\max_{\\lambda, \\pi^-, \\pi^+, \\theta} \\quad  \\sum_{i}^{\\mathcal{N}} (\\hat{r}_i (\\pi^+_i - \\pi^-_i) - \\theta_i ) - \\Gamma \\lambda\\\\
        s.t. \\quad &  w_i = \\pi^+_i - \\pi^-_i  \\quad \\forall i = 1:\\mathcal{N} \\\\
        &  \\Delta_i (\\pi^+_i + \\pi^-_i) - \\theta_i \\leq \\lambda \\quad \\forall i = 1:\\mathcal{N} \\\\
        & \\lambda \\geq 0 , \\; \\pi^- \\geq 0 , \\; \\pi^+ \\geq 0 , \\; \\theta \\geq 0 \\\\
        \\end{aligned}
        ```
        """
end

"""
    portfolio_return!(model::JuMP.Model, w, formulation::RobustBertsimas)

Returns worst case return (WCR) in Bertsimas's uncertainty set ([`RobustBertsimas`](@ref)).

WCR is defined by the following primal problem: 

$(_portfolio_return_latex_RobustBertsimas_primal())

Which is equivalent to the following dual problem:

$(_portfolio_return_latex_RobustBertsimas_dual())

To avoid solving an optimization problem we enforece the dual constraints in 
the upper level problem and return the objective expression (a lower bound of the optimum).

Arguments:
 - `model::JuMP.Model`: JuMP upper level portfolio optimization model.
 - `w`: portfolio optimization investment variable ("weights").
 - `formulation::RobustBertsimas`: Struct containing atributes of Bertsimas's uncertainty set.
"""
function portfolio_return!(model::JuMP.Model, w, formulation::RobustBertsimas)
    # parameters
    r̄ = formulation.predicted_mean
    numA = formulation.number_of_assets
    Δ = formulation.uncertainty_delta
    Λ = formulation.bertsimas_budget
    # dual variables
    @variable(model, λ >= 0)
    @variable(model, π_neg[i=1:numA] >= 0)
    @variable(model, π_pos[i=1:numA] >= 0)
    @variable(model, θ[i=1:numA] >= 0)
    # constraints: from duality
    @constraints(
        model,
        begin
            constrain_dual1[i=1:numA], w[i] == π_pos[i] - π_neg[i]
        end
    )
    @constraints(
        model,
        begin
            constrain_dual2[i=1:numA], Δ[i] * (π_pos[i] + π_neg[i]) - θ[i] <= λ
        end
    )

    return sum(r̄[i] * (π_pos[i] - π_neg[i]) for i in 1:numA) - sum(θ[i] for i in 1:numA) - λ * Λ
end

##################################
function _BenTal_latex()
    return """
        ```math
        \\left\\{ \\mu \\; \\middle| \\begin{array}{ll}
        s.t.  \\quad \\sqrt{(\\hat{r} - \\mu) ' \\Sigma^{-1} (\\hat{r} - \\mu)} \\leq \\delta \\\\
        \\end{array}
        \\right\\} \\\\
        ```
        """
end
"""
    BenTal <: AbstractMeanVariance

BenTal's uncertainty set:

$(_BenTal_latex())

Further information:
  - Ben-Tal, A. e Nemirovski, A. (2000). Robust solutions of linear programming problems contaminated with uncertain data. Mathematical programming, 88(3):411–424.

Atributes:
- `predicted_mean::Array{Float64,1}` (latex notation ``\\hat{r}``): Predicted mean of returns.
- `uncertainty_delta::Float64` (latex notation ``\\delta``): Uncertainty around mean.
- `predicted_covariance::Array{Float64,2}` (latex notation ``\\Sigma``): Predicted covariance of returns.
"""
struct RobustBenTal <: AbstractMeanVariance
    predicted_mean::Array{Float64,1}
    predicted_covariance::Array{Float64,2}
    uncertainty_delta::Float64
    number_of_assets::Int
end

function RobustBenTal(;
    predicted_mean::Array{Float64,1},
    predicted_covariance::Array{Float64,2},
    uncertainty_delta::Float64,
)
    number_of_assets = size(predicted_mean, 1)
    if number_of_assets != size(predicted_covariance, 1)
        error("size of predicted mean ($number_of_assets) different to size of predicted covariance ($(size(predicted_covariance, 1)))")
    end

    @assert uncertainty_delta >= 0
    @assert issymmetric(predicted_covariance)
    return RobustBenTal(
        predicted_mean, predicted_covariance, uncertainty_delta, number_of_assets
    )
end

function _portfolio_return_latex_RobustBenTal_primal()
    return """
        ```math
        \\begin{aligned}
        \\min_{\\mu} \\quad & \\mu ' w \\\\
        s.t. \\quad & ||Σ^{-\\frac{1}{2}} (\\mu - \\hat{r}) || \\leq \\delta \\quad &: \\theta \\\\
        \\end{aligned}
        ```
        """
end

function _portfolio_return_latex_RobustBenTal_dual()
    return """
        ```math
        \\begin{aligned}
        \\max_{\\theta} \\quad &  w ' \\hat{r} - \\theta \\delta \\\\
        s.t. \\quad & ||Σ^{\\frac{1}{2}}  w || \\leq \\theta \\\\
        \\end{aligned}
        ```
        """
end

"""
    portfolio_return!(model::JuMP.Model, w, formulation::RobustBenTal)

Returns worst case return (WCR) in BenTal's uncertainty set ([`RobustBenTal`](@ref)).

WCR is defined by the following primal problem: 

$(_portfolio_return_latex_RobustBenTal_primal())

Which is equivalent to the following dual problem:

$(_portfolio_return_latex_RobustBenTal_dual())

To avoid solving an optimization problem we enforece the dual constraints in 
the upper level problem and return the objective expression (a lower bound of the optimum).

Arguments:
    - `model::JuMP.Model`: JuMP upper level portfolio optimization model.
    - `w`: portfolio optimization investment variable ("weights").
    - `formulation::RobustBenTal`: Struct containing atributes of BenTal's uncertainty set.
"""
function portfolio_return!(model::JuMP.Model, w, formulation::RobustBenTal)
    # parameters
    r̄ = formulation.predicted_mean
    numA = formulation.number_of_assets
    δ = formulation.uncertainty_delta
    sqrt_Σ = sqrt(formulation.predicted_covariance)
    # dual variables
    @variable(model, θ)
    # constraints: from duality
    @constraint(model, [θ; sqrt_Σ * w] in JuMP.SecondOrderCone())

    return dot(w, r̄) - θ * δ
end
