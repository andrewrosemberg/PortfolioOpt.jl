function _MomentUncertainty_latex()
    return """
        ```math
        \\left\\{ r  \\; \\middle| \\begin{array}{ll}
        s.t.  \\quad (\\mathbb{E} [r] - \\hat{r}) ' \\Sigma^{-1} (\\mathbb{E} [r] - \\hat{r}) \\leq \\gamma_1 \\\\
        \\quad \\quad \\mathbb{E} [ (r - \\hat{r}) ' (r - \\hat{r}) ] \\leq \\gamma_2 \\Sigma \\\\
        \\end{array}
        \\right\\} \\\\
        ```
        """
end

"""
    MomentUncertainty <: AbstractMeanVariance

Delage's Ambiguity set:

$(_MomentUncertainty_latex())

Further information:
  - Delage paper on moment uncertainty (what I implemented): https://www.researchgate.net/publication/220244490_Distributionally_Robust_Optimization_Under_Moment_Uncertainty_with_Application_to_Data-Driven_Problems

Atributes:
- `predicted_mean::Array{Float64,1}` (latex notation ``\\hat{r}``): Predicted mean of returns.
- `predicted_covariance::Array{Float64,2}` (latex notation ``\\Sigma``): Predicted covariance of returns.
- `γ1::Float64`: Mean uncertainty parameter (has to be greater than 0).
- `γ2::Float64`: Covariance uncertainty parameter (has to be greater than 1).
- `utility_coeficients::Array{Float64,1}`: Piece-wise utility coeficients (default 1).
- `utility_intercepts::Array{Float64,1}`: Piece-wise utility intercepts (default 0).
"""
struct MomentUncertainty{T<:Real, D<:ContinuousMultivariateSampleable} <: DROSet
    d::D
    γ1::T
    γ2::T
end

function MomentUncertainty(;
    d::D
    γ1::T
    γ2::T
) where {T<:Real, D<:ContinuousMultivariateSampleable}
    @assert γ1 >= 0
    @assert γ2 >= 1
    return MomentUncertainty(d, γ1, γ2)
end

"""
    po_max_utility_return(formulation::MomentUncertainty)

Maximize expected utility of portfolio return under distribution uncertainty defined by
Delage's ambiguity set ([`MomentUncertainty`](@ref)).
"""
function po_max_utility_return(formulation::MomentUncertainty;
    current_wealth::Real = 1.0,
    model::JuMP.Model = base_model(formulation.number_of_assets; current_wealth=current_wealth),
    w = model[:w],
)
    # parameters
    r̄ = formulation.predicted_mean
    Σ = formulation.predicted_covariance
    numA = formulation.number_of_assets
    a = formulation.utility_coeficients
    b = formulation.utility_intercepts
    γ1 = formulation.γ1
    γ2 = formulation.γ2
    K = formulation.number_of_utility_pieces
    # dual variables
    @variable(model, P[i=1:numA, j=1:numA])
    @variable(model, p[i=1:numA])
    @variable(model, s)
    @variable(model, Q[i=1:numA, j=1:numA])
    @variable(model, q[i=1:numA])
    @variable(model, r)
    # constraints: from duality
    @constraint(model, p .== -q / 2 - Q * r̄)
    @constraint(model, [[P p]; [p' s]] >= 0, PSDCone())
    for k in 1:K
        @constraint(
            model, [[Q (q / 2 + a[k] * w / 2)]; [(q' / 2 + a[k] * w' / 2) (r + b[k])]] >= 0, PSDCone()
        )
    end

    # objective
    @objective(
        model,
        Min, # Min because this is the dual problem
        γ2 * dot(Σ, Q) - sum(r̄'Q * r̄) + r + dot(Σ, P) - 2 * dot(r̄, p) + γ1 * s
    )

    return model
end
