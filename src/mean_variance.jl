"""
    MeanVariance <: AbstractMeanVariance

Mean Variance Formulation (Markowitz). 

PS.: Equivalent to a single point uncertainty set.

Atributes:
- `predicted_mean::Array{Float64,1}`: Predicted mean of returns.
- `predicted_covariance::Array{Float64,2}`: Predicted covariance of returns.
"""
struct MeanVariance <: AbstractMeanVariance
    predicted_mean::Array{Float64,1}
    predicted_covariance::Array{Float64,2}
    number_of_assets::Int
end

function MeanVariance(;
    predicted_mean::Array{Float64,1},
    predicted_covariance::Array{Float64,2},
)
    number_of_assets = size(predicted_mean, 1)
    if number_of_assets != size(predicted_covariance, 1)
        error("size of predicted mean ($number_of_assets) different to size of predicted covariance ($(size(predicted_covariance, 1)))")
    end

    @assert issymmetric(predicted_covariance)
    return MeanVariance(
        predicted_mean, predicted_covariance, number_of_assets
    )
end

"""
    predicted_portfolio_return!(model::JuMP.Model, w, formulation::AbstractMeanVariance)

Return predicted return used in the formulation.
"""
function predicted_portfolio_return!(::JuMP.Model, w, formulation::AbstractMeanVariance)
    return sum(formulation.predicted_mean'w)
end

"""
    portfolio_return!(model::JuMP.Model, w, formulation::AbstractMeanVariance; kwargs...)

Return worst case return of the Uncertainty set defined by the formulation.
"""
function portfolio_return!(model::JuMP.Model, w, formulation::AbstractMeanVariance; kwargs...)
    return predicted_portfolio_return!(model, w, formulation; kwargs...)
end

"""
    predicted_portfolio_variance!(model::JuMP.Model, w, formulation::AbstractMeanVariance)

Return predicted variance used in the formulation.
"""
function predicted_portfolio_variance!(::JuMP.Model, w, formulation::AbstractMeanVariance)
    return sum(w'formulation.predicted_covariance * w)
end

"""
    portfolio_variance!(model::JuMP.Model, w, formulation::AbstractMeanVariance; kwargs...)

Return worst case variance of the Uncertainty set defined by the formulation.
"""
function portfolio_variance!(model::JuMP.Model, w, formulation::AbstractMeanVariance; kwargs...)
    return predicted_portfolio_variance!(model, w, formulation; kwargs...)
end

function _po_min_variance_limit_return_latex()
    return """
        ```math
        \\begin{aligned}
        \\min_{w} & V \\\\
        s.t. & R = (\\min r'w \\; | \\; r \\in \\Omega) \\\\
        & V = (\\max w ' \\Sigma w  \\; | \\; \\Sigma \\in \\Omega) \\\\
        & R \\geq R * W_0 \\\\
        & w \\in \\mathcal{X} \\\\
        \\end{aligned}
        ```
        """
end

"""
    po_min_variance_limit_return!(model::JuMP.Model, w, formulation::AbstractPortfolioFormulation, R)

Mean-Variance Portfolio Alocation (with a risk free asset). Posed as a quadratic convex problem.
Minimizes worst case portfolio variance (``V``) and limit worst case portfolio return (``R``) in the uncertainty set (``\\Omega``)
to a minimal return parameter (``R_0``) normalized by current wealth (``W_0``).

$(_po_min_variance_limit_return_latex())

Where ``\\mathcal{X}`` represents the additional constraints defined in the model by the user 
(like maximum invested money).
"""
function po_min_variance_limit_return!(model::JuMP.Model, w, formulation::AbstractPortfolioFormulation, R_0; 
    rf = 0, W_0 = 1,
    portfolio_return = portfolio_return!,
    portfolio_variance = portfolio_variance!
)
    # auxilary variables
    @variable(model, R)
    if !haskey(object_dictionary(model), :sum_invested)
        @variable(model, sum_invested)
        @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    else
        sum_invested = model[:sum_invested]
    end
    # model
    @constraint(model, R == portfolio_return(model, w, formulation) + rf * (W_0 - sum_invested))
    @constraint(model, R >= R_0 * W_0)
    @objective(model, Min, portfolio_variance(model, w, formulation))
    return nothing
end

function _po_max_return_limit_variance_latex()
    return """
        ```math
        \\max_{w} \\quad  R \\\\
        s.t.  \\quad R = (\\min r'w \\; | \\; r \\in \\Omega) \\\\
        \\quad \\quad V \\leq MaxRisk * W_0\\\\
        \\quad \\quad V = (\\max w ' \\Sigma w  \\; | \\; \\Sigma \\in \\Omega) \\\\
        \\quad \\quad w \\in \\mathcal{X} \\\\
        ```
        """
end

"""
    po_max_return_limit_variance!(model::JuMP.Model, w, formulation::AbstractPortfolioFormulation, V_0)

Mean-Variance Portfolio Alocation (with a risk free asset). Posed as a quadratic convex problem.
Maximizes worst case portfolio return (``R``) and limit worst case portfolio variance (``V``) in the uncertainty set (``\\Omega``) 
to a minimal risk parameter (``V_0``) normalized by current wealth (``W_0``).

$(_po_max_return_limit_variance_latex())

Where ``\\mathcal{X}`` represents the additional constraints defined in the model by the user 
(like maximum invested money).
"""
function po_max_return_limit_variance!(model::JuMP.Model, w, formulation::AbstractPortfolioFormulation, V_0; 
    rf = 0, W_0 = 1,
    portfolio_return = portfolio_return!,
    portfolio_variance = portfolio_variance!
)
    # auxilary variables
    @variable(model, R)
    if !haskey(object_dictionary(model), :sum_invested)
        @variable(model, sum_invested)
        @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    else
        sum_invested = model[:sum_invested]
    end
    # model
    @constraint(model, R == portfolio_return(model, w, formulation) + rf * (W_0 - sum_invested))
    @constraint(model, portfolio_variance(model, w, formulation) <= V_0 * W_0)
    @objective(model, Max, R)
    return nothing
end

###################### Not updated ######################
"""
Mean-Variance Portfolio Alocation With no risk free asset. Analytical solution.
"""
function mean_variance_noRf_analytical(formulation::MeanVariance, R_0)
    # parameters
    r̄ = formulation.predicted_mean
    Σ = formulation.predicted_covariance
    numA = formulation.number_of_assets
    # model
    invΣ = pinv(Σ, 1E-25)
    x = zeros(numA)
    A = sum(invΣ'r̄)
    B = sum(r̄'r̄)
    C = sum(invΣ)
    one = ones(size(r̄, 1))
    D = sum(r̄' * (invΣ'r̄))
    mu = (R_0 * C - A) / (C * D - A)
    x = (1 / C) * (invΣ * one) + mu * invΣ * (r̄ - one * (A / C))
    return x
end
