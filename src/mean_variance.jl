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

Return predicted portfolio return used in the formulation.

Arguments:
 - `model::JuMP.Model`: JuMP upper level portfolio optimization model.
 - `w`: Portfolio optimization investment variable ("weights").
"""
function predicted_portfolio_return!(::JuMP.Model, w, formulation::AbstractMeanVariance)
    return sum(formulation.predicted_mean'w)
end

"""
    portfolio_return!(model::JuMP.Model, w, formulation::AbstractMeanVariance; kwargs...)

Return worst case portfolio return of the Uncertainty set defined by the formulation.

Arguments:
 - `model::JuMP.Model`: JuMP upper level portfolio optimization model.
 - `w`: Portfolio optimization investment variable ("weights").
"""
function portfolio_return!(model::JuMP.Model, w, formulation::AbstractMeanVariance; kwargs...)
    return predicted_portfolio_return!(model, w, formulation; kwargs...)
end

"""
    predicted_portfolio_variance!(model::JuMP.Model, w, formulation::AbstractMeanVariance)

Return predicted portfolio variance used in the formulation: 

Arguments:
 - `model::JuMP.Model`: JuMP upper level portfolio optimization model.
 - `w`: Portfolio optimization investment variable ("weights").
"""
function predicted_portfolio_variance!(::JuMP.Model, w, formulation::AbstractMeanVariance)
    return sum(w'formulation.predicted_covariance * w)
end

"""
    portfolio_variance!(model::JuMP.Model, w, formulation::AbstractMeanVariance; kwargs...)

Return worst case portfolio variance of the Uncertainty set defined by the formulation.

Arguments:
 - `model::JuMP.Model`: JuMP upper level portfolio optimization model.
 - `w`: Portfolio optimization investment variable ("weights").
"""
function portfolio_variance!(model::JuMP.Model, w, formulation::AbstractMeanVariance; kwargs...)
    return predicted_portfolio_variance!(model, w, formulation; kwargs...)
end

function _po_min_variance_limit_return_latex()
    return """
        ```math
        \\begin{aligned}
        \\min_{w} \\quad & V \\\\
        s.t. \\quad & R = (\\min r'w \\; | \\; r \\in \\Omega) \\\\
        & V = (\\max w ' \\Sigma w  \\; | \\; \\Sigma \\in \\Omega) \\\\
        & R \\geq R * W_0 \\\\
        & w \\in \\mathcal{X} \\\\
        \\end{aligned}
        ```
        """
end

"""
    po_min_variance_limit_return(formulation::AbstractPortfolioFormulation, R)

Mean-Variance Portfolio Allocation (with a risk free asset). Posed as a quadratic convex problem.
Minimizes the worst case portfolio variance (``V``) and limits the worst case portfolio return (``R``) in the uncertainty set (``\\Omega``)
to a minimal return parameter (``R_0``) normalized by current wealth (``W_0``).

$(_po_min_variance_limit_return_latex())

Where ``\\mathcal{X}`` represents the additional constraints defined in the model by the user 
(e.g. a limit on maximum invested money).

Arguments:
 - `formulation::AbstractPortfolioFormulation`: Struct containing attributes of the formulation.
 - `minimal_return::Real`: Minimal normalized return accepted.

Optional Keywork Arguments:
 - `rf::Real = 0.0`: Risk-free return of money not invested.
 - `current_wealth::Real = 1.0`: Current available wealth to be invested.
 - `model::JuMP.Model = base_model(...)`: JuMP upper level portfolio optimization model. Defaults to a generic market specification.
 - `w`: Portfolio optimization investment variable reference ("weights").
 - `portfolio_return::Function = portfolio_return!`: Function that calculates ``R``. 
 - `portfolio_variance::Function = portfolio_variance!`: Function that calculates ``V``. 
"""
function po_min_variance_limit_return(formulation::AbstractPortfolioFormulation, minimal_return::Real;
    rf::Real = 0.0, current_wealth::Real = 1.0,
    model::JuMP.Model = base_model(formulation.number_of_assets; current_wealth=current_wealth), 
    w=model[:w],
    portfolio_return::Function = portfolio_return!,
    portfolio_variance::Function = portfolio_variance!
)
    # auxilary variables
    @variable(model, R)
    sum_invested = create_sum_invested_variable(model, w)

    # model
    @constraint(model, R == portfolio_return(model, w, formulation) + rf * (current_wealth - sum_invested))
    @constraint(model, R >= minimal_return * current_wealth)
    @objective(model, Min, portfolio_variance(model, w, formulation))
    return model
end

function _po_max_return_limit_variance_latex()
    return """
        ```math
        \\begin{aligned}
        \\max_{w} \\quad & R \\\\
        s.t. \\quad & R = (\\min r'w \\; | \\; r \\in \\Omega) \\\\
        & V \\leq V_0 * W_0\\\\
        & V = (\\max w ' \\Sigma w  \\; | \\; \\Sigma \\in \\Omega) \\\\
        & w \\in \\mathcal{X} \\\\
        \\end{aligned}
        ```
        """
end

"""
    po_max_return_limit_variance(formulation::AbstractPortfolioFormulation, V_0)

Mean-Variance Portfolio Allocation (with a risk free asset). Posed as a quadratic convex problem.
Maximizes the worst case portfolio return (``R``) and limits the worst case portfolio variance (``V``) in the uncertainty set (``\\Omega``) 
to a maximal risk parameter (``V_0``) normalized by current wealth (``W_0``).

$(_po_max_return_limit_variance_latex())

Where ``\\mathcal{X}`` represents the additional constraints defined in the model by the user 
(e.g. a limit on maximum invested money).

Arguments:
 - `formulation::AbstractPortfolioFormulation`: Struct containing attributes of formulation.
 - `max_risk::Real`: Maximal normalized variance accepted.

Optional Keywork Arguments:
 - `rf::Real = 0.0`: Risk-free return of money not invested.
 - `current_wealth::Real = 1.0`: Current available wealth to be invested.
 - `model::JuMP.Model = base_model(...)`: JuMP upper level portfolio optimization model. Defaults to a generic market specification.
 - `w`: Portfolio optimization investment variable reference ("weights").
 - `portfolio_return::Function = portfolio_return!`: Function that calculates ``R``. 
 - `portfolio_variance::Function = portfolio_variance!`: Function that calculates ``V``. 
"""
function po_max_return_limit_variance(formulation::AbstractPortfolioFormulation, max_risk::Real;
    rf::Real = 0.0, current_wealth::Real = 1.0,
    model::JuMP.Model = base_model(formulation.number_of_assets; current_wealth=current_wealth), 
    w=model[:w],
    portfolio_return::Function = portfolio_return!,
    portfolio_variance::Function = portfolio_variance!
)
    # auxilary variables
    @variable(model, R)
    sum_invested = create_sum_invested_variable(model, w)

    # model
    @constraint(model, R == portfolio_return(model, w, formulation) + rf * (current_wealth - sum_invested))
    @constraint(model, portfolio_variance(model, w, formulation) <= max_risk * current_wealth)
    @objective(model, Max, R)
    return model
end

###################### Not updated ######################
"""
Mean-Variance Portfolio Allocation With no risk free asset. Analytical solution.
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
