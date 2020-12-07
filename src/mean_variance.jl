"""
    MeanVariance <: AbstractMeanVariance

Mean Variance Formulation (Markowitz). 

PS.: Equivalent to a single point uncertainty set.

Atributes:
- `predicted_mean::Array{Float64,1}` (latex notation \\hat{r}): Predicted mean of returns.
- `predicted_covariance::Array{Float64,2}`: Predicted covariance of returns (formulation atribute).
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
        \\min_{w} \\quad \\text{WCV} \\\\
        s.t. \\quad \\text{WCR} = (\\min r'w | r \\in \\text{UncertaintySet}) \\\\
        \\quad \\quad \\text{WCV} = (\\max w ' \\Sigma w  | \\Sigma \\in \\text{UncertaintySet}) \\\\
        \\quad \\quad \\text{WCR} \\geq R * \\text{current_wealth} \\\\
        ```
        """
end

"""
    po_min_variance_limit_return!(model::JuMP.Model, w, formulation::AbstractPortfolioFormulation, R)

Mean-Variance Portfolio Alocation With risk free asset. Quadratic problem.
Minimize limit worst case variance and limit worst case return in uncertainty set.

$(_po_min_variance_limit_return_latex())
"""
function po_min_variance_limit_return!(model::JuMP.Model, w, formulation::AbstractPortfolioFormulation, R; 
    rf = 0, current_wealth = 1,
    portfolio_return = portfolio_return!,
    portfolio_variance = portfolio_variance!
)
    # auxilary variables
    @variable(model, E)
    if !haskey(object_dictionary(model), :sum_invested)
        @variable(model, sum_invested)
        @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    else
        sum_invested = model[:sum_invested]
    end
    # model
    @constraint(model, E == portfolio_return(model, w, formulation) + rf * (current_wealth - sum_invested))
    @constraint(model, E >= R * current_wealth)
    @objective(model, Min, portfolio_variance(model, w, formulation))
    return nothing
end

function _po_max_return_limit_variance_latex()
    return """
        ```math
        \\max_{w} \\quad  \\text{WCR} \\\\
        s.t.  \\quad \\text{WCR} = (\\min r'w | r \\in \\text{UncertaintySet}) \\\\
        \\quad \\quad \\text{WCV} \\leq \\text{max_risk} * \\text{current_wealth}\\\\
        \\quad \\quad \\text{WCV} = (\\max w ' \\Sigma w  | \\Sigma \\in \\text{UncertaintySet}) \\\\
        ```
        """
end

"""
    po_max_return_limit_variance!(model::JuMP.Model, w, formulation::AbstractPortfolioFormulation, max_risk)

Mean-Variance Portfolio Alocation With risk free asset. Quadratic problem.
Maximize worst case return in uncertainty set and limit worst case variance.

$(_po_max_return_limit_variance_latex())
"""
function po_max_return_limit_variance!(model::JuMP.Model, w, formulation::AbstractPortfolioFormulation, max_risk; 
    rf = 0, current_wealth = 1,
    portfolio_return = portfolio_return!,
    portfolio_variance = portfolio_variance!
)
    # auxilary variables
    @variable(model, E)
    if !haskey(object_dictionary(model), :sum_invested)
        @variable(model, sum_invested)
        @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    else
        sum_invested = model[:sum_invested]
    end
    # model
    @constraint(model, E == portfolio_return(model, w, formulation) + rf * (current_wealth - sum_invested))
    @constraint(model, portfolio_variance(model, w, formulation) <= max_risk * current_wealth)
    @objective(model, Max, E)
    return nothing
end

###################### Not updated ######################
"""
Mean-Variance Portfolio Alocation With no risk free asset. Analytical solution.
"""
function mean_variance_noRf_analytical(formulation::MeanVariance, R)
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
    mu = (R * C - A) / (C * D - A)
    x = (1 / C) * (invΣ * one) + mu * invΣ * (r̄ - one * (A / C))
    return x
end
