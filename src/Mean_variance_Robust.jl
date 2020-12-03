"""Bertsimas's uncertainty set"""
struct RobustBertsimas <: AbstractMeanVariance
    predicted_mean::Array{Float64,1}
    predicted_covariance::Array{Float64,2}
    uncertainty_delta::Array{Float64,1}
    bertsimas_budjet::Float64
    number_of_assets::Int
end

function RobustBertsimas(;
    predicted_mean::Array{Float64,1},
    predicted_covariance::Array{Float64,2},
    uncertainty_delta::Array{Float64,1},
    bertsimas_budjet::Float64,
)
    number_of_assets = size(predicted_mean, 1)
    if number_of_assets != size(predicted_covariance, 1)
        error("size of predicted mean ($number_of_assets) different to size of predicted covariance ($(size(predicted_covariance, 1)))")
    end
    if number_of_assets != size(uncertainty_delta, 1)
        error("size of predicted mean ($number_of_assets) different to size of uncertainty deltas ($(size(uncertainty_delta,1)))")
    end

    @assert prod(uncertainty_delta .>= 0)
    @assert issymmetric(predicted_covariance)
    return RobustBertsimas(
        predicted_mean, predicted_covariance, uncertainty_delta, bertsimas_budjet, number_of_assets
    )
end

function _portfolio_return_latex()
    return """
        ```math
        \\max   \\Gamma \\lambda \\sum_{i}^{mathcal{N}} \\hat{r}_i (\\pi 2_i \\pi 1_i) - \\theta_i \\\\
        s.t.    w_i == \\pi 2_i - \\pi 1_i  \\forall i = 1:N \\\\
                \\Delta_i (\\pi 2_i + \\pi 1_i) - \\theta_i <= \\lambda    \\forall i = 1:N \\\\
        ```
        """
end

"""
    portfolio_return!(model::JuMP.Model, w, formulation::RobustBertsimas)

Returns worst case return in Bertsimas's uncertainty set, defined by the following dual problem: 

$(_portfolio_return_latex())

"""
function portfolio_return!(model::JuMP.Model, w, formulation::RobustBertsimas)
    # parameters
    r̄ = formulation.predicted_mean
    numA = formulation.number_of_assets
    Δ = formulation.uncertainty_delta
    Λ = formulation.bertsimas_budjet
    # dual variables
    @variable(model, λ >= 0)
    @variable(model, π1[i=1:numA] >= 0)
    @variable(model, π2[i=1:numA] >= 0)
    @variable(model, θ[i=1:numA] >= 0)
    # constraints: from duality
    @constraints(
        model,
        begin
            constrain_dual1[i=1:numA], w[i] == π2[i] - π1[i]
        end
    )
    @constraints(
        model,
        begin
            constrain_dual2[i=1:numA], Δ[i] * (π2[i] + π1[i]) - θ[i] <= λ
        end
    )

    return sum(r̄[i] * (π2[i] - π1[i]) for i in 1:numA) - sum(θ[i] for i in 1:numA)
end

##################################
"""BenTal's uncertainty set"""
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

"""
returns worst case return in BenTal's uncertainty set.
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
    norm_2_pi = @variable(model)
    @constraint(model, [norm_2_pi; sqrt_Σ * w] in JuMP.SecondOrderCone())
    @constraint(model, norm_2_pi <= θ)

    return dot(w, r̄) - θ * δ
end
