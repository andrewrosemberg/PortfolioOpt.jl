"""Delague's uncertainty set"""
struct RobustDelague <: AbstractPortfolioFormulation # AbstractMeanVariance
    predicted_mean::Array{Float64,1}
    predicted_covariance::Array{Float64,2}
    γ1::Float64
    γ2::Float64
    utility_coeficients::Array{Float64,1}
    utility_intercepts::Array{Float64,1}
    number_of_utility_pieces::Int
    number_of_assets::Int
end

function RobustDelague(;
    predicted_mean::Array{Float64,1},
    predicted_covariance::Array{Float64,2},
    γ1::Float64,
    γ2::Float64,
    utility_coeficients::Array{Float64,1},
    utility_intercepts::Array{Float64,1},
)
    number_of_assets = size(predicted_mean, 1)
    number_of_utility_pieces = size(utility_coeficients, 1)
    if number_of_assets != size(predicted_covariance, 1)
        error("size of predicted mean ($number_of_assets) different to size of predicted covariance ($(size(predicted_covariance, 1)))")
    end
    if number_of_utility_pieces != size(utility_intercepts, 1)
        error("number of utility coeficients ($(number_of_utility_pieces)) different to number of utility intercepts ($(size(utility_intercepts,1)))")
    end

    @assert γ1 >= 0
    @assert γ2 >= 1
    @assert issymmetric(predicted_covariance)
    return RobustDelague(
        predicted_mean, predicted_covariance, γ1, γ2, utility_coeficients, 
        utility_intercepts, number_of_utility_pieces, number_of_assets
    )
end

"""Maximize expected return under distribution uncertainty using DRO"""
function po_max_utility_return!(model::JuMP.Model, w, formulation::RobustDelague)
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
    @SDconstraint(model, [[P p]; [p' s]] >= 0)
    for k in 1:K
        @SDconstraint(
            model, [[Q (q / 2 + a[k] * w / 2)]; [(q' / 2 + a[k] * w' / 2) (r + b[k])]] >= 0
        )
    end

    # objective
    @objective(
        model,
        Min, # Min because this is the dual problem
        γ2 * dot(Σ, Q) - sum(r̄'Q * r̄) + r + dot(Σ, P) - 2 * dot(r̄, p) + γ1 * s
    )

    return nothing
end
