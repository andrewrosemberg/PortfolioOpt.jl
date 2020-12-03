"""Betina Robust Sampled based Formulation"""
struct RobustBetina <: AbstractSampleBased
    sampled_returns::Array{Float64,2}
    number_of_assets::Int
    number_of_samples::Int
end

function RobustBetina(;
    sampled_returns::Array{Float64,2}
)
    number_of_samples, number_of_assets = size(sampled_returns)

    return RobustBetina(
        sampled_returns, number_of_assets, number_of_samples
    )
end

"""
returns worst case return in Betina's uncertainty set.
"""
function portfolio_return!(model::JuMP.Model, w, formulation::RobustBetina)
    # auxilary variables
    E_risky = @variable(model, E_risky)
    # convex hull
    @constraint(model, sum(formulation.sampled_returns * w , dims=2) .>= E_risky)

    return E_risky
end
