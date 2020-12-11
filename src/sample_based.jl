"""Betina Robust Sampled based Formulation"""
struct SampleBased <: AbstractSampleBased
    sampled_returns::Array{Float64,2}
    number_of_assets::Int
    number_of_samples::Int
end

function SampleBased(;
    sampled_returns::Array{Float64,2}
)
    number_of_samples, number_of_assets = size(sampled_returns)

    return SampleBased(
        sampled_returns, number_of_assets, number_of_samples
    )
end

function predicted_portfolio_return!(::JuMP.Model, w, formulation::AbstractSampleBased; 
    predicted_mean = mean(formulation.sampled_returns, dims=1)'[:,1]
)
    return sum(w'predicted_mean)
end

function portfolio_return!(model::JuMP.Model, w, formulation::AbstractSampleBased; kwargs...)
    return predicted_portfolio_return!(model, w, formulation; kwargs...)
end

function predicted_portfolio_variance!(::JuMP.Model, w, formulation::AbstractSampleBased; 
    predicted_covariance = cov(formulation.sampled_returns, dims=1)
)
    return sum(w'predicted_covariance * w)
end

function portfolio_variance!(model::JuMP.Model, w, formulation::AbstractSampleBased; kwargs...)
    return predicted_portfolio_variance!(model, w, formulation)
end

function po_max_predicted_return_limit_return(formulation::AbstractPortfolioFormulation, minimal_return; 
    rf::Real = 0.0, current_wealth::Real = 1.0,
    model::JuMP.Model = base_model(formulation.number_of_assets; current_wealth=current_wealth),
    w = model[:w],
    kwargs... 
)
    # auxilary variables
    @variable(model, R)
    sum_invested = create_sum_invested_variable(model, w)

    # model
    @constraint(model, R == portfolio_return!(model, w, formulation) + rf * (current_wealth - sum_invested))
    @constraint(model, R >= minimal_return * current_wealth)

    # objective function
    @objective(model, Max, predicted_portfolio_return!(model, w, formulation; kwargs...) + rf * (current_wealth - sum_invested))

    return model
end
