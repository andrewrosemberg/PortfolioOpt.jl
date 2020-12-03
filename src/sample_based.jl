"""Betina Robust Sampled based Formulation"""
struct SampleBased <: AbstractSampleBased
    sampled_returns::Array{Float64,2}
    number_of_assets::Int
    number_of_samples::Int
end

function SampleBased(;
    sampled_returns
)
    number_of_samples, number_of_assets = size(sampled_returns)

    return SampleBased(
        sampled_returns, number_of_assets, number_of_samples
    )
end

function predicted_portfolio_return!(::JuMP.model, w, formulation::AbstractSampleBased; 
    predicted_mean = mean(formulation.sampled_returns, dims=1)'[:,1]
)
    return sum(w'predicted_mean)
end

function portfolio_return!(model::JuMP.model, w, formulation::AbstractSampleBased; kwargs...)
    return predicted_portfolio_return!(model, w, formulation; kwargs...)
end

function predicted_portfolio_variance!(::JuMP.model, w, formulation::AbstractSampleBased; 
    predicted_covariance = cov(formulation.sampled_returns, dims=1)
)
    return sum(w'predicted_covariance * w)
end

function portfolio_variance!(::JuMP.model, w, formulation::AbstractSampleBased; kwargs...)
    return predicted_portfolio_variance!(model, w, formulation)
end

function po_max_predicted_return_limit_return!(model, w, formulation::AbstractSampleBased, R;
    current_wealth = 1.0, rf = 0, kwargs... 
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
    @constraint(model, E = portfolio_return!(model, w, formulation) + rf * (current_wealth - sum_invested))
    @constraint(model, E >= R * current_wealth)

    # objective function
    @objective(model, Max, predicted_portfolio_return!(model, w, formulation; kwargs...) + rf * (current_wealth - sum_invested))

    return nothing
end
