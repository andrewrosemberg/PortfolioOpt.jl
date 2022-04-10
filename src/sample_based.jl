struct DeterministicSamples{T<:Real} <: ContinuousMultivariateSampleable
    samples::Array{T,2}
end

Base.length(s::DeterministicSamples) = size(s.samples, 1)

function Distributions.rand(
    rng::AbstractRNG, s::DeterministicSamples{T}, num_requested_samples::Int
) where {T<:Real}
    samples = s.samples
    x = Array{T, 2}(undef, length(s), num_requested_samples)
    num_samples = size(samples, 2)
    num_first_samples = min(num_requested_samples, num_samples)
    x[:, 1:num_first_samples] = samples[:, 1:num_first_samples]
    num_last_samples = num_requested_samples - num_first_samples
    if num_last_samples > 0
        x[:, num_first_samples+1:end] = samples[:, rand(rng, 1:num_samples, num_last_samples)]
    end
    return x
end

function Distributions.rand(
    ::AbstractRNG, s::DeterministicSamples{T}, ::Nothing
) where {T<:Real}
    return s.samples
end

mean(s::DeterministicSamples) = Statistics.mean(s.samples, dims=1)'[:,1]
cov(s::DeterministicSamples) = Statistics.cov(s.samples, dims=1)

# TODO
function po_max_predicted_return_limit_return(formulation::AmbiguitySet, minimal_return; 
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
