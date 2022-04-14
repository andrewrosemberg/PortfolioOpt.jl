"""conditional_expectation = -cvar = -expected_shortfall"""
function calculate_measure!(measure::ConditionalExpectedReturn{α,N,ContinuousMultivariateSampleable,EstimatedCase}, w) where {N}
    model = owner_model(w)
    s = ambiguityset(measure)

    # parameters
    numS = sample_size(measure)
    samples = rand(s, numS)
    sample_probability = fill(1/numS,numS)

    # dual variables
    @variable(model, z)
    @variable(model, y[i=1:numS] >= 0)

    @constraints(
        model,
        begin
            ys[s=1:numS], y[s] >= z - sum(dot(samples[:, s], w))
        end
    )
    return z - dot(sample_probability,y) / (1.0 - α)
end


function po_max_conditional_expectation_limit_predicted_return(formulation::AbstractSampleBased, minimal_return; 
    rf::Real = 0.0, current_wealth::Real = 1.0,
    model::JuMP.Model = base_model(formulation.number_of_assets; current_wealth=current_wealth),
    w = model[:w],
    quantile = 0.95, kwargs... 
)
    # auxilary variables
    sum_invested = create_sum_invested_variable(model, w)

    # model
    @constraint(model, predicted_portfolio_return!(model, w, formulation) + rf * (current_wealth - sum_invested) >= minimal_return * current_wealth)

    # objective function
    @objective(model, Max, conditional_expectation!(model, w, formulation; quantile = quantile, kwargs...))

    return model
end

function po_max_predicted_return_limit_conditional_expectation(formulation::AbstractSampleBased, minimal_return;
    rf::Real = 0.0, current_wealth::Real = 1.0,
    model::JuMP.Model = base_model(formulation.number_of_assets; current_wealth=current_wealth),
    w = model[:w],
    quantile = 0.95, kwargs... 
)
    # auxilary variables
    sum_invested = create_sum_invested_variable(model, w)

    # model
    @constraint(model, conditional_expectation!(model, w, formulation; quantile = quantile, kwargs...) >= minimal_return)

    # objective function
    @objective(model, Max, predicted_portfolio_return!(model, w, formulation) + rf * (current_wealth - sum_invested))

    return model
end

