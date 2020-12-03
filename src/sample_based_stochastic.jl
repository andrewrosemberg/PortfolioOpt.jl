"""conditional_expectation = -cvar = -expected_shortfall"""
function conditional_expectation!(model, w, formulation::AbstractSampleBased; 
    sample_probability = fill(1/formulation.number_of_samples,formulation.number_of_samples),
    quantile = 0.95
)
    # parameters
    numA = formulation.number_of_assets
    numS = formulation.number_of_samples
    α = quantile
    # dual variables
    @variable(model, z)
    @variable(model, y[i=1:numS] >= 0)

    @constraints(
        model,
        begin
            ys[s=1:numS], y[s] >= z - sum(formulation.sampled_returns[s, :]'w)
        end
    )
    return z - sum(sample_probability'y) / (1.0 - α)
end


function po_max_conditional_expectation_limit_predicted_return!(model, w, formulation::AbstractSampleBased, R;
    current_wealth = 1.0, rf = 0, quantile = 0.95, kwargs... 
)
    if !haskey(object_dictionary(model), :sum_invested)
        @variable(model, sum_invested)
        @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    else
        sum_invested = model[:sum_invested]
    end

    # model
    @constraint(model, predicted_portfolio_return!(model, w, formulation) + rf * (current_wealth - sum_invested) >= R * current_wealth)

    # objective function
    @objective(model, Max, conditional_expectation!(model, w, formulation; quantile = quantile, kwargs...))

    return nothing
end

function po_max_conditional_expectation_limit_predicted_return!(model, w, formulation::AbstractSampleBased, λ;
    current_wealth = 1.0, rf = 0, quantile = 0.95, kwargs... 
)
    if !haskey(object_dictionary(model), :sum_invested)
        @variable(model, sum_invested)
        @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    else
        sum_invested = model[:sum_invested]
    end

    # model
    @constraint(model, conditional_expectation!(model, w, formulation; quantile = quantile, kwargs...) >= λ)

    # objective function
    @objective(model, Max, predicted_portfolio_return!(model, w, formulation) + rf * (current_wealth - sum_invested))

    return nothing
end

