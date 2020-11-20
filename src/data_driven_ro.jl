"""Portifolio optimization with data-driven robust loss constraint."""
function betina_robust(model, w, returns, r̄, rf, λ; j_robust::Int64 = 45, max_wealth=1.0)
    numD,numA = size(returns)

    if !haskey(object_dictionary(model), :sum_invested)
        @variable(model, sum_invested)
        @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    else
        sum_invested = model[:sum_invested]
    end

    # Robust Constraints
    @constraints(model, begin
    robust[j=1:j_robust],  sum(returns[end-j,i]*w[i] for i=2:numA) >= λ*max_wealth
    end)

    # objective function
    @objective(model, Max, sum(r̄'w) + rf*(max_wealth-sum_invested))

    return 
end