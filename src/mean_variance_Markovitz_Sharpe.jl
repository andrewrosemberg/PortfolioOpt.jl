using JuMP

"""
Mean-Variance Portfolio Alocation With no risk free asset. Analytical solution.
"""
function mean_variance_noRf_analytical(Σ,r̄,R)
    numA = size(r̄,1)
    invΣ = pinv(Σ, 1E-25)
    x = zeros(numA)
    A = sum(invΣ'r̄)
    B = sum(r̄'r̄)
    C = sum(invΣ)
    one = ones(size(r̄,1))
    D = sum(r̄'*(invΣ'r̄))
    mu = (R*C-A)/(C*D-A)
    x = (1/C)*(invΣ*one)+mu*invΣ*(r̄-one*(A/C))
    return x
end

"""
Maximize sharp coeficient alocation.
"""
function max_sharpe(Σ,r̄,rf)
    one = ones(size(r̄,1))
    invΣ = pinv(Σ, 1E-25)
    v = invΣ*(r̄.-one*rf)
    return v./sum(v)
end

"""
Mean-Variance Portfolio Alocation With no risk free asset. Quadratic problem.
"""
function po_mean_variance_noRf!(model, w, Σ, r̄, R)
    @constraint(model, sum(r̄'w)>=R)
    @objective(model, Min, sum(w'Σ*w))
end

"""
Mean-Variance Portfolio Alocation With risk free asset. Quadratic problem.
"""
function po_mean_variance_Rf!(model, w, Σ,r̄,R,rf, max_wealth)
    @variable(model, w[i=1:numA])
    @variable(model, E)
    if !haskey(object_dictionary(model), :sum_invested)
      @variable(model, sum_invested)
      @constraint(model, [sum_invested; w] in JuMP.NormOneCone())
    else
      sum_invested = model[:sum_invested]
    end
    @constraint(model, sum(r̄'w) + rf*(max_wealth-sum_invested)==E)
    @constraint(model, E ==R)
    @objective(model, Min, sum(w'Σ*w))
end

function base_model(numA::Integer)
    model = Model()
    w = @variable(model, w[i=1:numA])
    @variable(model, sum_invested)
    @constraint(model, [sum_invested; w] in JuMP.NormOneCone())
    @constraint(model, sum_invested==1)
    return model, w
end
