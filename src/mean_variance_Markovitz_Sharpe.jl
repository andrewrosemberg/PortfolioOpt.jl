using JuMP
using LinearAlgebra

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
Mean-Variance Portfolio Alocation With no risk free asset. Quadratic problem.
Minimize Variance and limit mean.
"""
function po_minvar_limitmean_noRf!(model, w, Σ, r̄, R)
    @constraint(model, sum(r̄'w)>=R)
    @objective(model, Min, sum(w'Σ*w))
end

"""
Mean-Variance Portfolio Alocation With risk free asset. Quadratic problem.
Minimize Variance and limit mean.
"""
function po_minvar_limitmean_Rf!(model, w, Σ, r̄, R, rf, max_wealth)
    @variable(model, E)
    if !haskey(object_dictionary(model), :sum_invested)
      @variable(model, sum_invested)
      @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    else
      sum_invested = model[:sum_invested]
    end
    @constraint(model, sum(r̄'w) + rf*(max_wealth-sum_invested) == E)
    @constraint(model, E >= R)
    @objective(model, Min, sum(w'Σ*w))
end

"""
Mean-Variance Portfolio Alocation With risk free asset. Quadratic problem.
Maximize mean and limit variance.
"""
function po_maxmean_limitvar_Rf!(model, w, Σ, r̄, max_risk, rf, max_wealth)
    @variable(model, E)
    if !haskey(object_dictionary(model), :sum_invested)
      @variable(model, sum_invested)
      @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    else
      sum_invested = model[:sum_invested]
    end
    @constraint(model, sum(r̄'w) + rf*(max_wealth-sum_invested) == E)
    @constraint(model, sum(w'Σ*w) <= max_risk*max_wealth)
    @objective(model, Max, E)
end
