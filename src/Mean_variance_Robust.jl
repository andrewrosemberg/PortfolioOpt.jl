using JuMP
using LinearAlgebra: dot 

"""
Mean-Variance Portfolio Alocation. Robust return restriction (Worst case return from uncertainty set must be
greater than chosen value). Bertsimas's uncertainty set.
"""
function po_mean_variance_robust_bertsimas!(model, w, Σ, r̄, rf, R, Δ, Λ, max_wealth)
    # num of assets
    numA = size(r̄,1)
    # dual variables
    @variable(model, λ >=0)
    @variable(model, π1[i=1:numA]>=0)
    @variable(model, π2[i=1:numA]>=0)
    @variable(model, θ[i=1:numA]>=0)

    # objective: minimize variance
    @objective(model, Min, sum(w'Σ*w))

    # constraint: minimun return
    @variable(model, E)
    if !haskey(object_dictionary(model), :sum_invested)
      @variable(model, sum_invested)
      @constraint(model, [sum_invested; w] in JuMP.NormOneCone())
    else
      sum_invested = model[:sum_invested]
    end
    @constraint(model, E == rf*(max_wealth-sum_invested) - λ*Λ + sum(r̄[i]*(π2[i]-π1[i]) for i=1:numA) - sum(θ[i] for i=1:numA))
    @constraint(model, E == R)

    # constraints: from duality
    @constraints(model, begin
      constrain_dual1[i=1:numA], w[i] == π2[i]-π1[i]
    end)
    @constraints(model, begin
      constrain_dual2[i=1:numA], λ >= π2[i]*Δ[i]+π1[i]*Δ[i]-θ[i]
    end)
end

"""
Mean-Variance Portfolio Alocation. Robust return restriction (Worst case return from uncertainty set must be
greater than chosen value). BenTal's uncertainty set.
# TODO: Correct
"""
function po_mean_variance_robust_bental!(model, w, Σ, r̄, rf, R, δ, Ucov)
    # num of assets
    numA = size(r̄,1)
    # inverse cov
    invΣ = pinv(Σ, 1E-25)

    @variable(model, θ)
    @variable(model, E)
    @objective(model, Min, sum(w'Σ*w))
    if !haskey(object_dictionary(model), :sum_invested)
      @variable(model, sum_invested)
      @constraint(model, [sum_invested; w] in JuMP.NormOneCone())
    else
      sum_invested = model[:sum_invested]
    end
    @constraint(model, sum_invested <= θ)
    @constraint(model, E == rf*(max_wealth-sum_invested) + dot(w, r̄) - θ*δ)
    @constraint(model, E >=R)
end
