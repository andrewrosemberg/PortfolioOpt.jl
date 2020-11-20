using JuMP
using LinearAlgebra: dot

"""
Mean-Variance Portfolio Alocation. Robust return restriction (Worst case return from uncertainty set must be
greater than chosen value). Bertsimas's uncertainty set.
Minimize Variance and limit mean.
"""
function po_minvar_limitmean_robust_bertsimas!(model, w, Σ, r̄, rf, R, Δ, Λ, max_wealth)
    # num of assets
    numA = size(r̄, 1)
    # dual variables
    @variable(model, λ >= 0)
    @variable(model, π1[i=1:numA] >= 0)
    @variable(model, π2[i=1:numA] >= 0)
    @variable(model, θ[i=1:numA] >= 0)

    # objective: minimize variance
    @objective(model, Min, sum(w'Σ * w))

    # constraint: minimun return
    @variable(model, E)
    if !haskey(object_dictionary(model), :sum_invested)
        @variable(model, sum_invested)
        @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    else
        sum_invested = model[:sum_invested]
    end
    @constraint(
        model,
        E ==
        rf * (max_wealth - sum_invested) - λ * Λ +
        sum(r̄[i] * (π2[i] - π1[i]) for i in 1:numA) - sum(θ[i] for i in 1:numA)
    )
    @constraint(model, E >= R)

    # constraints: from duality
    @constraints(
        model,
        begin
            constrain_dual1[i=1:numA], w[i] == π2[i] - π1[i]
        end
    )
    @constraints(
        model,
        begin
            constrain_dual2[i=1:numA], Δ[i] * (π2[i] + π1[i]) - θ[i] <= λ
        end
    )
    return nothing
end

"""
Mean-Variance Portfolio Alocation. Robust return restriction (Worst case return from uncertainty set must be
greater than chosen value). BenTal's uncertainty set.
Minimize Variance and limit mean.
"""
function po_minvar_limitmean_robust_bental!(model, w, Σ, r̄, rf, R, δ, max_wealth)
    # # num of assets
    numA = size(r̄, 1)
    # # inverse cov
    # # invΣ = pinv(Σ, 1E-25)
    sqrt_Σ = sqrt(Σ)

    @variable(model, θ)
    @variable(model, E)
    @objective(model, Min, sum(w'Σ * w))
    if !haskey(object_dictionary(model), :sum_invested)
        @variable(model, sum_invested)
        @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    else
        sum_invested = model[:sum_invested]
    end

    norm_2_pi = @variable(model)
    @constraint(model, [norm_2_pi; sqrt_Σ * w] in JuMP.SecondOrderCone())
    @constraint(model, norm_2_pi <= θ)
    @constraint(model, E == rf * (max_wealth - sum_invested) + dot(w, r̄) - θ * δ)
    @constraint(model, E >= R)
    return nothing
end

"""
Mean-Variance Portfolio Alocation. Robust return restriction (Worst case return from uncertainty set must be
greater than chosen value). Bertsimas's uncertainty set.
Maximize mean and limit variance.
"""
function po_maxmean_limitvar_robust_bertsimas!(
    model, w, Σ, r̄, rf, max_risk, Δ, Λ, max_wealth
)
    # num of assets
    numA = size(r̄, 1)
    # dual variables
    @variable(model, λ >= 0)
    @variable(model, π1[i=1:numA] >= 0)
    @variable(model, π2[i=1:numA] >= 0)
    @variable(model, θ[i=1:numA] >= 0)

    # constraint: minimun return
    @variable(model, E)
    if !haskey(object_dictionary(model), :sum_invested)
        @variable(model, sum_invested)
        @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    else
        sum_invested = model[:sum_invested]
    end
    @constraint(
        model,
        E ==
        rf * (max_wealth - sum_invested) - λ * Λ +
        sum(r̄[i] * (π2[i] - π1[i]) for i in 1:numA) - sum(θ[i] for i in 1:numA)
    )
    @constraint(model, sum(w'Σ * w) <= max_risk * max_wealth)

    # constraints: from duality
    @constraints(
        model,
        begin
            constrain_dual1[i=1:numA], w[i] == π2[i] - π1[i]
        end
    )
    @constraints(
        model,
        begin
            constrain_dual2[i=1:numA], Δ[i] * (π2[i] + π1[i]) - θ[i] <= λ
        end
    )

    @objective(model, Max, E)
    return nothing
end

"""
Mean-Variance Portfolio Alocation. Robust return restriction (Worst case return from uncertainty set must be
greater than chosen value). BenTal's uncertainty set.
Maximize mean and limit variance.
"""
function po_maxmean_limitvar_robust_bental!(model, w, Σ, r̄, rf, max_risk, δ, max_wealth)
    # # num of assets
    numA = size(r̄, 1)
    # # inverse cov
    # # invΣ = pinv(Σ, 1E-25)
    sqrt_Σ = sqrt(Σ)

    @variable(model, θ)
    @variable(model, E)

    if !haskey(object_dictionary(model), :sum_invested)
        @variable(model, sum_invested)
        @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1))
    else
        sum_invested = model[:sum_invested]
    end

    norm_2_pi = @variable(model)
    @constraint(model, [norm_2_pi; sqrt_Σ * w] in JuMP.SecondOrderCone())
    @constraint(model, norm_2_pi <= θ)
    @constraint(model, E == rf * (max_wealth - sum_invested) + dot(w, r̄) - θ * δ)
    @constraint(model, sum(w'Σ * w) <= max_risk * max_wealth)

    @objective(model, Max, E)
    return nothing
end
