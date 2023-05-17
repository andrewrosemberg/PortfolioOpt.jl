
function calculate_measure!(measure::ExpectedReturn{S}, w) where {S<:AmbiguitySet}
    return dot(mean(measure.ambiguity_set), w)
end

function calculate_measure!(measure::Variance{S}, w::Union{Vector{VariableRef},Real}) where {S<:AmbiguitySet}
    return dot(cov(measure.ambiguity_set) * w, w)
end

function calculate_measure!(measure::Variance{S}, w::Vector{AffExpr}) where {S<:AmbiguitySet}
    model = owner_model(w)
    
    # Cholesky decomposition of the covariance matrix
    Σ = PDMat(cov(measure.ambiguity_set))
    sqrt_Σ = collect(Σ.chol.U)

    # Extra dimention to represent the portfolio variance
    risk = @variable(model);
    @constraint(model, [risk; 0.5; sqrt_Σ * w] in JuMP.RotatedSecondOrderCone())

    return risk
end

function calculate_measure!(measure::SqrtVariance{S}, w) where {S<:AmbiguitySet}
    model = owner_model(w)
    
    # Cholesky decomposition of the covariance matrix
    Σ = PDMat(cov(measure.ambiguity_set))
    sqrt_Σ = collect(Σ.chol.U)

    # Extra dimention to represent the square root of the portfolio variance
    risk = @variable(model);
    @constraint(model, [risk; sqrt_Σ * w] in JuMP.SecondOrderCone())

    return risk
end
