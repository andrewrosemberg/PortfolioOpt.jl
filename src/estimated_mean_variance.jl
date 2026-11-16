
function calculate_measure!(measure::ExpectedReturn{S}, w) where {S<:AmbiguitySet}
    return dot(mean(measure.ambiguity_set), w)
end

# quadratic expression when `w` are plain variables (or numbers)
function calculate_measure!(measure::Variance{S}, w::Union{AbstractVector{<:JuMP.AbstractVariableRef},AbstractVector{<:Real},Real}) where {S<:AmbiguitySet}
    return dot(cov(measure.ambiguity_set) * w, w)
end

# rotated second order cone for any other JuMP expression (e.g. affine expressions of the variables)
function calculate_measure!(measure::Variance{S}, w::AbstractVector{<:JuMP.AbstractJuMPScalar}) where {S<:AmbiguitySet}
    model = _owner_model(w)
    
    # Cholesky decomposition of the covariance matrix
    Σ = PDMat(cov(measure.ambiguity_set))
    sqrt_Σ = collect(Σ.chol.U)

    # Extra dimention to represent the portfolio variance
    risk = @variable(model);
    @constraint(model, [risk; 0.5; sqrt_Σ * w] in JuMP.RotatedSecondOrderCone())

    return risk
end

function calculate_measure!(measure::SqrtVariance{S}, w) where {S<:AmbiguitySet}
    model = _owner_model(w)
    
    # Cholesky decomposition of the covariance matrix
    Σ = PDMat(cov(measure.ambiguity_set))
    sqrt_Σ = collect(Σ.chol.U)

    # Extra dimention to represent the square root of the portfolio variance
    risk = @variable(model);
    @constraint(model, [risk; sqrt_Σ * w] in JuMP.SecondOrderCone())

    return risk
end
