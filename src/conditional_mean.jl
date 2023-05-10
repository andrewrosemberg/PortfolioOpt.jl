"""
    calculate_measure!(measure::ConditionalExpectedReturn{1.0,S,R}, w)

Returns worst case return in the convex hull uncertainty set, defined by the following dual problem: 

```math
    \\max_{\\theta} \\quad  \\theta \\\\
    s.t.  \\quad \\theta \\leq r_s ' w \\quad \\forall s = 1:\\mathcal{S} \\\\
```

Further information:
  - Fernandes, B., Street, A., ValladA˜ £o, D., e Fernandes, C. (2016). An adaptive robust portfolio optimization model with loss constraints based on data-driven polyhedral uncertainty sets. European Journal of Operational Research, 255(3):961 – 970. ISSN 0377-2217. URL.

"""
function convex_hull(measure, w)
    model = owner_model(w)
    s = ambiguityset(measure)
    samples = rand(s)

    # auxilary variables
    θ = @variable(model)
    # convex hull
    @constraint(model, sum(samples' * w , dims=2) .>= θ)

    return θ
end

"""conditional_expectation = -cvar = -expected_shortfall"""
function calculate_measure!(measure::ConditionalExpectedReturn{α,S,EstimatedCase}, w) where {S<:DeterministicSamples,α}
    if α == 1.0
        return convex_hull(measure, w)
    end
    model = owner_model(w)
    s = ambiguityset(measure)

    # parameters
    numS = sample_size(s)
    samples = rand(s)
    sample_probability = samples_probability(s)

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

