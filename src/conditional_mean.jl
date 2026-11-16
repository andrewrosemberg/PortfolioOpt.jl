"""
    calculate_measure!(measure::ConditionalExpectedReturn{0.0,S}, w)

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

"""
    calculate_measure!(measure::ConditionalExpectedReturn{α,S}, w) where {S<:DeterministicSamples,α}

Returns the conditional expected return of the portfolio over its worst `α` fraction of
samples (i.e., `-CVaR_α = -expected shortfall`), using the Rockafellar-Uryasev formulation:

```math
    \\max_{z, y \\geq 0} \\quad z - \\frac{1}{\\alpha} \\sum_{s=1}^{\\mathcal{S}} p_s y_s \\\\
    s.t.  \\quad y_s \\geq z - r_s ' w \\quad \\forall s = 1:\\mathcal{S} \\\\
```

`α = 0.0` gives the worst-case return over the samples (see [`convex_hull`](@ref)), and
`α = 1.0` gives the sample mean.
"""
function calculate_measure!(measure::ConditionalExpectedReturn{α,S}, w) where {S<:DeterministicSamples,α}
    if α == 0.0
        return convex_hull(measure, w)
    end
    model = owner_model(w)
    s = ambiguityset(measure)

    # parameters
    numS = sample_size(s)
    samples = rand(s)
    sample_probability = samples_probability(s)

    # dual variables
    z = @variable(model)
    y = @variable(model, [i=1:numS], lower_bound = 0)

    @constraint(model, [s=1:numS], y[s] >= z - dot(samples[:, s], w))
    return z - dot(sample_probability, y) / α
end

