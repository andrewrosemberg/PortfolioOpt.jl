"""
    calculate_measure!(w, measure::ConditionalExpectedReturn{Inf,N,S,R})

Returns worst case return in the convex hull uncertainty set, defined by the following dual problem: 

```math
    \\max_{\\theta} \\quad  \\theta \\\\
    s.t.  \\quad \\theta \\leq r_s ' w \\quad \\forall s = 1:\\mathcal{S} \\\\
```

Further information:
  - Fernandes, B., Street, A., ValladA˜ £o, D., e Fernandes, C. (2016). An adaptive robust portfolio optimization model with loss constraints based on data-driven polyhedral uncertainty sets. European Journal of Operational Research, 255(3):961 – 970. ISSN 0377-2217. URL.

"""
function calculate_measure!(w, measure::ConditionalExpectedReturn{Inf,N,ContinuousMultivariateSampleable,R}) where {N,R}
    model = owner_model(w)
    s = ambiguityset(measure)
    samples = rand(s, sample_size(measure))

    # auxilary variables
    θ = @variable(model)
    # convex hull
    @constraint(model, sum(samples' * w , dims=2) .>= θ)

    return θ
end