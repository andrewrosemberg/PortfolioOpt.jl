"""Maximize expected return under distribution uncertainty using DRO"""
function po_maxmean_delague(model, w, r̄, Σ, a, b, γ1, γ2, K)
    numA = size(r̄, 1)

    @variable(model, P[i=1:numA, j=1:numA])
    @variable(model, p[i=1:numA])
    @variable(model, s)
    @variable(model, Q[i=1:numA, j=1:numA])
    @variable(model, q[i=1:numA])
    @variable(model, r)

    @constraint(model, p .== -q / 2 - Q * r̄)

    @SDconstraint(model, [[P p]; [p' s]] >= 0)

    for k in 1:K
        @SDconstraint(
            model, [[Q (q / 2 + a[k] * w / 2)]; [(q' / 2 + a[k] * w' / 2) (r + b[k])]] >= 0
        )
    end

    @objective(
        model,
        Min,
        γ2 * dot(Σ, Q) - sum(r̄'Q * r̄) + r + dot(Σ, P) - 2 * dot(r̄, p) + γ1 * s
    )

    return nothing
end
