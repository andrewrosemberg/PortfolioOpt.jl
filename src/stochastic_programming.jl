using JuMP
using Distributions

function min_cvar_noRf!(model, w, r̄, R, r, P, α)
    numA = size(r̄,1)
    numS = size(P,1)
    @variable(model, z)
    @variable(model, y[i=1:numS]>=0)

    @constraint(model, sum(r̄'w)>=R)
    @constraints(model, begin
      ys[s=1:numS], y[s] >= z-sum(r[s,:]'w)
    end)

    @objective(model, Max,  z- sum(P'y)/(1.0-α))
end

function max_return_lim_cvar_noRf!(model, w, r̄, λ, r, P, α)
    numA = size(r̄,1)
    numS = size(P,1)
    set_lower_bound.(w, 0)
    @variable(model, z)
    @variable(model, y[i=1:numS]>=0)

    @constraint(model, z- sum(P'y)/(1.0-α) >= -λ)
    @constraint(model, sum(w)==1)
    @constraints(model, begin
      ys[s=1:numS], y[s] >= z-sum(r[s,:]'w)
    end)

    @objective(model, Max, sum(r̄'w))
end
