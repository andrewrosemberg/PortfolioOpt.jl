using Gurobi
using JuMP

"""
Mean-Variance Portfolio Alocation. Robust return restriction (Worst case return from uncertainty set must be
greater than chosen value). Bertsimas's uncertainty set.
"""
function Mean_variance_Robust_Bertsimas(Σ, r̄,rf,R,Δ,Λ)
  # num of assets
  numA = size(r̄,1)
  # problems solver (Quadratic)
  model = Model(solver = GurobiSolver(OutputFlag=0))
  # decision variable
  @variable(model, x[i=1:numA])
  # dual variables
  @variable(model, λ >=0)
  @variable(model, π1[i=1:numA]>=0)
  @variable(model, π2[i=1:numA]>=0)
  @variable(model, θ[i=1:numA]>=0)

  # objective: minimize variance
  @objective(model, Min, sum(x'Σ*x))

  # constraint: minimun return
  @variable(model, E)
  @constraint(model, E == rf*(1-sum(x)) - λ*Λ + sum(r̄[i]*(π2[i]-π1[i]) for i=1:numA) - sum(θ[i] for i=1:numA))
  @constraint(model, E ==R)

  # constraints: from duality
  @constraints(model, begin
    constrain_dual1[i=1:numA], x[i] == π2[i]-π1[i]
  end)
  @constraints(model, begin
    constrain_dual2[i=1:numA], λ >= π2[i]*Δ[i]+π1[i]*Δ[i]-θ[i]
  end)

  status = solve(model);
  x = getvalue(x)
  return x, getobjectivevalue(model), getvalue(E)
end

using Convex

"""
Mean-Variance Portfolio Alocation. Robust return restriction (Worst case return from uncertainty set must be
greater than chosen value). BenTal's uncertainty set.
"""
function Mean_variance_Robust_BenTal(Σ, r̄,rf,R,δ,Ucov)
  # num of assets
  numA = size(r̄,1)
  # inverse cov
  invΣ = pinv(Σ, 1E-25)

  w = Variable(numA)
  θ = Variable()
  π = Variable(numA)
  E = Variable()
  p = minimize(quadform(w,Σ))
  p.constraints += abs(w) <= θ
  p.constraints += w == -π
  p.constraints += E == rf*(1-sum(w[i] for i=1:numA)) + dot(w, r̄) - θ*δ
  p.constraints += E >=R
  solve!(p)
  return round(w.value, 2), round(p.optval, 2), round(E.value, 2)

end
