using Gurobi
using JuMP
using Distributions

function retornos_montecarlo(Σ,r̄, numS)
  d = MvNormal(r̄, Σ)
  r = rand(d, numS)
  P = pdf(d,r)
  return r',P/sum(P)
end

function minCvar_noRf(r̄,R,r,P,α)
  numA = size(r̄,1)
  numS = size(P,1)
  model = Model(solver=GurobiSolver(OutputFlag=0))
  @variable(model, w[i=1:numA])
  @variable(model, z)
  @variable(model, y[i=1:numS]>=0)

  @constraint(model, sum(r̄'w)>=R)
  @constraint(model, sum(w)==1)
  @constraints(model, begin
    ys[s=1:numS], y[s] >= z-sum(r[s,:]'w)
  end)

  @objective(model, Max,  z- sum(P'y)/(1.0-α) )
  status = solve(model)
  w = getvalue(w)
  Cvar = -getobjectivevalue(model)
  r = sum(r̄'w)
  q1_α = getvalue(z)
  return w,r,Cvar,q1_α
end

function maxReturn_LimCvar_noRf(r̄,λ,r,P,α)
  numA = size(r̄,1)
  numS = size(P,1)
  model = Model(solver=GurobiSolver(OutputFlag=0))
  @variable(model, w[i=1:numA]>=0)
  @variable(model, z)
  @variable(model, y[i=1:numS]>=0)

  @constraint(model, z- sum(P'y)/(1.0-α) >= -λ)
  @constraint(model, sum(w)==1)
  @constraints(model, begin
    ys[s=1:numS], y[s] >= z-sum(r[s,:]'w)
  end)

  @objective(model, Max, sum(r̄'w))
  status = solve(model)
  w = getvalue(w)
  r = getobjectivevalue(model)
  q1_α = getvalue(z)

  return w,r,q1_α
end
