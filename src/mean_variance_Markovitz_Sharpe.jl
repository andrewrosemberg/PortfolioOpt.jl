using Gurobi
using JuMP

"""
Mean-Variance Portfolio Alocation With no risk free asset. Analytical solution.
"""
function mv_pfal_noRf_anal(Σ,r̄,R)
  numA = size(r̄,1)
  invΣ = pinv(Σ, 1E-25)
  x = zeros(numA)
  A = sum(invΣ'r̄)
  B = sum(r̄'r̄)
  C = sum(invΣ)
  one = ones(size(r̄,1))
  D = sum(r̄'*(invΣ'r̄))
  mu = (R*C-A)/(C*D-A)
  x = (1/C)*(invΣ*one)+mu*invΣ*(r̄-one*(A/C))
  return x
end

"""
Mean and Variance of returns
"""
function mean_variance(returns)
  r̄ = mean(returns,1)'
  Σ = cov(returns)
  return Σ,r̄
end

"""
Mean-Variance Portfolio Alocation With no risk free asset. Quadratic problem.
"""
function mv_pfal_quadratic_noRf(Σ,r̄,R)

  numA = size(r̄,1)
  model = Model(solver=GurobiSolver())
  @variable(model, w[i=1:numA])
  @constraint(model, sum(r̄'w)>=R)
  @constraint(model, sum(w)==1)

  @objective(model, Min, sum(w'Σ*w))
  status = solve(model)
  w = getvalue(w)
  r = sum(r̄'w)
  return w,getobjectivevalue(model),r
end

"""
Mean-Variance Portfolio Alocation With risk free asset. Quadratic problem.
"""
function mv_pfal_quadratic_Rf(Σ,r̄,R,rf)
  numA = size(r̄,1)
  model = Model(solver=GurobiSolver(OutputFlag=0))
  @variable(model, w[i=1:numA])
  @variable(model, E)
  @constraint(model, sum(r̄'w) + rf*(1-sum(w))==E)
  @constraint(model, E ==R)
  @objective(model, Min, sum(w'Σ*w))
  status = solve(model)
  w = getvalue(w)
  return w, getobjectivevalue(model), getvalue(E)
end

"""
Maximize sharp coeficient alocation.
"""
function max_sharpe(Σ,r̄,rf)
  one = ones(size(r̄,1))
  invΣ = pinv(Σ, 1E-25)
  v = invΣ*(r̄.-one*rf)
  return v./sum(v)
end
