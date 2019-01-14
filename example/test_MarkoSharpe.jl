using Gurobi
using JuMP
using Gadfly
include("mean_variance_Markovitz_Sharpe.jl")

############ Read Prices (listed form most recent to oldest) #############
Prices = readcsv(".\\data.csv")
numD,numA = size(Prices) # A: Assets    D: Days

############# Calculating returns #####################
returns = (Prices[numD-1:-1:1,:]./Prices[numD:-1:2,:]).-1

# risk free asset
rDi = readcsv(".\\rf.csv")

############ Efficient frontier ###################
t = 200
k_back =60
Σ,r̄ = mean_variance(returns[t-k_back:t-1,:])

range = -0.01:0.0005:0.025
len = size(range,1)
E = zeros(len)
σ = zeros(len)
E_quad = zeros(len)
σ_quad = zeros(len)
E_quad_Rf = zeros(len)
σ_quad_Rf = zeros(len)

x_sharpe = max_sharpe(Σ,r̄,rDi[t])
E_sharpe = r̄'x_sharpe
σ_sharpe = sqrt(x_sharpe'Σ*x_sharpe)
iter = 0
for R=range
  iter += 1
  x = mv_pfal_noRf_anal(Σ,r̄,R)
  x_quad,r = mv_pfal_quadratic_noRf(Σ,r̄,R)
  x_quad_Rf = mv_pfal_quadratic_Rf(Σ,r̄,R,rDi[t])

  E[iter] = R
  σ[iter] = sqrt((x'Σ*x)[1])
  E_quad[iter] = r[1]
  σ_quad[iter] = sqrt(x_quad'Σ*x_quad)
  E_quad_Rf[iter] = R
  σ_quad_Rf[iter] = sqrt(x_quad_Rf'Σ*x_quad_Rf)
end
#plot frontier
plot(#layer(x=σ, y=E,
     #Geom.point,Theme(default_color=color("orange"))),
     layer(x=σ_quad, y=E_quad,
          Geom.point,Theme(default_color=color("blue"))),
     #layer(x=σ_quad_Rf, y=E_quad_Rf,
           #Geom.point,Theme(default_color=color("white"))),

    layer(x=[0;σ_sharpe], y=[rDi[t];E_sharpe],
          Geom.point,Geom.line,Theme(default_color=color("white"))),
    Guide.xlabel("σ"), Guide.ylabel("r"), Guide.title("Efficient Frontier Mean-Variance"))
