using Gadfly

include(".\\ProgEst.jl")

pathprices = ".\\data.xlsx"

pathrf = ".\\rf.csv"

pathrm = ".\\rm.csv" # market index retuns

returns, rf, rm, prices = readprices(pathprices,pathrf,pathrm)

numD,numA = size(returns)

############ Efficient frontier ###################
# Parameters
t = 200
k_back =60
numS = 10000
α = 0.95

# Statistics
Σ,r̄ = mean_variance(returns[t-k_back:t-1,:])
r,P = retornos_montecarlo(Σ,r̄[:,1], numS)

# test opt functions
R = rf[t]
x,e,Cvar,q1_α = minCvar_noRf(r̄,+1000,r,P,α)
w,e,q1_α = maxReturn_LimCvar_noRf(r̄,Cvar,r,P,α)

range = 0:0.0002274:0.005
len = size(range,1)
E_noR = zeros(len)
σ_noR = zeros(len)
E_Cvar = zeros(len)
σ_Cvar = zeros(len)
E_limitCvar = zeros(len)
σ_limitCvar = zeros(len)

x = zeros(len,size(r̄,1))
iter = 0
for R=range
  iter += 1

  x[iter,:],E_Cvar[iter],Cvar,q1_α = minCvar_noRf(r̄,R,r,P,α)
  # x1,v_noR,E_noR[iter] = mv_pfal_quadratic_noRf(Σ,r̄,R)
  x2,E_limitCvar[iter] = maxReturn_LimCvar_noRf(r̄,Cvar,r,P,α)

  # σ_noR[iter] = sqrt(v_noR)
  # σ_Cvar[iter] = sqrt(sum(x[iter,:]'Σ*x[iter,:]))
  σ_limitCvar[iter] = sqrt(sum(x2'Σ*x2))

end

#plot frontier
plot(#layer(x=σ_noR, y=E_noR,
    #  Geom.point,Theme(default_color=color("white"))),
      layer(x=σ_Cvar, y=E_Cvar,
         Geom.point,Theme(default_color=color("blue"))),
    layer(x=σ_limitCvar, y=E_limitCvar,
       Geom.point,Theme(default_color=color("magenta"))),


    Guide.xlabel("σ"), Guide.ylabel("r"), Guide.title("Efficient Frontier Cvar vs Mean-Variance"))
