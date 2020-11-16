using Gadfly

include(".\\mean_variance_robust.jl")

include(".\\mean_variance_markovitz_sharpe.jl")

pathprices = ".\\data.xlsx"

pathrf = ".\\rf.csv"

pathrm = ".\\rm.csv"

returns, rf, rm, prices = readprices(pathprices,pathrf,pathrm) # generic read

numD,numA = size(returns)

############ Efficient frontier ###################
t = 200
k_back =60
Σ,r̄ = mean_variance(returns[t-k_back:t-1,:])
δ =rf[t]*10 # Defining the uncertainty set
Δ = fill(δ,size(r̄,1)) # Defining the uncertainty set
range = 0:0.5:10
len = size(range,1)
E_ber = zeros(len)
σ_ber = zeros(len)
E_noR = zeros(len)
σ_noR = zeros(len)
E_soy = zeros(len)
σ_soy = zeros(len)
E_ben_tal = zeros(len)
σ_ben_tal = zeros(len)
x = zeros(len,size(r̄,1))
iter = 0
for R=range
  iter += 1
  model, w = base_model(numA)
  po_mean_variance_Rf!(model, w, Σ,r̄,R,rf[t], 1) # Mean_variance_Robust_Bertsimas(Σ, r̄,rf[t],R,Δ,0)
  x[iter,:],v_noR,E_noR[iter] = compute_solution_dual(model, w)
  model, w = base_model(numA)
  po_mean_variance_robust_bertsimas!(model, w, Σ, r̄, rf[t], R, Δ, 2.5, 1)
  x2,v_ber,E_ber[iter] = compute_solution_dual(model, w)
  model, w = base_model(numA)
  po_mean_variance_robust_bertsimas!(model, w, Σ, r̄, rf[t], R, Δ, 5.0, 1)
  x3,v_soy,E_soy[iter] = compute_solution_dual(model, w)
  model, w = base_model(numA)
  po_mean_variance_robust_bental!(model, w, Σ, r̄, rf[t], R, δ, 0)
  x4,v_ben_tal,E_ben_tal[iter] = compute_solution_dual(model, w)

  σ_noR[iter] = sqrt(v_noR)
  σ_ber[iter] = sqrt(sum(x2'Σ*x2))
  σ_soy[iter] = sqrt(v_soy)
  σ_ben_tal[iter] = sqrt(v_ben_tal)

end

#plot frontier
plot(layer(x=σ_noR, y=E_noR,
     Geom.point,Theme(default_color=color("white"))),
     layer(x=σ_ber, y=E_ber,
        Geom.point,Theme(default_color=color("blue"))),
     layer(x=σ_soy, y=E_soy,
           Geom.point,Theme(default_color=color("orange"))),
     layer(x=σ_ben_tal, y=E_ben_tal,
           Geom.point,Theme(default_color=color("magenta"))),

    Guide.xlabel("σ"), Guide.ylabel("r"), Guide.title("Efficient Frontier Robust Mean-Variance"))
