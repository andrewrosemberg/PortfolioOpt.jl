using JuMP
using MarketData
using LinearAlgebra
using Plots

include("./examples/test_utils.jl")
include("./src/mean_variance_markovitz_sharpe.jl")
include("./src/mean_variance_robust.jl")

############ Read Prices #############
Prices = get_test_data()
numD,numA = size(Prices) # A: Assets    D: Days

############# Calculating returns #####################
returns_series = percentchange(Prices)
returns = values(returns_series)
# risk free asset
rf =  fill(3.2e-4, numD) # readcsv(".\\rf.csv")
rm = mean(returns, dims=2)

############ Efficient frontier ###################
t = 120
k_back = 60
Σ,r̄ = mean_variance(returns[t-k_back:t-1,:])
δ =0.032 # Defining the uncertainty set
Δ = fill(rf[t]*2,size(r̄,1)) # Defining the uncertainty set
range = 0:0.0005:0.0025;
len = size(range,1)
E_ber = zeros(len);
σ_ber = zeros(len);
E_noR = zeros(len);
σ_noR = zeros(len);
E_soy = zeros(len);
σ_soy = zeros(len);
E_ben_tal = zeros(len);
σ_ben_tal = zeros(len);
x = zeros(len,size(r̄,1));
x2 = zeros(len,size(r̄,1));
x3 = zeros(len,size(r̄,1));
x4 = zeros(len,size(r̄,1));
iter = 0
for R=range
      global iter += 1
      model, w = base_model(numA; allow_borrow = false)
      po_minvar_limitmean_Rf!(model, w, Σ,r̄,R,rf[t], 1) # Mean_variance_Robust_Bertsimas(Σ, r̄,rf[t],R,Δ,0)
      x[iter,:], v_noR, E_noR[iter] = compute_solution_dual(model, w)
      model, w = base_model(numA; allow_borrow = false)
      po_minvar_limitmean_robust_bertsimas!(model, w, Σ, r̄, rf[t], R, Δ, 3.0, 1)
      x2[iter,:], v_ber, E_ber[iter] = compute_solution_dual(model, w)
      model, w = base_model(numA; allow_borrow = false)
      po_minvar_limitmean_robust_bertsimas!(model, w, Σ, r̄, rf[t], R, Δ, numA, 1)
      x3[iter,:], v_soy, E_soy[iter] = compute_solution_dual(model, w)
      model, w = base_model(numA; allow_borrow = false)
      po_minvar_limitmean_robust_bental!(model, w, Σ, r̄, rf[t], R, δ, 1)
      x4[iter,:], v_ben_tal, E_ben_tal[iter] = compute_solution_dual(model, w)

      σ_noR[iter] = sqrt(sum(x[iter,:]'Σ*x[iter,:])) #sqrt(v_noR)
      σ_ber[iter] = sqrt(sum(x2[iter,:]'Σ*x2[iter,:])) #sqrt(v_ber)
      σ_soy[iter] = sqrt(sum(x3[iter,:]'Σ*x3[iter,:])) #sqrt(v_soy)
      σ_ben_tal[iter] = sqrt(sum(x4[iter,:]'Σ*x4[iter,:])) #sqrt(v_ben_tal)
end

# plot frontier
plt  = plot(σ_noR, E_noR,
      title="Efficient Frontier Robust Mean-Variance",
      xlabel = "σ",
      ylabel = "r",
      label = "Markowitz",
      legend = :outertopright 
);
plot!(plt, σ_ber, E_ber, label = "Bertsimas");
plot!(plt, σ_soy, E_soy, label = "Soyster");
plot!(plt, σ_ben_tal, E_ben_tal, label = "BenTal");
plt
