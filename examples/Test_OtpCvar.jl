using JuMP
using MarketData
using LinearAlgebra
using Plots

include("./examples/test_utils.jl")
include("./src/stochastic_programming.jl")

############ Read Prices #############
Prices = get_test_data()
numD,numA = size(Prices) # A: Assets    D: Days

############ Efficient frontier ###################
# Parameters
t = 100
k_back = 60
numS = 10000
α = 0.95

# Statistics
Σ,r̄ = mean_variance(returns[t-k_back:t-1,:])
r,P = returns_montecarlo(Σ,r̄[:,1], numS)

# test opt functions
R = 0.003
model, w = base_model(numA)
min_cvar_noRf!(model, w, r̄, R, r, P, α);
x,e,Cvar,q1_α = compute_solution_stoc(model, w)
model, w = base_model(numA)
max_return_lim_cvar_noRf!(model, w, r̄, Cvar, r, P, α)
w,e,q1_α = compute_solution_stoc_2(model, w)

# Loop
range = 0:0.0002274:0.005;
len = size(range,1);
E_noR = zeros(len);
σ_noR = zeros(len);
E_Cvar = zeros(len);
σ_Cvar = zeros(len);
E_limitCvar = zeros(len);
σ_limitCvar = zeros(len);

x = zeros(len,size(r̄,1));
iter = 0;
for R=range
    global iter += 1

    model, w = base_model(numA)
    min_cvar_noRf!(model, w, r̄, R, r, P, α)
    x[iter,:],E_Cvar[iter],Cvar,q1_α = compute_solution_stoc(model, w)
    model, w = base_model(numA)
    max_return_lim_cvar_noRf!(model, w, r̄, Cvar, r, P, α)
    x2,E_limitCvar[iter] = compute_solution_stoc_2(model, w)

    σ_Cvar[iter] = sqrt(sum(x[iter,:]'Σ*x[iter,:]))
    σ_limitCvar[iter] = sqrt(sum(x2'Σ*x2))

    model, w = base_model(numA)
    po_minvar_limitmean_noRf!(model, w, Σ, r̄, R)
    x_quad, obj, E_noR[iter] = compute_solution(model, w)
    σ_noR[iter] = sqrt(obj)
end

# plot frontier
plt  = plot(σ_limitCvar, E_Cvar,
    title="Efficient Frontier Cvar", # Cvar vs Mean-Variance
    xlabel = "σ",
    ylabel = "r",
    label = "Limit Cvar",
    legend = :topleft 
);
plot!(plt, σ_Cvar, E_Cvar, label = "Min Cvar");
plt
