using Plots
using PortfolioOpt

include("./test/test_utils/auxilary_functions.jl")
include("./examples/literature_examples/example_utils.jl")

############ Read Prices #############
Prices = get_test_data()
numD,numA = size(Prices) # A: Assets    D: Days

############# Calculating returns #####################
returns_series = percentchange(Prices)
returns = values(returns_series)
# risk free asset
rf =  fill(3.2e-6, numD) # readcsv(".\\rf.csv")

############ Efficient frontier ###################
t = 100
k_back =60
Σ,r̄ = mean_variance(returns[t-k_back:t-1,:])

range = -0.01:0.0005:0.025;
len = size(range,1);
E = zeros(len);
σ = zeros(len);
E_quad = zeros(len);
σ_quad = zeros(len);
E_quad_Rf = zeros(len);
σ_quad_Rf = zeros(len);

x_sharpe = max_sharpe(Σ,r̄,rf[t])
E_sharpe = r̄'x_sharpe
σ_sharpe = sqrt(x_sharpe'Σ*x_sharpe)
iter = 0
for R=range
      global iter += 1
      # analytical
      x = mean_variance_noRf_analytical(Σ,r̄,R)
      # quadratic optimization no risk free asset
      model, w = base_model(numA)
      po_minvar_limitmean_noRf!(model, w, Σ, r̄, R)
      x_quad, obj, r = compute_solution(model, w)
      # quadratic optimization with risk free asset
      model, w = base_model(numA)
      po_minvar_limitmean_Rf!(model, w, Σ,r̄,R,rf[t], 1)
      x_quad_Rf, obj, r = compute_solution(model, w)

      E[iter] = sum(x'r̄)
      σ[iter] = sqrt((x'Σ*x)[1])
      E_quad[iter] = r[1]
      σ_quad[iter] = sqrt(x_quad'Σ*x_quad)
      E_quad_Rf[iter] = R
      σ_quad_Rf[iter] = sqrt(x_quad_Rf'Σ*x_quad_Rf)
end

# plot frontier
plt  = plot(σ_quad, E_quad,
      title="Efficient Frontier Mean-Variance",
      xlabel = "σ",
      ylabel = "r",
      label = "Quadratic",
      legend = :outertopright 
);
plot!(plt, [0;σ_sharpe], [rf[t];E_sharpe], label = "Sharpe");
plot!(plt, σ, E, label = "Analytical");
plot!(plt, σ_quad_Rf, E_quad_Rf, label = "Quadratic with RF");
plt
