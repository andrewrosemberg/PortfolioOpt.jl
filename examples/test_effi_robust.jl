using COSMO
using Plots
using PortfolioOpt
using PortfolioOpt.TestUtils: 
    backtest_po, get_test_data, mean_variance, 
    percentchange, timestamp, rename!

############ Read Prices #############
Prices = get_test_data()
numD, numA = size(Prices) # A: Assets    D: Days

############# Calculating returns #####################
returns_series = percentchange(Prices)
returns = values(returns_series)
# risk free asset
rf = fill(0.0, numD) # fill(3.2e-4, numD) # readcsv(".\\rf.csv")
rm = mean(returns; dims=2)

############ Efficient frontier ###################

DEFAULT_SOLVER = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 9000000
)

# General Parameters
t = 120
k_back = 60
Σ, r̄ = mean_variance(returns[(t - k_back):(t - 1), :])

# Parameter Strategies
δ = 0.028 * 6.05
Δ = std(returns[(end - k_back):end, :]; dims=1)'[:,1] / 5

formulation_markowitz = MeanVariance(;
    predicted_mean = r̄,
    predicted_covariance = Σ,
)
formulation_soyster = RobustBertsimas(;
    predicted_mean = r̄,
    predicted_covariance = Σ,
    uncertainty_delta = Δ,
    bertsimas_budget = 6.0,
)
formulation_bertsimas = RobustBertsimas(;
    predicted_mean = r̄,
    predicted_covariance = Σ,
    uncertainty_delta = Δ,
    bertsimas_budget = 3.0,
)
formulation_bental = RobustBenTal(;
    predicted_mean = r̄,
    predicted_covariance = Σ,
    uncertainty_delta = δ
)

# Loop range and log (no robustness)
range_R = 0:0.00005:0.0033; # Used for `po_min_variance_limit_return`
len = size(range_R, 1)
R_noR = zeros(len);
E_noR = zeros(len);
σ_noR = zeros(len);
R_soy = zeros(len);
x_noR = zeros(len, size(r̄, 1));

# Loop no robust
iter = 0
for R_0 in range_R
    global iter += 1
    model = po_min_variance_limit_return(formulation_markowitz, R_0; rf = rf[t])
    x_noR[iter, :], R_noR[iter] = compute_solution(model, DEFAULT_SOLVER), value.(model[:R])
    E_noR[iter] = sum(x_noR[iter, :]'r̄)
    σ_noR[iter] = sqrt(sum(x_noR[iter, :]'Σ * x_noR[iter, :]))
end

# Loop range and log 
range_R = 0:0.00005:0.0015; # Used for `po_min_variance_limit_return`
len = size(range_R, 1)
R_ber = zeros(len);
E_ber = zeros(len);
σ_ber = zeros(len);
R_soy = zeros(len);
E_soy = zeros(len);
σ_soy = zeros(len);
R_ben_tal = zeros(len);
E_ben_tal = zeros(len);
σ_ben_tal = zeros(len);
x_soy = zeros(len, size(r̄, 1));
x_ber = zeros(len, size(r̄, 1));
x_ben_tal = zeros(len, size(r̄, 1));

# Loop robust
iter = 0
for R_0 in range_R
    global iter += 1
    model = po_min_variance_limit_return(formulation_soyster, R_0; rf = rf[t])
    x_soy[iter, :], R_soy[iter] = compute_solution(model, DEFAULT_SOLVER), value.(model[:R])
    model = po_min_variance_limit_return(formulation_bertsimas, R_0; rf = rf[t])
    x_ber[iter, :], R_ber[iter] = compute_solution(model, DEFAULT_SOLVER), value.(model[:R])
    model = po_min_variance_limit_return(formulation_bental, R_0; rf = rf[t])
    x_ben_tal[iter, :], R_ben_tal[iter] = compute_solution(model, DEFAULT_SOLVER), value.(model[:R])

    E_ber[iter] = sum(x_ber[iter, :]'r̄)
    E_soy[iter] = sum(x_soy[iter, :]'r̄)
    E_ben_tal[iter] = sum(x_ben_tal[iter, :]'r̄)

    σ_ber[iter] = sqrt(sum(x_ber[iter, :]'Σ * x_ber[iter, :]))
    σ_soy[iter] = sqrt(sum(x_soy[iter, :]'Σ * x_soy[iter, :]))
    σ_ben_tal[iter] = sqrt(sum(x_ben_tal[iter, :]'Σ * x_ben_tal[iter, :]))
end

# plot true frontier
plt = plot(
    σ_noR,
    E_noR;
    title="Efficient Frontier Robust Mean-Variance",
    xlabel="Standard Deviation",
    ylabel="Expected Return",
    label="Markowitz",
    legend=:outertopright,
);
plot!(plt, σ_ber, E_ber; label="Bertsimas");
plot!(plt, σ_soy, E_soy; label="Soyster");
plot!(plt, σ_ben_tal, E_ben_tal; label="BenTal");
plt

# plot believed frontier
plt = plot(
    σ_noR,
    R_noR;
    title="Efficient Frontier Robust Mean-Variance",
    xlabel="Standard Deviation",
    ylabel="Optimized Return (R)",
    label="Markowitz",
    legend=:outertopright,
);
plot!(plt, σ_ber, R_ber; label="Bertsimas");
plot!(plt, σ_soy, R_soy; label="Soyster");
plot!(plt, σ_ben_tal, R_ben_tal; label="BenTal");
plt

