using COSMO
using Plots
using PortfolioOpt
using PortfolioOpt.TestUtils: 
    backtest_po, get_test_data, mean_variance, 
    percentchange, timestamp, rename!
using Statistics

include("./examples/example_utils.jl")

############ Read Prices #############
Prices = get_test_data()
numD, numA = size(Prices) # A: Assets    D: Days

############# Calculating returns #####################
returns_series = percentchange(Prices)
returns = values(returns_series)
# risk free asset
rf = fill(0.0, numD) # fill(3.2e-4, numD) # readcsv(".\\rf.csv")

############ Efficient frontier ###################

DEFAULT_SOLVER = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 9000000, "eps_abs" => 1e-6, "eps_rel" => 1e-6
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
    bertsimas_budget = 2.8,
)
formulation_bental = RobustBenTal(;
    predicted_mean = r̄,
    predicted_covariance = Σ,
    uncertainty_delta = δ
)

# Loop range and log (no robustness)
range_R = 0:0.00005:0.0015; # Used for `po_min_variance_limit_return`
len = size(range_R, 1)
R_noR = zeros(len);
E_noR = zeros(len);
σ_noR = zeros(len);
R_soy = zeros(len);
x_noR = zeros(len, size(r̄, 1));

# Loop
iter = 0
for R_0 in range_R
    global iter += 1
    model = po_min_variance_limit_return(formulation_markowitz, R_0; rf = rf[t])
    x_noR[iter, :], R_noR[iter] = compute_solution(model, DEFAULT_SOLVER), value.(model[:R])
    E_noR[iter] = sum(x_noR[iter, :]'r̄)
    σ_noR[iter] = sqrt(sum(x_noR[iter, :]'Σ * x_noR[iter, :]))
end

# Loop range and log 
range_R = 0:0.00005:0.0006; # Used for `po_min_variance_limit_return`
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
    model = po_min_variance_limit_return(formulation_markowitz, R_0; rf = rf[t])
    x_noR[iter, :], R_noR[iter] = compute_solution(model, DEFAULT_SOLVER), value.(model[:R])
    E_noR[iter] = sum(x_noR[iter, :]'r̄)
    σ_noR[iter] = sqrt(sum(x_noR[iter, :]'Σ * x_noR[iter, :]))

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

# plot optimistic frontier
plt = plot(
    σ_noR,
    E_noR;
    title="Efficient Frontier Mean-Variance",
    xlabel="Standard Deviation",
    ylabel="Expected Return",
    label="Markowitz",
    legend=:outertopright,
    ylims = (0.0, 0.0015),
    xlims = (0.0, 0.01)
    # markershape=:circle
);
plot!(plt, σ_ber, E_ber; label="Bertsimas");
plot!(plt, σ_soy, E_soy; label="Soyster");
# plot!(plt, σ_ben_tal, E_ben_tal; label="BenTal");
plt

# plot robust frontier
plt = plot(
    σ_noR,
    R_noR;
    title="Problem-Specific Efficient Frontier Mean-Variance",
    xlabel="Standard Deviation",
    ylabel="Robust Return (R)",
    label="Markowitz",
    legend=:outertopright,
    ylims = (0.0, 0.0006),
    xlims = (0.0, 0.015)
);
plot!(plt, σ_ber, R_ber; label="Bertsimas");
plot!(plt, σ_soy, R_soy; label="Soyster");
# plot!(plt, σ_ben_tal, R_ben_tal; label="BenTal");
plt

# plot frontier regret bertsimas
min_return_noR_bertsimas = [worst_case_return(x_noR[i, :], formulation_bertsimas, DEFAULT_SOLVER) for i = 1:length(R_noR)]
max_return_noR_bertsimas = [best_case_return(x_noR[i, :], formulation_bertsimas, DEFAULT_SOLVER) for i = 1:length(R_noR)]
max_return_bertsimas = [best_case_return(x_ber[i, :], formulation_bertsimas, DEFAULT_SOLVER) for i = 1:length(R_ber)]

plt = plot(
    σ_noR,
    R_noR;
    title="Uncertain Efficient Frontier Mean-Variance (Bertsimas)",
    xlabel="Standard Deviation",
    ylabel="Return",
    label="Markowitz",
    legend=:outertopright,
    ribbon=(R_noR-min_return_noR_bertsimas,max_return_noR_bertsimas - R_noR),
    xlims = (0.0, 0.005),
    ylims = (0.0, 0.003),
);
plot!(plt, σ_ber, (R_ber+max_return_bertsimas)/2; 
    label="Bertsimas",
    ribbon=((R_ber+max_return_bertsimas)/2-R_ber,max_return_bertsimas - (R_ber+max_return_bertsimas)/2)
);
plt

# plot frontier regret soyster
min_return_noR_soyster = [worst_case_return(x_noR[i, :], formulation_soyster, DEFAULT_SOLVER) for i = 1:length(R_noR)]
max_return_noR_soyster = [best_case_return(x_noR[i, :], formulation_soyster, DEFAULT_SOLVER) for i = 1:length(R_noR)]
max_return_bertsimas_soyester = [best_case_return(x_ber[i, :], formulation_soyster, DEFAULT_SOLVER) for i = 1:length(R_ber)]
min_return_bertsimas_soyester = [worst_case_return(x_ber[i, :], formulation_soyster, DEFAULT_SOLVER) for i = 1:length(R_ber)]
max_return_soyester = [best_case_return(x_soy[i, :], formulation_soyster, DEFAULT_SOLVER) for i = 1:length(R_soy)]

plt = plot(
    σ_noR,
    R_noR;
    title="Uncertain Efficient Frontier Mean-Variance (Soyster)",
    xlabel="Standard Deviation",
    ylabel="Return",
    label="Markowitz",
    legend=:outertopright,
    ribbon=(R_noR-min_return_noR_soyster,max_return_noR_soyster - R_noR),
    xlims = (0.0, 0.005),
    ylims = (-0.001, 0.004),
);
plot!(plt, σ_ber, (min_return_bertsimas_soyester+max_return_bertsimas_soyester)/2; 
    label="Bertsimas",
    ribbon=((min_return_bertsimas_soyester+max_return_bertsimas_soyester)/2-min_return_bertsimas_soyester,max_return_bertsimas_soyester - (min_return_bertsimas_soyester+max_return_bertsimas_soyester)/2)
);
plot!(plt, σ_soy, (R_soy+max_return_soyester)/2; 
    label="Soyster",
    ribbon=((R_soy+max_return_soyester)/2-R_soy,max_return_soyester - (R_soy+max_return_soyester)/2)
);
plt

# plot frontier regret bental
min_return_noR_bental = [worst_case_return(x_noR[i, :], formulation_bental, DEFAULT_SOLVER) for i = 1:length(R_noR)]
max_return_noR_bental = [best_case_return(x_noR[i, :], formulation_bental, DEFAULT_SOLVER) for i = 1:length(R_noR)]
max_return_bental = [best_case_return(x_ben_tal[i, :], formulation_bental, DEFAULT_SOLVER) for i = 1:length(R_ben_tal)]

plt = plot(
    σ_noR,
    R_noR;
    title="Uncertain Efficient Frontier Mean-Variance (Ben-Tal)",
    xlabel="Standard Deviation",
    ylabel="Return",
    label="Markowitz",
    legend=:outertopright,
    ribbon=(R_noR-min_return_noR_bental,max_return_noR_bental - R_noR),
);
plot!(plt, σ_ben_tal, (R_ben_tal+max_return_bental)/2; 
    label="Ben-Tal",
    ribbon=((R_ben_tal+max_return_bental)/2-R_ben_tal,max_return_bental - (R_ben_tal+max_return_bental)/2)
);
plt