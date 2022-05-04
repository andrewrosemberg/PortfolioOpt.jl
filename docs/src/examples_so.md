# Example Markowitz

## Empirical Forecast

Example of backtest with Mean-Variance strategy with a simple empirical forecaster.

```@example Backtest
using COSMO
using Distributions
using PortfolioOpt
using PortfolioOpt.TestUtils

# Read Prices 
prices = get_test_data();
numD, numA = size(prices) # A: Assets    D: Days

# Calculating returns 
returns_series = percentchange(prices);

# Backtest Parameters 
DEFAULT_SOLVER = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

date_range = timestamp(returns_series)[100:end];

# Backtest
backtest_results = Dict()
backtest_results["EP_markowitz_limit_var"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Parameters
    max_std = 0.003 / market_budget(market)
    k_back = 60

    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    
    # Forecast
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    d = MvNormal(r̄, Σ)

    # PO Formulation
    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(SqrtVariance(d), LessThan(max_std)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

```

## GP Forecast

### Train Forecast

```@example Backtest

# Load our GP-related packages.
using AbstractGPs
using Distributions
using KernelFunctions
using LinearMixingModels
using LinearAlgebra
using KernelFunctions: MOInputIsotopicByOutputs
using Optim # Standard optimisation algorithms.
using ParameterHandling # Helper functionality for dealing with model parameters.
using Zygote # Algorithmic Differentiation
using ParameterHandling: flatten

# Prep forecaster
num_latent_variables = 3
U, S, _ = svd(rand(numA, num_latent_variables));

flat_initial_params, unflatten = flatten((
    var_kernel = positive(0.6),
    λ = positive(2.5),
    var_noise = positive(0.1),
    H = Matrix(Orthogonal(U, Diagonal(S)))
))

# Construct a function to unpack flattened parameters and pull out the raw values.
unpack = ParameterHandling.value ∘ unflatten
params = unpack(flat_initial_params)

function build_gp(params)
    sogp = GP(params.var_kernel * Matern52Kernel() ∘ ScaleTransform(params.λ))
    latent_gp = independent_mogp([sogp for _ in 1:num_latent_variables]);
    return ILMM(latent_gp, params.H);
end

date2float(date) = float(parse(Int, (replace(string(date), "-" => ""))))

# Forecast
dates_for_training = timestamp(returns_series)[1:99]

x_train = MOInputIsotopicByOutputs(date2float.(dates_for_training), numA);
y_train = vec(values(returns_series[dates_for_training]));

function objective(params)
    ilmm = build_gp(params)
    return -logpdf(ilmm(x_train, params.var_noise), y_train)
end

objective(params)

# Optimise using Optim. 
training_results = Optim.optimize(
    objective ∘ unpack,
    θ -> only(Zygote.gradient(objective ∘ unpack, θ)),
    flat_initial_params,
    BFGS(
        alphaguess = Optim.LineSearches.InitialStatic(scaled=true),
        linesearch = Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(show_trace = true);
    inplace=false,
)

# Extracting the final values of the parameters.
# Should be close to truth.
final_params = unpack(training_results.minimizer)

# Test Predict
dates_for_test = timestamp(returns_series)[1:120]
num_test = length(dates_for_test)
x_test = MOInputIsotopicByOutputs(date2float.(dates_for_test), numA);
y_test = vec(values(returns_series[dates_for_test]));

ilmm = build_gp(final_params)
ilmmx = ilmm(x_train, final_params.var_noise)
p_ilmmx = posterior(ilmmx, y_train)
p_i = p_ilmmx(x_test, 1e-6);
marg_pi = marginals(p_i)

# plot
using Plots

plt = Array{Any}(undef, numA);
num_train = 99
for i=1:numA
    plt[i] = scatter(dates_for_training, y_train[num_train*(i-1)+1:num_train*(i)]; label = "Train Data $i", size=(900, 700), Title="Asset $i");
    plot!(plt[i], dates_for_test, mean.(marg_pi[num_test*(i-1)+1:num_test*(i)]); ribbon=std.(marg_pi[num_test*(i-1)+1:num_test*(i)]), label = "Forecast $i");
    scatter!(plt[i], dates_for_test, y_test[num_test*(i-1)+1:num_test*(i)]; label = "Test Data $i")
end
plot(plt...)

```
### Run backtest

```@example Backtest

backtest_results["GP_markowitz_limit_var"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Parameters
    max_std = 0.003 / market_budget(market)
    k_back = 60

    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    
    # Forecast
    dates_for_training = timestamp(past_returns)[end-k_back:end]
    x_train = MOInputIsotopicByOutputs(date2float.(dates_for_training), numA)
    y_train = vec(values(past_returns[dates_for_training]))
    x_test = MOInputIsotopicByOutputs(date2float.([ext[:date]]), numA);

    ilmm = build_gp(final_params)
    ilmmx = ilmm(x_train, final_params.var_noise)
    p_ilmmx = posterior(ilmmx, y_train)
    d = p_ilmmx(x_test, 1e-6)

    # PO Formulation
    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(SqrtVariance(d), LessThan(max_std)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end
```

## Plot

```@example Backtest
using Plots
using Plots.PlotMeasures

plt = plot(;title="Culmulative Wealth",
    xlabel="Time",
    ylabel="Wealth",
    legend=:outertopright,
    left_margin=10mm
);
for (strategy_name, recorders) in backtest_results
    plot!(plt, 
        axes(get_records(recorders[:wealth]), 1), get_records(recorders[:wealth]).data;
        label=strategy_name,
    )
end
plt
```