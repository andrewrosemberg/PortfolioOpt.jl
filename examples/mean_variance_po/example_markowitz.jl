using Pkg
Pkg.activate(@__DIR__)

using SCS
using Distributions
using Plots
using Plots.PlotMeasures
using PortfolioOpt
using PortfolioOpt.TestUtils
using CovarianceEstimation

# Read Prices
prices = get_test_data();
numD, numA = size(prices) # A: Assets    D: Days

# Calculating returns
returns_series = percentchange(prices);

day2float(day) = float(parse(Int, (replace(replace(string(day), " day" => ""), "s" => ""))))

plot_dates = timestamp(returns_series)[1:120];
train_dates = plot_dates[1:99];
test_dates = plot_dates[100:end];
all_days = plot_dates - minimum(plot_dates);
x_plot = day2float.(all_days);
x_train = x_plot[1:99];
x_test = x_plot[100:end];
y_train = values(returns_series[plot_dates[1:99]]);
y_test = values(returns_series[plot_dates[100:end]]);

# normalise the observations
ymean = mean(y_train);
ystd = std(y_train);
y_train_norm = (y_train .- ymean) ./ ystd;

# Backtest Parameters
DEFAULT_SOLVER = optimizer_with_attributes(
    #COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
    SCS.Optimizer
)

date_range = timestamp(returns_series)[100:end];

############################################################################################################
# Empirical forecaster
############################################################################################################

# Backtest
backtest_results = Dict()
backtest_results["EP_markowitz_limit_var"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext

    # println("backtest day ", ext[:date])

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
end;

backtest_results["biweight_markowitz_limit_var"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext

    # println("backtest day ", ext[:date])

    # Parameters
    max_std = 0.003 / market_budget(market)
    k_back = 60

    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)

    # Forecast
    _, r̄ = mean_variance(returns[(end - k_back):end, :])
    Σ = cov(BiweightMidcovariance(; c=9.0, modify_sample_size=false), returns[(end - k_back):end, :])
    d = MvNormal(r̄, Σ)

    # PO Formulation
    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(SqrtVariance(d), LessThan(max_std)),
    )

    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end;

############################################################################################################
## GP Forecaster
############################################################################################################

# Load our GP-related packages.
using AbstractGPs
using KernelFunctions
using LinearMixingModels
using LinearAlgebra
using KernelFunctions: MOInputIsotopicByOutputs
using Optim # Standard optimisation algorithms.
using ParameterHandling # Helper functionality for dealing with model parameters.
# using Zygote # Algorithmic Differentiation
using ForwardDiff
using ParameterHandling: flatten
using PrincipledRisk
using FactorisedDistributions
using Flux


# Prep forecaster
num_latent_variables = 3
U, S, _ = svd(rand(numA, num_latent_variables));

period = 7.0  # assume weekly periodicity
len_init = period * 5.0
period_len_init = 0.5

flat_initial_params, unflatten = flatten((
    len = positive(len_init),
    period_len = positive(period_len_init),  # bound this param to prevent numerical errors
    var_noise = positive(0.1),
    H = Matrix(Orthogonal(U, Diagonal(S)))
))

# Construct a function to unpack flattened parameters and pull out the raw values.
unpack = ParameterHandling.value ∘ unflatten
params = unpack(flat_initial_params)

# TODO: our desired kernel doesn't work with LinearMixingModels, raise an issue
function build_gp(θ)
    k_maper = compose(MAPeriodicKernel(; r=[θ.period_len]), ScaleTransform(1 / period))
    k_rq = with_lengthscale(RationalQuadraticKernel(), θ.len)
    sogp = GP(k_rq * k_maper)
    latent_gp = independent_mogp([sogp for _ in 1:num_latent_variables]);
    return ILMM(latent_gp, θ.H);
end

x_train_gp = MOInputIsotopicByOutputs(x_train, numA);
y_train_gp = vec(y_train_norm);

function objective(θ)
    ilmm = build_gp(θ)
    return -logpdf(ilmm(x_train_gp, θ.var_noise), y_train_gp)
end

objective(params)

# Optimise using Optim.
training_results = Optim.optimize(
    objective ∘ unpack,
    # θ -> only(Zygote.gradient(objective ∘ unpack, θ)),
    θ -> ForwardDiff.gradient(objective ∘ unpack, θ),
    flat_initial_params,
    BFGS(
        alphaguess = Optim.LineSearches.InitialStatic(scaled=true),
        linesearch = Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(show_trace = true, iterations=50);
    inplace=false,
)

# Extracting the final values of the parameters.
# Should be close to truth.
final_params = unpack(training_results.minimizer);

# Test Predict
x_plot_gp = MOInputIsotopicByOutputs(x_plot, numA);
# x_test_gp = MOInputIsotopicByOutputs(x_test, numA);

ilmm = build_gp(final_params);
# TODO: it is unclear whether var_noise should be included at both steps here
ilmmx = ilmm(x_train_gp, final_params.var_noise);
p_ilmmx = posterior(ilmmx, y_train_gp);
p_i = p_ilmmx(x_plot_gp, final_params.var_noise);  # TODO: check whether obs noise needs to be injected here
marg_pi = marginals(p_i);

# un-normalise the predictions
marg_pi_unnorm = marg_pi .* ystd .+ ymean;

xmin = minimum(plot_dates);
xmax = maximum(plot_dates);
num_plot = length(plot_dates);
plt = Array{Any}(undef, numA);
for i=1:numA
    plt[i] = scatter(train_dates, y_train[:, i]; label = "Train Data $i", size=(900, 700), Title="Asset $i");
    plot!(plt[i], plot_dates, mean.(marg_pi_unnorm[num_plot*(i-1)+1:num_plot*(i)]); ribbon=std.(marg_pi_unnorm[num_plot*(i-1)+1:num_plot*(i)]), label = "Forecast $i", xlims=(xmin, xmax));
    scatter!(plt[i], test_dates, y_test[:, i]; label = "Test Data $i")
end
plot(plt..., size=(950, 600), xrotation = 15)

backtest_results["GP_markowitz_limit_var"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext

    # println("backtest day ", ext[:date])

    # Parameters
    max_std = 0.003 / market_budget(market)
    k_back = 60

    # Prep
    numD, numA = size(past_returns)

    # Forecast
    dates_for_training = timestamp(past_returns)[end-k_back:end]
    days_for_training = dates_for_training - minimum(dates_for_training)
    day_for_test = ext[:date] - minimum(dates_for_training)
    x_train_gp = MOInputIsotopicByOutputs(day2float.(days_for_training), numA)
    x_test_gp = MOInputIsotopicByOutputs([day2float(day_for_test)], numA)
    y_train = values(past_returns[dates_for_training])
    y_train_gp = vec((y_train .- ymean) ./ ystd)  # normalise the observations

    ilmm = build_gp(final_params)
    # TODO: it is unclear whether var_noise should be included at both steps here
    ilmmx = ilmm(x_train_gp, final_params.var_noise)
    p_ilmmx = posterior(ilmmx, y_train_gp)
    d = p_ilmmx(x_test_gp, final_params.var_noise)
    # un-normalise the predictions
    r̄ = mean(d) .* ystd .+ ymean
    Σ = cov(d) .* ystd^2
    d = MvNormal(r̄, Σ)

    # # PO Formulation
    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(SqrtVariance(d), LessThan(max_std)),
    )

    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end;

############################################################################################################
## GPRN Forecaster
############################################################################################################

num_latents = num_latent_variables + numA * num_latent_variables;
num_weights = num_latents-num_latent_variables;

# define the likelihood model:
likelihood(θ) = GPRNLikelihood(numA, num_latent_variables, θ.output_noise_scale, θ.latent_noise_scale)

# TODO: how should we initialise?
len_latent_init = period * 5.0
len_weight_init = period * 10.0  # the mixing weights should vary more slowly
period_len_latent_init = 0.5
period_len_weight_init = 2.0

θ_latents = (;  # hyperparameters for the GP prior latent factors
    len=positive(len_latent_init),
    period_len=positive(period_len_latent_init),
);

θ_weights = (;  # hyperparameters for the GP prior weight factors
    len=positive(len_weight_init),
    period_len=positive(period_len_weight_init),
);

θ_lik = (;  # hyperparameters for the likelihood model
    output_noise_scale=positive.(0.1ones(numA)),
    latent_noise_scale=positive.(0.25ones(num_latent_variables)),
);

θ_init = (;  # all hyperparameters
    factors = vcat(
        repeat([θ_latents], num_latent_variables),
        repeat([θ_weights], num_weights),
    ),
    lik = θ_lik
);


function kernel(θ)
    k_maper = compose(MAPeriodicKernel(; r=[θ.period_len]), ScaleTransform(1 / period))
    k_rq = with_lengthscale(RationalQuadraticKernel(), θ.len)
    return k_rq * k_maper
end

gp_model = UnevaluatedLatentFactorisedGP(
    repeat([UnevaluatedGP(kernel)], num_latents),
    likelihood
)


function hyperprior(θ)
    """ a Gaussian prior over the kernel lengthscales """
    lens = map(f -> f.len, θ.factors)
    period_lens = map(f -> f.period_len, θ.factors)
    lprior = MvNormal(
        vcat(len_latent_init*ones(num_latent_variables), len_weight_init*ones(num_weights)),
        Diagonal(vcat(20.0ones(num_latent_variables), 40.0ones(num_weights)))
        )
    plprior = MvNormal(
        vcat(period_len_latent_init*ones(num_latent_variables), period_len_weight_init*ones(num_weights)),
        Diagonal(vcat(1.0ones(num_latent_variables), 1.0ones(num_weights)))
        )
    return logpdf(lprior, lens) + logpdf(plprior, period_lens)
end

y_train_gprn = RowVecs(y_train_norm);

println("training model...")
lr_newton = 0.3;
lr_adam = 0.05;
num_iters = 50;
opt = ADAM(lr_adam);
cubature = Unscented3rdOrder(num_latents);
post, θ_opt = train(
    x_train,
    y_train_gprn,
    gp_model,
    θ_init,
    num_iters,
    lr_newton,
    opt,
    cubature,
    hyperprior
);
# println(ParameterHandling.value(θ))

marginal_posterior_predictive = predict_in_data_space(post, x_plot, cubature);
marg_mean = vecvec_to_matrix(mean.(marginal_posterior_predictive));
marg_std = vecvec_to_matrix((diag.(cov.(marginal_posterior_predictive)))) .^ 0.5;

# un-normalise the predictions
marg_mean_unnorm = marg_mean .* ystd .+ ymean;
marg_std_unnorm = marg_std .* ystd;

plt = Array{Any}(undef, numA);
for i=1:numA
    plt[i] = scatter(train_dates, y_train[:, i]; label = "Train Data $i", size=(900, 700), Title="Asset $i");
    plot!(plt[i], plot_dates, marg_mean_unnorm[:, i]; ribbon=marg_std_unnorm[:, i], label = "Forecast $i", xlims=(xmin, xmax));
    scatter!(plt[i], test_dates, y_test[:, i]; label = "Test Data $i")
end
plot(plt..., size=(950, 600), xrotation = 15)

# function sample_noise_cov(p, latent_prediction::AbstractMvNormal, num_samps)
#     predsamp = rand.(repeat([latent_prediction], num_samps))
#     moments = conditional_moments.((p.prior.lik, ), predsamp)
#     secondmoment = map(m -> m[3], moments)
#     return secondmoment
# end

# sample_noise_cov(p, latent_prediction::AbstractVector{<:AbstractMvNormal}, num_samps) = sample_noise_cov.((p, ), latent_prediction, (num_samps, ))

function sample_cov(p, latent_prediction::AbstractMvNormal, num_samps)
    l = p.prior.lik
    predsamp = rand.(repeat([latent_prediction], num_samps))
    # the following is based on the specific form of the GPRNLikelihood from PrincipledRisk.jl
    F_cov = cov(latent_prediction)[1:l.latent_dim, 1:l.latent_dim] + Diagonal(l.latent_noise_scale)
    W_samp = map(p -> p[l.latent_dim+1:end] .+ l.additive_const, predsamp)  # sample the weights
    W_samp = reshape.(W_samp, l.output_dim, l.latent_dim)
    sample_cov = W_samp .* (F_cov, ) .* transpose.(W_samp) .+ (Diagonal(l.output_noise_scale), )
    return sample_cov
end

sample_cov(p, latent_prediction::AbstractVector{<:AbstractMvNormal}, num_samps) = sample_cov.((p, ), latent_prediction, (num_samps, ))


# measure the uncertainty in the noise covariance at a single test point
test_pred = predict_in_data_space(post, x_test[1:1], cubature);
testcov_unnorm = cov(only(test_pred)) .* ystd^2;
latent_predict = predict(post, x_test[1:1]);
s_cov = sample_cov(post, latent_predict, 5000);
s_cov_unnorm = s_cov .* ystd^2;
# cov_mean = mean.(s_cov_unnorm);
cov_std = std.(s_cov_unnorm);

Matisless(cov_1,cov_2) = minimum(eigvals(cov_2 .- cov_1))>=0
λ2 = (1:0.1:20)[findfirst((λ) -> all([Matisless(cov_aux, testcov_unnorm * λ) for cov_aux in s_cov_unnorm[1]]), 1:0.1:20)]
biggest_cov = testcov_unnorm * λ2

p1 = heatmap(testcov_unnorm, title="Predictive Covariance");
p2 = heatmap(only(cov_std), title="std of Predictive Covariance");
p3 = heatmap(biggest_cov, title="SDP UB of Predictive Covariance");
plot(p1, p2, p3, size=(900, 350))

backtest_results["GPRN_markowitz_limit_var"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext

    # println("backtest day ", ext[:date])

    # Parameters
    max_std = 0.003 / market_budget(market)
    k_back = 60

    # Prep
    numD, numA = size(past_returns)

    # Forecast
    dates_for_training = timestamp(past_returns)[end-k_back:end]
    days_for_training = dates_for_training - minimum(dates_for_training)
    day_for_test = ext[:date] - minimum(dates_for_training)
    x_train_gprn = day2float.(days_for_training)
    x_test_gprn = [day2float(day_for_test)]
    y_train = values(past_returns[dates_for_training])
    y_train_gprn = RowVecs((y_train .- ymean) ./ ystd)  # normalise the observations

    # need to re-fit the variational parameters
    num_iters_backtest = 7
    opt_backtest = ADAM(0.0)  # don't change the hyperparameters
    post, θ_opt = train(
        x_train_gprn,
        y_train_gprn,
        gp_model,
        θ_init,
        num_iters_backtest,
        lr_newton,
        opt_backtest,
        cubature,
        hyperprior
    )

    pred = only(predict_in_data_space(post, x_test_gprn, cubature))
    # un-normalise the predictions
    r̄ = mean(pred) .* ystd .+ ymean
    Σ = cov(pred) .* ystd^2
    d = MvNormal(r̄, Σ)

    # PO Formulation
    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(SqrtVariance(d), LessThan(max_std)),
    )

    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end;

############################################################################################################
## GPRN Forecaster + DRO Delage
############################################################################################################

# Inflated mean covariance

backtest_results["GPRN_delage_inflated_mean"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext

    # println("backtest day ", ext[:date])

    # Parameters
    max_risk = 0.001 / market_budget(market)
    k_back = 60
    num_samples_cov = 500

    # Prep
    numD, numA = size(past_returns)

    # Forecast
    dates_for_training = timestamp(past_returns)[end-k_back:end]
    days_for_training = dates_for_training - minimum(dates_for_training)
    day_for_test = ext[:date] - minimum(dates_for_training)
    x_train_gprn = day2float.(days_for_training)
    x_test_gprn = [day2float(day_for_test)]
    y_train = values(past_returns[dates_for_training])
    y_train_gprn = RowVecs((y_train .- ymean) ./ ystd)  # normalise the observations

    # need to re-fit the variational parameters
    num_iters_backtest = 7
    opt_backtest = ADAM(0.0)  # don't change the hyperparameters
    post, θ_opt = train(
        x_train_gprn,
        y_train_gprn,
        gp_model,
        θ_init,
        num_iters_backtest,
        lr_newton,
        opt_backtest,
        cubature,
        hyperprior
    )

    pred = only(predict_in_data_space(post, x_test_gprn, cubature))
    latent_predict = predict(post, x_test_gprn);
    s_cov = sample_cov(post, latent_predict, num_samples_cov);

    # un-normalise the predictions
    r̄ = mean(pred) .* ystd .+ ymean
    Σ = cov(pred) .* ystd^2
    s_cov_unnorm = s_cov .* ystd^2;
    d = MvNormal(r̄, Σ)

    γ2 = (1:0.1:20)[findfirst((λ) -> all([Matisless(cov_aux, Σ * λ) for cov_aux in s_cov_unnorm[1]]), 1:0.1:20)]

    # PO Formulation
    s = MomentUncertainty(d; γ1=0.005, γ2=γ2)
    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(ConditionalExpectedReturn{WorstCase}(0.95, s, 999), LessThan(max_risk)),
    )

    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end;


backtest_results["GPRN_delage_std"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext

    # println("backtest day ", ext[:date])

    # Parameters
    max_risk = 0.003 / market_budget(market)
    k_back = 60
    num_samples_cov = 500

    # Prep
    numD, numA = size(past_returns)

    # Forecast
    dates_for_training = timestamp(past_returns)[end-k_back:end]
    days_for_training = dates_for_training - minimum(dates_for_training)
    day_for_test = ext[:date] - minimum(dates_for_training)
    x_train_gprn = day2float.(days_for_training)
    x_test_gprn = [day2float(day_for_test)]
    y_train = values(past_returns[dates_for_training])
    y_train_gprn = RowVecs((y_train .- ymean) ./ ystd)  # normalise the observations

    # need to re-fit the variational parameters
    num_iters_backtest = 7
    opt_backtest = ADAM(0.0)  # don't change the hyperparameters
    post, θ_opt = train(
        x_train_gprn,
        y_train_gprn,
        gp_model,
        θ_init,
        num_iters_backtest,
        lr_newton,
        opt_backtest,
        cubature,
        hyperprior
    )

    pred = only(predict_in_data_space(post, x_test_gprn, cubature))
    latent_predict = predict(post, x_test_gprn);
    s_cov = sample_cov(post, latent_predict, num_samples_cov);

    # un-normalise the predictions
    r̄ = mean(pred) .* ystd .+ ymean
    Σ = cov(pred) .* ystd^2
    s_cov_unnorm = s_cov .* ystd^2;
    Σ = Σ + only(std.(s_cov_unnorm))
    Σ_psd = (Σ .+ Σ') / 2
    d = MvNormal(r̄, Σ_psd)

    # PO Formulation
    s = MomentUncertainty(d; γ1=0.005, γ2=1.0)
    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(ConditionalExpectedReturn{WorstCase}(0.95, s, 999), LessThan(max_risk)),
    )

    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end;

############################################################################################################
## Plot Backtest results
############################################################################################################

plt = plot(;title="Culmulative Wealth",
    xlabel="Time",
    ylabel="Wealth",
    legend=:outertopright,
    left_margin=10mm,
    size=(900, 550)
);
for (strategy_name, recorders) in backtest_results
    plot!(plt,
        axes(get_records(recorders[:wealth]), 1), get_records(recorders[:wealth]).data;
        label=strategy_name,
    )
end
plt

## CIs

using Bootstrap
using Statistics
using DataFrames
using RecipesBase
using Intervals

# Mean CIs
n_boot = 1000
cil = 0.95

function bootstrap_ci(f, data, boot_method, ci_method)
    bs = bootstrap(f, data, boot_method)
    return confint(bs, ci_method)
end

function ci_dataframe(
    metrics::AbstractArray, backtest_results
)
    return DataFrame(
        map(keys(backtest_results), values(backtest_results)) do strategy_name, recorders
            df_row = Dict{Symbol,Any}(
                Symbol(metric) => Interval(
                    bootstrap_ci(metric, get_records(recorders[:returns]).data, BasicSampling(n_boot), BasicConfInt(cil))[1][2:3]...
                ) for metric in metrics
            )
            df_row[:strategy] = strategy_name
            return df_row
        end,
    )
end

@userplot plot_cis
@recipe function f(plot::plot_cis; ci_df::AbstractArray)
    ci_df = plot.args

    metrics = setdiff(names(ci_df), ["strategy"])
    strategy_labels = ci_df[:, :strategy]
    num_metrics = length(metrics)
    num_cols = floor(Int, sqrt(num_metrics))
    num_rows = ceil(Int, num_metrics / float(num_cols))
    layout --> (num_rows, num_cols)

    ys = 1:0.1:(1 + (0.1) * (size(ci_df, 1)-1))
    yticks --> (ys, strategy_labels)
    ylims --> (0.9, ys[end] + 0.1)
    xrotation --> 45

    for (i, col) in enumerate(Symbol.(metrics))
        title := col
        label := ""
        subplot := i
        @series begin
            ci_df[:, col], ys
        end
    end
end

function expected_shortfall(returns; risk_level::Real=0.05)
    last_index = floor(Int, risk_level * length(returns))
    return -mean(partialsort(returns, 1:last_index))
end

function conditional_sharpe(returns; risk_level::Real=0.05)
    cvar = - expected_shortfall(returns; risk_level=risk_level)

    r̄ = mean(returns)

    return r̄ ./ (r̄ .- cvar)
end

ci_df = ci_dataframe([mean, expected_shortfall, conditional_sharpe], backtest_results)
plt = plot(plot_cis(ci_df),
    size=(900, 600)
)
