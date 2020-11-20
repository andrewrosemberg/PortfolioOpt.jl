using Plots
using PortfolioOpt

include("./test/test_utils/auxilary_functions.jl")
include("./test/test_utils/backtest.jl")

############ Read Prices #############
Prices = get_test_data()
numD, numA = size(Prices) # A: Assets    D: Days

############# Calculating returns #####################
returns_series = percentchange(Prices)

# risk free asset
# rf =  fill(3.2e-4, numD)

############# backtest Parameters #####################
start_date = timestamp(returns_series)[100]

## Strategies
# Average stats
k_back = 60
# Betina forecast
num_train = 20
kst = 20
klt = 40
kmom = 20
Q = 2

############# backtest with limit R strategies #####################

wealth_markowitz_limit_R, strategy_returns =
    backtest_po(returns_series; start_date=start_date) do past_returns, max_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        # Parameters
        R = 0.0015
        model, w = base_model(numA; allow_borrow=false)
        po_minvar_limitmean_Rf!(model, w, Σ, r̄, R, rf, 1)
        x = compute_solution_backtest(model, w)
        return x * max_wealth
    end;
rename!(wealth_markowitz_limit_R, :markowitz_limit_R);

wealth_markowitz_infeasible_limit_R, strategy_returns =
    backtest_po(returns_series; start_date=start_date) do past_returns, max_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        # Parameters
        R = 0.005
        model, w = base_model(numA; allow_borrow=false)
        po_minvar_limitmean_Rf!(model, w, Σ, r̄, R, rf, 1)
        x = compute_solution_backtest(model, w)
        return x * max_wealth
    end;
rename!(wealth_markowitz_infeasible_limit_R, :markowitz_infeasible_limit_R);

wealth_soyster_limit_R, strategy_returns =
    backtest_po(returns_series; start_date=start_date) do past_returns, max_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        # Parameters
        R = 0.0015
        Δ = std(returns[(end - k_back):end, :]; dims=1)' / 5 # Defining the uncertainty set
        model, w = base_model(numA; allow_borrow=false)
        po_minvar_limitmean_robust_bertsimas!(model, w, Σ, r̄, rf, R, Δ, numA, 1)
        x = compute_solution_backtest(model, w)
        return x * max_wealth
    end;
rename!(wealth_soyster_limit_R, :soyster_limit_R);

wealth_bertsimas_limit_R, strategy_returns =
    backtest_po(returns_series; start_date=start_date) do past_returns, max_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        # Parameters
        R = 0.0015
        Δ = std(returns[(end - k_back):end, :]; dims=1)' / 5 # Defining the uncertainty set
        model, w = base_model(numA; allow_borrow=false)
        po_minvar_limitmean_robust_bertsimas!(model, w, Σ, r̄, rf, R, Δ, 3, 1)
        x = compute_solution_backtest(model, w)
        return x * max_wealth
    end;
rename!(wealth_bertsimas_limit_R, :bertsimas_limit_R);

wealth_bental_limit_R, strategy_returns =
    backtest_po(returns_series; start_date=start_date) do past_returns, max_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        # Parameters
        R = 0.0015
        δ = 0.025 # Defining the uncertainty set
        model, w = base_model(numA; allow_borrow=false)
        po_minvar_limitmean_robust_bental!(model, w, Σ, r̄, rf, R, δ, 1)
        x = compute_solution_backtest(model, w)
        return x * max_wealth
    end;
rename!(wealth_bental_limit_R, :bental_limit_R);

############# backtest with limit Var strategies #####################
max_risk = 0.8

wealth_markowitz_limit_var, returns_markowitz_limit_var =
    backtest_po(returns_series; start_date=start_date) do past_returns, max_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ_s, r̄_s = mean_variance(returns[(end - 14):end, :])
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        # Parameters
        model, w = base_model(numA; allow_borrow=false)
        po_maxmean_limitvar_Rf!(model, w, Σ, r̄_s, max_risk, rf, 1)
        x = compute_solution_backtest(model, w)
        return x * max_wealth
    end;
rename!(wealth_markowitz_limit_var, :markowitz_limit_var);

wealth_soyster_limit_var, returns_soyster_limit_var =
    backtest_po(returns_series; start_date=start_date) do past_returns, max_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ_s, r̄_s = mean_variance(returns[(end - 14):end, :])
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        # Parameters
        Δ = std(returns[(end - k_back):end, :]; dims=1)' / 3 # Defining the uncertainty set
        model, w = base_model(numA; allow_borrow=false)
        po_maxmean_limitvar_robust_bertsimas!(model, w, Σ, r̄_s, rf, max_risk, Δ, numA, 1)
        x = compute_solution_backtest(model, w)
        return x * max_wealth
    end;
rename!(wealth_soyster_limit_var, :soyster_limit_var);

wealth_bertsimas_2_limit_var, returns_bertsimas_2_limit_var =
    backtest_po(returns_series; start_date=start_date) do past_returns, max_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ_s, r̄_s = mean_variance(returns[(end - 14):end, :])
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        # Parameters
        Δ = std(returns[(end - k_back):end, :]; dims=1)' / 3 # Defining the uncertainty set
        model, w = base_model(numA; allow_borrow=false)
        po_maxmean_limitvar_robust_bertsimas!(model, w, Σ, r̄_s, rf, max_risk, Δ, 2, 1)
        x = compute_solution_backtest(model, w)
        return x * max_wealth
    end;
rename!(wealth_bertsimas_2_limit_var, :bertsimas_2_limit_var);

wealth_bertsimas_4_limit_var, returns_bertsimas_4_limit_var =
    backtest_po(returns_series; start_date=start_date) do past_returns, max_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ_s, r̄_s = mean_variance(returns[(end - 14):end, :])
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        # Parameters
        Δ = std(returns[(end - k_back):end, :]; dims=1)' / 3 # Defining the uncertainty set
        model, w = base_model(numA; allow_borrow=false)
        po_maxmean_limitvar_robust_bertsimas!(model, w, Σ, r̄_s, rf, max_risk, Δ, 4, 1)
        x = compute_solution_backtest(model, w)
        return x * max_wealth
    end;
rename!(wealth_bertsimas_4_limit_var, :bertsimas_4_limit_var);

wealth_bental_limit_var, returns_bental_limit_var =
    backtest_po(returns_series; start_date=start_date) do past_returns, max_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ_s, r̄_s = mean_variance(returns[(end - 14):end, :])
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        # Parameters
        δ = 0.028 # Defining the uncertainty set
        model, w = base_model(numA; allow_borrow=false)
        po_maxmean_limitvar_robust_bental!(model, w, Σ, r̄_s, rf, max_risk, δ, 1)
        x = compute_solution_backtest(model, w)
        return x * max_wealth
    end;
rename!(wealth_bental_limit_var, :bental_limit_var);

############# backtest with Data-Driven RO strategies #####################
wealth_betina, returns_betina =
    backtest_po(returns_series; start_date=start_date) do past_returns, max_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        # Parameters
        j_robust = 25
        λ = -0.001
        # Predict return 
        r_bar_t = zeros(numA)
        for i in 1:numA
            r_bar_t[i] = mixed_signals_predict_return(
                returns[:, i], num_train, kst, klt, kmom, Q
            )
        end
        # solve
        model, w = base_model(numA; allow_borrow=false)
        betina_robust(model, w, returns, r_bar_t, rf, λ; j_robust=j_robust)
        x = compute_solution_backtest(model, w)
        return x * max_wealth
    end;
rename!(wealth_betina, :betina);

############# backtest with DRO strategies #####################

wealth_delague, returns_delague_limit_var =
    backtest_po(returns_series; start_date=start_date) do past_returns, max_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ_s, r̄_s = mean_variance(returns[(end - 14):end, :])
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        # Parameters
        K = 1
        # a =  [0.020646446303211906; 0.01456965970660176; 0.011268181574938444; 0.009189943838602721; 0.007760228688512961; 0.006716054241470283; 0.005919840492702679; 0.005292563196132596; 0.004785576102825234; 0.004367285293323039; 0.004367285293323039]
        # b = [12351.417663268794; 11949.933298064285; 11797.83904955914; 11743.66244416854; 11734.986173090612; 11749.533056761096; 11776.549758689594; 11810.379744602318; 11847.862045043921; 11887.152715141867; 11887.152715141867]
        a = [1.0]
        b = [0.0]
        γ1 = 0.0065
        γ2 = 5.5
        model, w = base_model(numA; allow_borrow=false)
        po_maxmean_delague(model, w, r̄_s, Σ, a, b, γ1, γ2, K)
        x = compute_solution_backtest(model, w)
        return x * max_wealth
    end;
rename!(wealth_delague, :delague);

############# extra strategies #####################

wealth_sharpe, returns_sharpe =
    backtest_po(returns_series; start_date=start_date) do past_returns, max_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        # Parameters
        x = max_sharpe(Σ, r̄, rf)
        return x * max_wealth
    end;
rename!(wealth_sharpe, :sharpe);

wealth_equal_weights, returns_equal =
    backtest_po(returns_series; start_date=start_date) do past_returns, max_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        # Parameters
        x = equal_weights(numA)
        return x * max_wealth
    end;
rename!(wealth_equal_weights, :equal_weights);

############# plot results  #####################
## Wealth
plt = plot(
    wealth_sharpe;
    title="Culmulative Wealth",
    xlabel="Time",
    ylabel="Wealth",
    legend=:outertopright,
);
plot!(plt, wealth_equal_weights);
# plot!(plt, wealth_markowitz_limit_R);
# plot!(plt, wealth_soyster_limit_R);
# plot!(plt, wealth_bertsimas_limit_R);
# plot!(plt, wealth_bental_limit_R);
# plot!(plt, wealth_markowitz_infeasible);
plot!(plt, wealth_markowitz_limit_var);
plot!(plt, wealth_soyster_limit_var);
plot!(plt, wealth_bertsimas_2_limit_var);
plot!(plt, wealth_bertsimas_4_limit_var);
plot!(plt, wealth_bental_limit_var);
plot!(plt, wealth_delague);
plot!(plt, wealth_betina);
plt

## Stats
plt = scatter(
    [std(returns_sharpe)],
    [mean(returns_sharpe)];
    label="Sharpe",
    title="Mean Vs Std",
    xlabel="σ",
    ylabel="E[r]",
    legend=:outertopright,
);
scatter!(plt, [std(returns_equal)], [mean(returns_equal)]; label="Equal");
scatter!(
    plt,
    [std(returns_markowitz_limit_var)],
    [mean(returns_markowitz_limit_var)];
    label="Markowitz",
);
scatter!(
    plt,
    [std(returns_soyster_limit_var)],
    [mean(returns_soyster_limit_var)];
    label="Soyster",
);
scatter!(
    plt,
    [std(returns_bertsimas_2_limit_var)],
    [mean(returns_bertsimas_2_limit_var)];
    label="Bertsimas 2",
);
scatter!(
    plt,
    [std(returns_bertsimas_4_limit_var)],
    [mean(returns_bertsimas_4_limit_var)];
    label="Bertsimas 4",
);
scatter!(
    plt, [std(returns_bental_limit_var)], [mean(returns_bental_limit_var)]; label="Bental"
);
scatter!(
    plt,
    [std(returns_delague_limit_var)],
    [mean(returns_delague_limit_var)];
    label="Delague",
);
scatter!(plt, [std(returns_betina)], [mean(returns_betina)]; label="Betina");
plt
