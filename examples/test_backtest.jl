using COSMO
using Distributions
using Plots
using PortfolioOpt
using PortfolioOpt.TestUtils: 
    backtest_po, get_test_data, mean_variance, 
    percentchange, timestamp, rename!

############ Read Prices #############
prices = get_test_data()
numD, numA = size(prices) # A: Assets    D: Days

############# Calculating returns #####################
returns_series = percentchange(prices)

# risk free asset
# rf =  fill(3.2e-4, numD)

############# backtest Parameters #####################
DEFAULT_SOLVER = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

start_date = timestamp(returns_series)[100]

## Strategies
# Look back days
k_back = 60

############# backtest with limit return strategies #####################

wealth_markowitz_limit_R, strategy_returns =
    backtest_po(returns_series; start_date=start_date) do past_returns, current_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        
        # Parameters
        R = 0.0015 / current_wealth
        formulation = MeanVariance(;
            predicted_mean = r̄,
            predicted_covariance = Σ,
        )
        
        # model
        model = po_min_variance_limit_return(formulation, R; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * current_wealth
    end;
rename!(wealth_markowitz_limit_R, :markowitz_limit_R);

wealth_markowitz_infeasible_limit_R, strategy_returns =
    backtest_po(returns_series; start_date=start_date) do past_returns, current_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        
        # Parameters
        R = 0.005 / current_wealth
        formulation = MeanVariance(;
            predicted_mean = r̄,
            predicted_covariance = Σ,
        )
        
        # model
        model = po_min_variance_limit_return(formulation, R; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * current_wealth
    end;
rename!(wealth_markowitz_infeasible_limit_R, :markowitz_infeasible_limit_R);

wealth_soyster_limit_R, strategy_returns =
    backtest_po(returns_series; start_date=start_date) do past_returns, current_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        
        # Parameters
        R = 0.0015 / current_wealth
        formulation = RobustBertsimas(;
            predicted_mean = r̄,
            predicted_covariance = Σ,
            uncertainty_delta = std(returns[(end - k_back):end, :]; dims=1)'[:,1] / 5,
            bertsimas_budjet = Float64(numA),
        )
        
        # model
        model = po_min_variance_limit_return(formulation, R; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * current_wealth
    end;
rename!(wealth_soyster_limit_R, :soyster_limit_R);

wealth_bertsimas_limit_R, strategy_returns =
    backtest_po(returns_series; start_date=start_date) do past_returns, current_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        
        # Parameters
        R = 0.0015 / current_wealth
        formulation = RobustBertsimas(;
            predicted_mean = r̄,
            predicted_covariance = Σ,
            uncertainty_delta = std(returns[(end - k_back):end, :]; dims=1)'[:,1] / 5,
            bertsimas_budjet = 3.0,
        )
        
        # model
        model = po_min_variance_limit_return(formulation, R; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * current_wealth
    end;
rename!(wealth_bertsimas_limit_R, :bertsimas_limit_R);

wealth_bental_limit_R, strategy_returns =
backtest_po(returns_series; start_date=start_date) do past_returns, current_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        
        # Parameters
        R = 0.0015 / current_wealth
        formulation = RobustBenTal(;
            predicted_mean = r̄,
            predicted_covariance = Σ,
            uncertainty_delta = 0.025
        )
        
        # model
        model = po_min_variance_limit_return(formulation, R; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * current_wealth
    end;
rename!(wealth_bental_limit_R, :bental_limit_R);

############# backtest with limit variance strategies #####################

wealth_markowitz_limit_var, returns_markowitz_limit_var =
backtest_po(returns_series; start_date=start_date) do past_returns, current_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ_s, r̄_s = mean_variance(returns[(end - 14):end, :])
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        
        # Parameters
        max_risk = 0.8
        formulation = MeanVariance(;
            predicted_mean = r̄_s,
            predicted_covariance = Σ,
        )
        
        # model
        model = po_max_return_limit_variance(formulation, max_risk; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * current_wealth
    end;
rename!(wealth_markowitz_limit_var, :markowitz_limit_var);

wealth_soyster_limit_var, returns_soyster_limit_var =
    backtest_po(returns_series; start_date=start_date) do past_returns, current_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ_s, r̄_s = mean_variance(returns[(end - 14):end, :])
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        
        # Parameters
        max_risk = 0.8
        formulation = RobustBertsimas(;
            predicted_mean = r̄_s,
            predicted_covariance = Σ,
            uncertainty_delta = std(returns[(end - k_back):end, :]; dims=1)'[:,1] / 3,
            bertsimas_budjet = Float64(numA),
        )
        
        # model
        model = po_max_return_limit_variance(formulation, max_risk; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * current_wealth
    end;
rename!(wealth_soyster_limit_var, :soyster_limit_var);

wealth_bertsimas_2_limit_var, returns_bertsimas_2_limit_var =
    backtest_po(returns_series; start_date=start_date) do past_returns, current_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ_s, r̄_s = mean_variance(returns[(end - 14):end, :])
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        
        # Parameters
        max_risk = 0.8
        formulation = RobustBertsimas(;
            predicted_mean = r̄_s,
            predicted_covariance = Σ,
            uncertainty_delta = std(returns[(end - k_back):end, :]; dims=1)'[:,1] / 3,
            bertsimas_budjet = 2.0,
        )
        
        # model
        model = po_max_return_limit_variance(formulation, max_risk; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * current_wealth
    end;
rename!(wealth_bertsimas_2_limit_var, :bertsimas_2_limit_var);

wealth_bertsimas_4_limit_var, returns_bertsimas_4_limit_var =
    backtest_po(returns_series; start_date=start_date) do past_returns, current_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ_s, r̄_s = mean_variance(returns[(end - 14):end, :])
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        
        # Parameters
        max_risk = 0.8
        formulation = RobustBertsimas(;
            predicted_mean = r̄_s,
            predicted_covariance = Σ,
            uncertainty_delta = std(returns[(end - k_back):end, :]; dims=1)'[:,1] / 3,
            bertsimas_budjet = 4.0,
        )
        
        # model
        model = po_max_return_limit_variance(formulation, max_risk; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * current_wealth
    end;
rename!(wealth_bertsimas_4_limit_var, :bertsimas_4_limit_var);

wealth_bental_limit_var, returns_bental_limit_var =
    backtest_po(returns_series; start_date=start_date) do past_returns, current_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ_s, r̄_s = mean_variance(returns[(end - 14):end, :])
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        
        # Parameters
        max_risk = 0.8
        formulation = RobustBenTal(;
            predicted_mean = r̄_s,
            predicted_covariance = Σ,
            uncertainty_delta = 0.028
        )
        
        # model
        model = po_max_return_limit_variance(formulation, max_risk; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * current_wealth
    end;
rename!(wealth_bental_limit_var, :bental_limit_var);

############# backtest with Data-Driven RO strategies #####################
wealth_betina, returns_betina =
    backtest_po(returns_series; start_date=start_date) do past_returns, current_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        # Parameters
        j_robust = 25
        R = -0.015 / current_wealth
        num_train = 20
        kst = 14
        klt = 30
        kmom = 20
        Q = 2

        formulation = RobustBetina(;
            sampled_returns = returns[(end - j_robust):end, :]
        )

        # Predict return 
        r_bar_t = zeros(numA)
        for i in 1:numA
            r_bar_t[i] = mixed_signals_predict_return(
                returns[:, i], num_train, kst, klt, kmom, Q
            )
        end
        
        # model
        model = po_max_predicted_return_limit_return(formulation, R; predicted_mean = r_bar_t)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * current_wealth
    end;
rename!(wealth_betina, :betina);

############# backtest with DRO strategies #####################

wealth_delague, returns_delague_limit_var =
    backtest_po(returns_series; start_date=start_date) do past_returns, current_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ_s, r̄_s = mean_variance(returns[(end - 14):end, :])
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        # Parameters
        # a =  [0.020646446303211906; 0.01456965970660176; 0.011268181574938444; 0.009189943838602721; 0.007760228688512961; 0.006716054241470283; 0.005919840492702679; 0.005292563196132596; 0.004785576102825234; 0.004367285293323039; 0.004367285293323039]
        # b = [12351.417663268794; 11949.933298064285; 11797.83904955914; 11743.66244416854; 11734.986173090612; 11749.533056761096; 11776.549758689594; 11810.379744602318; 11847.862045043921; 11887.152715141867; 11887.152715141867]

        formulation = RobustDelague(;
            predicted_mean = r̄_s,
            predicted_covariance = Σ,
            γ1 = 0.0065,
            γ2 = 5.5,
            utility_coeficients = [1.0],
            utility_intercepts = [0.0],
        )

        model = po_max_utility_return(formulation)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * current_wealth
    end;
rename!(wealth_delague, :delague);

############# extra strategies #####################

wealth_sharpe, returns_sharpe =
    backtest_po(returns_series; start_date=start_date) do past_returns, current_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        # Parameters
        x = max_sharpe(Σ, r̄, rf)
        return x * current_wealth
    end;
rename!(wealth_sharpe, :sharpe);

wealth_equal_weights, returns_equal =
    backtest_po(returns_series; start_date=start_date) do past_returns, current_wealth, rf
        # Prep
        numD, numA = size(past_returns)
        # Parameters
        x = equal_weights(numA)
        return x * current_wealth
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
