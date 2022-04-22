using COSMO
using Distributions
using Plots
using PortfolioOpt
using PortfolioOpt.TestUtils

############ Read Prices #############
prices = get_test_data()
numD, numA = size(prices) # A: Assets    D: Days

############# Calculating returns #####################
returns_series = percentchange(prices)

# risk free asset
# rf = fill(3.2e-4, numD)

market = VolumeMarket(numA)
market_history = VolumeMarketHistory(market, returns_series)

############# backtest Parameters #####################
DEFAULT_SOLVER = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

date_range = timestamp(returns_series)[100:end]

## Strategies
# Look back days
k_back = 60

############# backtest with limit return strategies #####################

wealth_markowitz_limit_R, strategy_returns = sequential_backtest_market(
    market_history; date_range=date_range
) do market, past_returns
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    # Parameters
    d = MvNormal(r̄, Σ)
    R = 0.0015 / market_budget(market)

    formulation = PortfolioFormulation(
        ObjectiveTerm(SqrtVariance(d)),
        RiskConstraint(ExpectedReturn(d), GreaterThan(R))
    )
    
    # model
    model, decision_variables = market_model(market, DEFAULT_SOLVER; sense=MIN_SENSE)
    portfolio_model!(model, formulation, decision_variables)

    change_bids!(market, model, decision_variables)
    return nothing
end;

plt = plot(
    collect(keys(wealth_markowitz_limit_R)), collect(values(wealth_markowitz_limit_R));
    title="Culmulative Wealth",
    xlabel="Time",
    ylabel="Wealth",
    legend=:outertopright,
)

wealth_markowitz_infeasible_limit_R, strategy_returns =
    sequential_backtest_market(returns_series; start_date=start_date) do past_returns, market_budget(market), rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        
        # Parameters
        R = 0.005 / market_budget(market)
        formulation = MeanVariance(;
            predicted_mean = r̄,
            predicted_covariance = Σ,
        )
        
        # model
        model = po_min_variance_limit_return(formulation, R; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * market_budget(market)
    end;
rename!(wealth_markowitz_infeasible_limit_R, :markowitz_infeasible_limit_R);

wealth_soyster_limit_R, strategy_returns =
    sequential_backtest_market(returns_series; start_date=start_date) do past_returns, market_budget(market), rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        
        # Parameters
        R = 0.0015 / market_budget(market)
        formulation = RobustBertsimas(;
            predicted_mean = r̄,
            predicted_covariance = Σ,
            uncertainty_delta = std(returns[(end - k_back):end, :]; dims=1)'[:,1] / 5,
            bertsimas_budget = Float64(numA),
        )
        
        # model
        model = po_min_variance_limit_return(formulation, R; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * market_budget(market)
    end;
rename!(wealth_soyster_limit_R, :soyster_limit_R);

wealth_bertsimas_limit_R, strategy_returns =
    sequential_backtest_market(returns_series; start_date=start_date) do past_returns, market_budget(market), rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        
        # Parameters
        R = 0.0015 / market_budget(market)
        formulation = RobustBertsimas(;
            predicted_mean = r̄,
            predicted_covariance = Σ,
            uncertainty_delta = std(returns[(end - k_back):end, :]; dims=1)'[:,1] / 5,
            bertsimas_budget = 3.0,
        )
        
        # model
        model = po_min_variance_limit_return(formulation, R; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * market_budget(market)
    end;
rename!(wealth_bertsimas_limit_R, :bertsimas_limit_R);

wealth_bental_limit_R, strategy_returns =
sequential_backtest_market(returns_series; start_date=start_date) do past_returns, market_budget(market), rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        
        # Parameters
        R = 0.0015 / market_budget(market)
        formulation = RobustBenTal(;
            predicted_mean = r̄,
            predicted_covariance = Σ,
            uncertainty_delta = 0.025
        )
        
        # model
        model = po_min_variance_limit_return(formulation, R; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * market_budget(market)
    end;
rename!(wealth_bental_limit_R, :bental_limit_R);

############# backtest with limit variance strategies #####################

wealth_markowitz_limit_var, returns_markowitz_limit_var =
sequential_backtest_market(returns_series; start_date=start_date) do past_returns, market_budget(market), rf
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
        return x * market_budget(market)
    end;
rename!(wealth_markowitz_limit_var, :markowitz_limit_var);

wealth_soyster_limit_var, returns_soyster_limit_var =
    sequential_backtest_market(returns_series; start_date=start_date) do past_returns, market_budget(market), rf
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
            bertsimas_budget = Float64(numA),
        )
        
        # model
        model = po_max_return_limit_variance(formulation, max_risk; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * market_budget(market)
    end;
rename!(wealth_soyster_limit_var, :soyster_limit_var);

wealth_bertsimas_2_limit_var, returns_bertsimas_2_limit_var =
    sequential_backtest_market(returns_series; start_date=start_date) do past_returns, market_budget(market), rf
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
            bertsimas_budget = 2.0,
        )
        
        # model
        model = po_max_return_limit_variance(formulation, max_risk; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * market_budget(market)
    end;
rename!(wealth_bertsimas_2_limit_var, :bertsimas_2_limit_var);

wealth_bertsimas_4_limit_var, returns_bertsimas_4_limit_var =
    sequential_backtest_market(returns_series; start_date=start_date) do past_returns, market_budget(market), rf
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
            bertsimas_budget = 4.0,
        )
        
        # model
        model = po_max_return_limit_variance(formulation, max_risk; rf = rf)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * market_budget(market)
    end;
rename!(wealth_bertsimas_4_limit_var, :bertsimas_4_limit_var);

wealth_bental_limit_var, returns_bental_limit_var =
    sequential_backtest_market(returns_series; start_date=start_date) do past_returns, market_budget(market), rf
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
        return x * market_budget(market)
    end;
rename!(wealth_bental_limit_var, :bental_limit_var);

############# backtest with Data-Driven RO strategies #####################
wealth_betina, returns_betina =
    sequential_backtest_market(returns_series; start_date=start_date) do past_returns, market_budget(market), rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        # Parameters
        j_robust = 25
        R = -0.015 / market_budget(market)
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
        return x * market_budget(market)
    end;
rename!(wealth_betina, :betina);

############# backtest with DRO strategies #####################

wealth_delague, returns_delague_limit_var =
    sequential_backtest_market(returns_series; start_date=start_date) do past_returns, market_budget(market), rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ_s, r̄_s = mean_variance(returns[(end - 14):end, :])
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        # Parameters
        # a =  [0.020646446303211906; 0.01456965970660176; 0.011268181574938444; 0.009189943838602721; 0.007760228688512961; 0.006716054241470283; 0.005919840492702679; 0.005292563196132596; 0.004785576102825234; 0.004367285293323039; 0.004367285293323039]
        # b = [12351.417663268794; 11949.933298064285; 11797.83904955914; 11743.66244416854; 11734.986173090612; 11749.533056761096; 11776.549758689594; 11810.379744602318; 11847.862045043921; 11887.152715141867; 11887.152715141867]

        formulation = MomentUncertainty(;
            predicted_mean = r̄_s,
            predicted_covariance = Σ,
            γ1 = 0.0065,
            γ2 = 5.5,
            utility_coeficients = [1.0],
            utility_intercepts = [0.0],
        )

        model = po_max_utility_return(formulation)
        x = compute_solution(model, DEFAULT_SOLVER)
        return x * market_budget(market)
    end;
rename!(wealth_delague, :delague);

############# extra strategies #####################

wealth_sharpe, returns_sharpe =
    sequential_backtest_market(returns_series; start_date=start_date) do past_returns, market_budget(market), rf
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
        # Parameters
        x = max_sharpe(Σ, r̄, rf)
        return x * market_budget(market)
    end;
rename!(wealth_sharpe, :sharpe);

wealth_equal_weights, returns_equal =
    sequential_backtest_market(returns_series; start_date=start_date) do past_returns, market_budget(market), rf
        # Prep
        numD, numA = size(past_returns)
        # Parameters
        x = equal_weights(numA)
        return x * market_budget(market)
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
    label="Delage",
);
scatter!(plt, [std(returns_betina)], [mean(returns_betina)]; label="Betina");
plt
