using COSMO
using Distributions
using Plots
using PortfolioOpt
using PortfolioOpt.TestUtils
plotly()

############ Read Prices #############
prices = get_test_data()
numD, numA = size(prices) # A: Assets    D: Days

############# Calculating returns #####################
returns_series = percentchange(prices)

# market_history = VolumeMarketHistory(returns_series)

############# backtest Parameters #####################
DEFAULT_SOLVER = optimizer_with_attributes(
    COSMO.Optimizer, "verbose" => false, "max_iter" => 900000
)

date_range = timestamp(returns_series)[100:end]

## Strategies parameters
# Look back days
k_back = 60
# results
backtest_results = Dict()

############# backtest strategies #####################

backtest_results["markowitz_limit_R"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    # Parameters
    d = MvNormal(r̄, Σ)
    R = 0.0015 / market_budget(market)

    formulation = PortfolioFormulation(
        ObjectiveTerm(SqrtVariance(d)),
        RiskConstraint(ExpectedReturn(d), GreaterThan(R)),
        MIN_SENSE
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

backtest_results["markowitz_limit_var"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    # Parameters
    d = MvNormal(r̄, Σ)
    V = 0.003 / market_budget(market)

    formulation = PortfolioFormulation(
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(SqrtVariance(d), LessThan(V)),
        MAX_SENSE
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

backtest_results["normal_limit_soyster"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    # Parameters
    d = MvNormal(r̄, Σ)
    R = -0.05 / market_budget(market)

    formulation = PortfolioFormulation(
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(ExpectedReturn(BudgetSet(d, maximum(abs.(returns); dims=1)'[:] .- r̄, numA * 1.0), WorstCase), GreaterThan(R)),
        MAX_SENSE
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

backtest_results["normal_limit_bertsimas_30"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    # Parameters
    d = MvNormal(r̄, Σ)
    R = -0.05 / market_budget(market)

    formulation = PortfolioFormulation(
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(ExpectedReturn(BudgetSet(d, maximum(abs.(returns); dims=1)'[:] .- r̄, numA * 0.3), WorstCase), GreaterThan(R)),
        MAX_SENSE
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

backtest_results["normal_limit_bertsimas_60"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    # Parameters
    d = MvNormal(r̄, Σ)
    R = -0.05 / market_budget(market)

    formulation = PortfolioFormulation(
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(ExpectedReturn(BudgetSet(d, maximum(abs.(returns); dims=1)'[:] .- r̄, numA * 0.6), WorstCase), GreaterThan(R)),
        MAX_SENSE
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end
#TODO
backtest_results["normal_limit_bental"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    Σ, r̄ = mean_variance(returns[(end - k_back):end, :])
    
    # Parameters
    d = MvNormal(r̄, Σ)
    R = -0.05 / market_budget(market)

    formulation = PortfolioFormulation(
        ObjectiveTerm(ExpectedReturn(d)),
        RiskConstraint(ExpectedReturn(EllipticalSet(d, 0.1), WorstCase), GreaterThan(R)),
        MAX_SENSE
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

plt = plot(;title="Culmulative Wealth",
    xlabel="Time",
    ylabel="Wealth",
    legend=:outertopright,
);
for (strategy_name, recorders) in backtest_results
    plot!(plt, 
        axes(get_records(recorders[:wealth]), 1), get_records(recorders[:wealth]).data;
        label=strategy_name,
    )
end
plt

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
