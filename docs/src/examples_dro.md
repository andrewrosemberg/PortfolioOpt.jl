# Example Wasserstein DRO

## No Ambiguity

Example of backtest with Mean-Variance strategy with a simple empirical forecaster.

```@example Backtest
using Gurobi
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
    Gurobi.Optimizer, "OutputFlag" => 0
)

date_range = timestamp(returns_series)[100:end];

# Backtest
backtest_results = Dict()

backtest_results["EP"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    returns = values(past_returns)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ExpectedReturn(DeterministicSamples(returns'[:,:]))),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

```

## NingNing Du Wasserstein: fixed budget

```@example Backtest
ϵ=0.005

backtest_results["wasserstein_fixed_budget_$(ϵ)"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    
    # Parameters
    j_robust = numD
    
    d = DeterministicSamples(returns'[:,:])
    s = DuWassersteinBall(d; ϵ=ϵ)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ConditionalExpectedReturn{WorstCase}(1.0, s, j_robust))
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

backtest_results["wasserstein_fixed_budget_$(ϵ)"][:wealth]

```

## NingNing Du Wasserstein: light tail

```@example Backtest

backtest_results["wasserstein_light_tail"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    
    # Parameters
    j_robust = numD
    δ = 0.1
    ϵ = log(1/δ)/j_robust
    
    d = DeterministicSamples(returns'[:,:])
    s = DuWassersteinBall(d; ϵ=ϵ)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ConditionalExpectedReturn{WorstCase}(1.0, s, j_robust))
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

```

## NingNing Du Wasserstein: risk limit

```@example Backtest

backtest_results["EP_limit_wasserstein"], _ = sequential_backtest_market(
    VolumeMarketHistory(returns_series), date_range,
) do market, past_returns, ext
    # Prep
    numD, numA = size(past_returns)
    returns = values(past_returns)
    
    # Parameters
    j_robust = numD
    δ = 0.1
    ϵ = log(1/δ)/j_robust
    R = -0.001 / market_budget(market)
    
    d = DeterministicSamples(returns'[:,:])
    s = DuWassersteinBall(d; ϵ=ϵ)

    formulation = PortfolioFormulation(MAX_SENSE,
        ObjectiveTerm(ConditionalExpectedReturn{WorstCase}(1.0, DuWassersteinBall(d; ϵ=0.005), j_robust)),
        RiskConstraint(ConditionalExpectedReturn{WorstCase}(1.0, s, j_robust), GreaterThan(R)),
    )
    
    pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
    return pointers
end

backtest_results["EP_limit_wasserstein"][:wealth]

```

## NingNing Du Wasserstein: multiple risk limit

```@example Backtest
ρ_max_range = 3:2:9

for ρ_max in ρ_max_range
    backtest_results["EP_limit_wasserstein_$(ρ_max)_cons"], _ = sequential_backtest_market(
        VolumeMarketHistory(returns_series), date_range,
    ) do market, past_returns, ext
        # Prep
        numD, numA = size(past_returns)
        returns = values(past_returns)
        
        # Parameters
        j_robust = numD
        ϵ=0.01
        R = -0.001 / market_budget(market)
        
        d = DeterministicSamples(returns'[:,:])

        formulation = PortfolioFormulation(MAX_SENSE,
            ObjectiveTerm(ConditionalExpectedReturn{WorstCase}(1.0, DuWassersteinBall(d; ϵ=0.005), j_robust)),
            [
                RiskConstraint(ConditionalExpectedReturn{WorstCase}(1.0, DuWassersteinBall(d; ϵ=ϵ * ρ), j_robust), GreaterThan(R * ρ))
                for ρ in 1:ρ_max
            ]
        )
        
        pointers = change_bids!(market, formulation, DEFAULT_SOLVER)
        return pointers
    end
end

```

## Culmulative Plot

```@example Backtest
using Plots
using Plots.PlotMeasures

plt = plot(;title="Culmulative Wealth",
    xlabel="Time",
    ylabel="Wealth",
    legend=:outertopright,
    left_margin=10mm,
    size=(900, 600)
);
for (strategy_name, recorders) in backtest_results
    plot!(plt, 
        axes(get_records(recorders[:wealth]), 1), get_records(recorders[:wealth]).data;
        label=strategy_name,
    )
end
plt
```

## CIs

```@example Backtest
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
```