# API reference

```@meta
CurrentModule = PortfolioOpt
```

## Formulations

```@docs
PortfolioFormulation
ObjectiveTerm
RiskConstraint
portfolio_model!
uid
AbstractPortfolioTerm
calculate_measure!
```

## Risk measures

```@docs
PortfolioRiskMeasure
ambiguityset
ExpectedReturn
Variance
SqrtVariance
WorstCaseReturn
ConditionalExpectedReturn
alpha_quantile
ExpectedUtility
CertaintyEquivalent
utility
ConeRegularizer
```

## Utility functions

```@docs
ConcaveUtilityFunction
PieceWiseUtility
coefficients
intercepts
ExponentialUtility
risk_aversion
```

## Distributions and ambiguity sets

```@docs
AmbiguitySet
CenteredAmbiguitySet
distribution
DeterministicSamples
sample_size
BudgetSet
EllipticalSet
MomentUncertainty
DuWassersteinBall
PortfolioOpt.primal_cone
```

## Forecasters

```@docs
AbstractForecaster
forecast
EmpiricalGaussian
EmpiricalSamples
```

## Markets

```@docs
PortfolioMarket
num_assets
market_budget
set_market_budget
bids
set_bids!
change_bids!
clear_market!
calculate_profit
total_profit
market_model
build_portfolio_model
portfolio_weights
VolumeMarket
market_volume_fee
risk_free_rate
```

## Backtests

```@docs
MarketHistory
VolumeMarketHistory
past_prices
current_prices
market_template
num_days
sequential_backtest_market
StateRecorder
record!
get_records
```

## TestUtils

```@autodocs
Modules = [PortfolioOpt.TestUtils]
Private = false
```

## Index

```@index
Pages = ["api.md"]
```
