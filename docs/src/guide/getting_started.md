# Getting started

## Installation

```julia
using Pkg
Pkg.add("PortfolioOpt")
Pkg.add("Clarabel") # or any conic solver supported by JuMP
```

PortfolioOpt builds JuMP models and needs a solver. The formulations use linear, second order,
exponential and semidefinite cones; [Clarabel](https://github.com/oxfordcontrol/Clarabel.jl)
supports all of them (SCS and Mosek do as well).

## A first portfolio

Given a ``T \times N`` matrix of past returns (one row per period, one column per asset), a
forecaster gives a distribution of the next period returns, a [`PortfolioFormulation`](@ref)
describes the problem, and [`portfolio_weights`](@ref) solves it on a long only market with a
budget of 1:

```@example getting_started
using PortfolioOpt, Clarabel, Distributions, LinearAlgebra, Random
Random.seed!(1)

# synthetic daily returns of 4 assets
true_returns = MvNormal([0.0010, 0.0008, 0.0012, 0.0005], Diagonal([0.020, 0.015, 0.025, 0.010] .^ 2))
returns = Matrix(rand(true_returns, 250)')

d = forecast(EmpiricalGaussian(120), returns) # MvNormal of the last 120 days
optimizer = optimizer_with_attributes(Clarabel.Optimizer, "verbose" => false)

markowitz = PortfolioFormulation(MAX_SENSE,
    ObjectiveTerm(ExpectedReturn(d)),                 # maximize the expected return
    RiskConstraint(SqrtVariance(d), LessThan(0.012)), # daily standard deviation at most 1.2%
)
w = portfolio_weights(markowitz, 4, optimizer)
```

The money that is not invested stays in the risk free asset:

```@example getting_started
sum(w), sqrt(w' * cov(d) * w)
```

## Changing the measure

The same pattern works with any measure and distribution. With the empirical distribution of the
last 120 days, limit the average return of the 5% worst days instead of the standard deviation:

```@example getting_started
s = forecast(EmpiricalSamples(120), returns) # DeterministicSamples, one column per day

cvar = PortfolioFormulation(MAX_SENSE,
    ObjectiveTerm(ExpectedReturn(s)),
    RiskConstraint(ConditionalExpectedReturn(s; α=0.05), GreaterThan(-0.02)),
)
portfolio_weights(cvar, 4, optimizer)
```

or protect the expected return against errors in the estimated mean with an ambiguity set:

```@example getting_started
robust = PortfolioFormulation(MAX_SENSE,
    ObjectiveTerm(ExpectedReturn(MomentUncertainty(d; γ1=0.05, γ2=1.5))), # worst case mean
    RiskConstraint(SqrtVariance(d), LessThan(0.012)),
)
portfolio_weights(robust, 4, optimizer)
```

The [Theory](../theory/problem.md) pages define each measure and set.

## Adding constraints

[`build_portfolio_model`](@ref) returns the JuMP model before it is solved, so any JuMP
constraint can be added (here, at most 40% per asset), and
[`change_bids!`](@ref) solves it and stores the decision in the market:

```@example getting_started
market = VolumeMarket(4; budget=1.0)
model, w, _ = build_portfolio_model(market, markowitz, optimizer)
@constraint(model, w .<= 0.4)
change_bids!(market, model, w)
bids(market)
```

## Short selling, fees and the risk free rate

[`VolumeMarket`](@ref) has options for short selling (the budget then limits the 1-norm of the
positions), a fee per unit invested and the return of the money that is not invested; they are
also keyword arguments of [`portfolio_weights`](@ref):

```@example getting_started
portfolio_weights(markowitz, 4, optimizer; allow_short_selling=true, volume_fee=0.0001, risk_free_rate=0.0002)
```

## Next steps

* [Concepts](concepts.md): how formulations, markets and backtests fit together.
* [Extending PortfolioOpt](extending.md): new measures, ambiguity sets, forecasters and markets.
* The examples: backtests of stochastic, robust and distributionally robust strategies, and how to
  assess them.
