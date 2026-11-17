# Breaking changes in v2.0

Version 2.0 makes the package self-contained and light. The core now depends only on JuMP,
MathOptInterface, Distributions, PDMats, ProgressMeter and standard libraries. The integrations
with OptimalBids.jl, MarketData.jl and TimeSeries.jl are now
[package extensions](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)).
They load automatically when the other package is loaded.

| v1 | v2 | Migration |
|:--|:--|:--|
| `PortfolioMarket <: OptimalBids.Market` | `PortfolioMarket` has no supertype | Markets subtype [`PortfolioMarket`](@ref) as before. Code that dispatched on `OptimalBids.Market` to reach a `PortfolioMarket` should dispatch on `PortfolioMarket`. |
| `change_bids!`, `clear_market!`, `calculate_profit` were the functions of OptimalBids | They are functions of PortfolioOpt ([`change_bids!`](@ref), [`clear_market!`](@ref), [`calculate_profit`](@ref)) | Extend `PortfolioOpt.clear_market!` etc. When OptimalBids is loaded, `OptimalBids.change_bids!`, `OptimalBids.clear_market!` and `OptimalBids.calculate_profit` forward to them for any `PortfolioMarket`. |
| `using PortfolioOpt` re-exported OptimalBids (`profit_for_bid!`, `profit_curve!`, ...) | Not re-exported | `using OptimalBids` explicitly. |
| `PortfolioOpt.TestUtils` re-exported MarketData/TimeSeries (`percentchange`, `timestamp`, `TimeArray`, ...) | Not re-exported | `using MarketData` (or `using TimeSeries`) explicitly. |
| `get_test_data()` always available | Needs MarketData | `using MarketData`, then `get_test_data()` works as before. |
| `VolumeMarketHistory` of a `TimeArray` always supported | Needs TimeSeries (MarketData loads it) | `using TimeSeries` or `using MarketData`. Any collection indexed by its `keys` still works without them. |
| `ExpectedReturn(::MomentUncertainty)` was the nominal mean | Worst case expected return over the set | Use `ExpectedReturn(distribution(s))` for the nominal mean. |

A v1 backtest script usually only needs one extra line:

```julia
using PortfolioOpt
using PortfolioOpt.TestUtils
using MarketData # new: test data, TimeArray functions and the TimeArray support of VolumeMarketHistory
```
