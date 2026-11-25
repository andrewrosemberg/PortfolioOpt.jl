module PortfolioOptTimeSeriesExt

using TimeSeries: TimeArray, timestamp
using PortfolioOpt: PortfolioOpt

# a `TimeArray` of returns (one column per asset) can be used as the history of a
# `VolumeMarketHistory`
PortfolioOpt._timestamps(x::TimeArray) = timestamp(x)

end
