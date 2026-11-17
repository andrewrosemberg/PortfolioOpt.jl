module PortfolioOptOptimalBidsExt

# The generic functions of OptimalBids accept any `PortfolioMarket`, so that code written against
# the OptimalBids market functions can drive the markets of PortfolioOpt.

import OptimalBids
using PortfolioOpt: PortfolioOpt, PortfolioMarket

OptimalBids.change_bids!(market::PortfolioMarket, new_bids::Vector) = PortfolioOpt.change_bids!(market, new_bids)
OptimalBids.clear_market!(market::PortfolioMarket) = PortfolioOpt.clear_market!(market)
OptimalBids.clear_market!(market::PortfolioMarket, clearing_prices) = PortfolioOpt.clear_market!(market, clearing_prices)
OptimalBids.calculate_profit(market::PortfolioMarket) = PortfolioOpt.calculate_profit(market)

end
