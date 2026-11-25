module PortfolioOptOptimalBidsExt

# The generic functions of OptimalBids accept any `PortfolioMarket`, so that code written against
# the OptimalBids market functions can drive the markets of PortfolioOpt.

using OptimalBids: OptimalBids
using PortfolioOpt: PortfolioOpt, PortfolioMarket

function OptimalBids.change_bids!(market::PortfolioMarket, new_bids::Vector)
    return PortfolioOpt.change_bids!(market, new_bids)
end
OptimalBids.clear_market!(market::PortfolioMarket) = PortfolioOpt.clear_market!(market)
function OptimalBids.clear_market!(market::PortfolioMarket, clearing_prices)
    return PortfolioOpt.clear_market!(market, clearing_prices)
end
function OptimalBids.calculate_profit(market::PortfolioMarket)
    return PortfolioOpt.calculate_profit(market)
end

end
