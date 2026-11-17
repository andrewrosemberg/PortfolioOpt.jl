module PortfolioOptMarketDataExt

using MarketData
using MarketData.Dates: Date, Year, Month
import PortfolioOpt.TestUtils: get_test_data

function get_test_data(;
    start_date::Date=Date(2009, 9, 1), end_date::Date=start_date + Year(1) + Month(3)
)
    df_AAPL = rename(to(from(MarketData.AAPL[:Close], start_date), end_date), :AAPL)
    df_BA = rename(to(from(MarketData.BA[:Close], start_date), end_date), :BA)
    df_DELL = rename(to(from(MarketData.DELL[:Close], start_date), end_date), :DELL)
    df_CAT = rename(to(from(MarketData.CAT[:Close], start_date), end_date), :CAT)
    df_EBAY = rename(to(from(MarketData.EBAY[:Close], start_date), end_date), :EBAY)
    df_F = rename(to(from(MarketData.F[:Close], start_date), end_date), :F)
    return merge(df_AAPL, df_BA, df_DELL, df_CAT, df_EBAY, df_F)
end

end
