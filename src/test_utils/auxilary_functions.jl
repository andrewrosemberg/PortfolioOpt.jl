"""
Get test data (Prices) from MarketData.
"""
function get_test_data(;
    start_date=Date(2009, 9, 1), end_date=start_date + Year(1) + Month(3)
)
    df_AAPL = rename(to(from(AAPL[:Close], start_date), end_date), :AAPL)
    df_BA = rename(to(from(BA[:Close], start_date), end_date), :BA)
    df_DELL = rename(to(from(DELL[:Close], start_date), end_date), :DELL)
    df_CAT = rename(to(from(CAT[:Close], start_date), end_date), :CAT)
    df_EBAY = rename(to(from(EBAY[:Close], start_date), end_date), :EBAY)
    df_F = rename(to(from(F[:Close], start_date), end_date), :F)
    df = merge(df_AAPL, df_BA, df_DELL, df_CAT, df_EBAY, df_F)
    return df
end

"""
Mean and Variance of returns
"""
function mean_variance(returns; digits::Union{Nothing,Int}=nothing)
    r̄ = mean(returns; dims=1)'[:,1]
    Σ = cov(returns)
    if !isnothing(digits)
        return round.(Σ, digits=digits), round.(r̄, digits=digits)
    end
    return Σ, r̄
end
