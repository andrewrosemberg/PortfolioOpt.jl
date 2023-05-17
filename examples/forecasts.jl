
# Features extraction from returns proposed by Betina and al.
# KLT stands for the long-term period
# KST stands for the short-term period
# ϱ is the number of standard deviations above the means
function feature_indexes_sgns(asset_rtn_sgn, kst, klt, kmom, ϱ=2)
    #current time
    t = size(asset_rtn_sgn, 1)
    # sig_1 and sig_2 are
    # short-term and long-term simple moving averages
    sig_1 = (1 / kst) * sum(asset_rtn_sgn[t - d] for d in 0:(kst - 1))
    sig_2 = (1 / klt) * sum(asset_rtn_sgn[t - d] for d in 0:(klt - 1))

    # sig_3 and sig_4 are exponential moving average (short and long)
    sig_3 =
        (1.0 / sum(exp(-d / kst) for d in 1:kst)) *
        sum(exp(-d / kst) * asset_rtn_sgn[t - d] for d in 0:(kst - 1))
    sig_4 =
        (1.0 / sum(exp(-d / klt) for d in 1:klt)) *
        sum(exp(-d / klt) * asset_rtn_sgn[t - d] for d in 0:(klt - 1))
    # RSI
    R_pos = 0
    R_neg = 0
    num_pos = 0
    num_neg = 0
    for i in (t - kmom + 1):t
        if (asset_rtn_sgn[i] > 0)
            R_pos = R_pos + asset_rtn_sgn[i]
            num_pos = num_pos + 1
        else
            R_neg = R_neg + asset_rtn_sgn[i]
            num_neg = num_neg + 1
        end
    end
    med_pos = R_pos / num_pos
    if num_neg == 0
        med_neg = 1
    else
        med_neg = abs(R_neg / num_neg)
    end
    sig_5 = med_pos / med_neg
    # estimated shor-t and long-t standard deviations
    σ_st_6 = sqrt((1.0 / kst) * sum((asset_rtn_sgn[t - d] - sig_1)^2 for d in 0:(kst - 1)))
    σ_lt_7 = sqrt((1.0 / klt) * sum((asset_rtn_sgn[t - d] - sig_2)^2 for d in 0:(klt - 1)))
    # sig_6 and sig_7 are

    if sig_1 <= 0
        sig_6 = sig_1 - ϱ * σ_st_6
    else
        sig_6 = sig_1 + ϱ * σ_st_6
    end
    if sig_2 <= 0
        sig_7 = sig_2 - ϱ * σ_lt_7
    else
        sig_7 = sig_2 + ϱ * σ_lt_7
    end
    σ_st_8 = sqrt((1.0 / kst) * sum((asset_rtn_sgn[t - d] - sig_3)^2 for d in 0:(kst - 1)))
    σ_lt_9 = sqrt((1.0 / klt) * sum((asset_rtn_sgn[t - d] - sig_4)^2 for d in 0:(klt - 1)))
    if sig_3 <= 0
        sig_8 = sig_3 - ϱ * σ_st_8
    else
        sig_8 = sig_3 + ϱ * σ_st_8
    end
    if sig_4 <= 0
        sig_9 = sig_4 - ϱ * σ_lt_9
    else
        sig_9 = sig_4 + ϱ * σ_lt_9
    end

    return [sig_1; sig_2; sig_3; sig_4; sig_5; sig_6; sig_7; sig_8; sig_9]
end

# Mixed signals predictor proposed by Betina and al. Univariate
function mixed_signals_predict_return(asset_rtn_sgn, num_t, kst_a, klt_a, kmom, ϱ=2)
    numD = size(asset_rtn_sgn, 1)
    num_train = min(numD - 1, num_t)

    ## optimize features weights (least squares based on past returns)
    # signal features extraction train (from klt+1 to klt+num_train)
    #kst = max(min(numD-num_train-10,kst_a),0)
    #klt = max(min(numD-num_train-10,klt_a),0)
    sig_features = zeros(num_train, 9) #each line is the i'th-element's features
    for i in 1:num_train
        #kst = max(min(numD-i-2,kst_a),0)
        klt = max(min(numD - i - 2, klt_a), 0)
        kst = Int64(floor(klt / 2.0))
        sig_features[num_train + 1 - i, :] = feature_indexes_sgns(
            asset_rtn_sgn[1:(numD - i)], kst, klt, kmom
        )
    end
    # optimize weights (least squares - pseudo inverse)
    try
        weights_opt = \(
            sig_features'sig_features,
            sig_features'asset_rtn_sgn[(numD - num_train + 1):numD],
        )
    catch
        weights_opt = pinv(sig_features, 1E-07) * asset_rtn_sgn[(numD - num_train + 1):numD]
    end
    #  normalize features weights
    weights_opt = weights_opt ./ sum(weights_opt)
    ## extract features current return
    kst = min(numD, kst_a)
    klt = min(numD, klt_a)
    features = feature_indexes_sgns(asset_rtn_sgn[1:numD], kst, klt, kmom)
    ## extrapolate return

    return (features'weights_opt)[1]
end
