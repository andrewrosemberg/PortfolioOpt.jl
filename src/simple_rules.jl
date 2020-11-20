"""
Maximize sharp coeficient alocation.
"""
function max_sharpe(Σ,r̄,rf)
    one = ones(size(r̄,1))
    invΣ = pinv(Σ, 1E-25)
    v = invΣ*(r̄.-one*rf)
    return v
end

"""
Equal weights.
"""
function equal_weights(numA)
    return fill(1/numA, numA)
end