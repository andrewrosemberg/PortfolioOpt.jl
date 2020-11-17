using Distributions
using MarketData
using COSMO

DEFAULT_SOLVER = optimizer_with_attributes(
    COSMO.Optimizer, 
    "verbose" => false, 
    "max_iter" => 5000
)

## Get data
function get_test_data(;
    start_date = Date(2009, 9, 1),
    end_date = start_date + Year(1) + Month(3)
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

## Prep data
"""
Mean and Variance of returns
"""
function mean_variance(returns)
    r̄ = mean(returns,dims=1)'
    Σ = cov(returns)
    return Σ,r̄
end

function returns_montecarlo(Σ,r̄, numS)
    d = MvNormal(r̄, Σ)
    r = rand(d, numS)
    P = pdf(d,r)
    return r',P/sum(P)
end

## Prep solution 
function compute_solution(model::JuMP.Model, w; solver = DEFAULT_SOLVER)
    set_optimizer(model, solver)
    optimize!(model)
    w_values = value.(w)
    if sum(w_values) > 1.0
        w_values = w_values / sum(w_values)
    end
    r = sum(r̄'w_values)
    return w_values, objective_value(model), r
end

function compute_solution_dual(model::JuMP.Model, w; solver = DEFAULT_SOLVER)
    set_optimizer(model, solver)
    optimize!(model)
    w_values = value.(w)
    if sum(w_values) > 1.0
        w_values = w_values / sum(w_values)
    end
    return w_values, objective_value(model), value(model[:E])
end

function compute_solution_stoc(model::JuMP.Model, w; solver = DEFAULT_SOLVER)
    set_optimizer(model, solver)
    optimize!(model)
    w_values = value.(w)
    if sum(w_values) > 1.0
        w_values = w_values / sum(w_values)
    end
    Cvar = -objective_value(model)
    r = sum(r̄'w_values)
    q1_α = value.(model[:z])
    return w_values, r, Cvar, q1_α
end

function compute_solution_stoc_2(model::JuMP.Model, w; solver = DEFAULT_SOLVER)
    set_optimizer(model, solver)
    optimize!(model)
    w_values = value.(w)
    if sum(w_values) > 1.0
        w_values = w_values / sum(w_values)
    end
    r = objective_value(model)
    q1_α = value.(model[:z])
    return w_values, r, q1_α
end

# Create base PO model
function base_model(numA::Integer)
    model = Model()
    w = @variable(model, w[i=1:numA])
    @variable(model, sum_invested)
    # @constraint(model, [sum_invested; w] in MOI.NormOneCone(length(w) + 1)) # for no stock renting
    @constraint(model, sum_invested == sum(w))
    @constraint(model, sum_invested == 1)
    return model, w
end
