@testset "Robust Formulations" begin
    numA = 3
    d = generate_gaussian_distribution(numA, rng)
    
    @testset "BudgetSet" begin
        @testset "constructor" begin
            @test BudgetSet(d) isa BudgetSet
            @test BudgetSet(d; Δ=ones(numA)) isa BudgetSet
            @test BudgetSet(d; Δ=ones(numA), Γ=2.0) isa BudgetSet
        end
        ambiguity_set = BudgetSet(d)
        @testset "distribution" begin
            @test PortfolioOpt.distribution(ambiguity_set) == d
        end
        @testset "ExpectedReturn" begin
            measure = ExpectedReturn(ambiguity_set, WorstCase)
            model = Model(DEFAULT_SOLVER)
            w = @variable(model, [1:numA])
            fix.(w, ones(numA)/numA)
            er_pointer = PortfolioOpt.calculate_measure!(measure, w)

            @objective(model, Max, er_pointer)
            JuMP.optimize!(model)

            @test isapprox(value(er_pointer), (mean(d) - sqrt.(var(d)) ./ 5)' * value.(w), atol=1e-4)
        end
    end
end