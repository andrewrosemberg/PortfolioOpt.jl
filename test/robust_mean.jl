@testset "Robust Formulations" begin
    numA = 3
    d = generate_gaussian_distribution(numA, rng)
    
    @testset "BudgetSet" begin
        @testset "constructor" begin
            @test BudgetSet(d) isa BudgetSet
            @test BudgetSet(d; Δ=ones(numA)) isa BudgetSet
            @test BudgetSet(d; Δ=ones(numA), Γ=2.0) isa BudgetSet
            @test_throws ArgumentError BudgetSet(d; Γ=-2.0)
            @test_throws ArgumentError BudgetSet(d; Δ=-ones(numA))
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
    @testset "EllipticalSet" begin
        @testset "constructor" begin
            @test EllipticalSet(d) isa EllipticalSet
            @test EllipticalSet(d; Δ=0.1) isa EllipticalSet
            @test_throws ArgumentError EllipticalSet(d; Δ=-0.1)
        end
        ambiguity_set = EllipticalSet(d)
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

            @test isapprox(value(er_pointer), 
                mean(d)' * value.(w) - sqrt(value.(w)' * cov(d) * value.(w)) * ambiguity_set.Δ, atol=1e-4
            )
        end
    end
end