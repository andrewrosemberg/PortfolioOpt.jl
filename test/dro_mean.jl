@testset "CenteredAmbiguitySet" begin
    numA = 3
    d = generate_gaussian_distribution(numA, rng)
    @testset "MomentUncertainty" begin
        @testset "constructor" begin
            @test MomentUncertainty(d, 0.1, 1.5) isa MomentUncertainty
            @test MomentUncertainty(d; γ1=0.1, γ2=1.5) isa MomentUncertainty
            @test_throws ArgumentError MomentUncertainty(d, -0.1, 1.5)
            @test_throws ArgumentError MomentUncertainty(d, 0.1, 0.1)
        end
        ambiguity_set = MomentUncertainty(d, 0.1, 1.5)
        @testset "distribution" begin
            @test PortfolioOpt.distribution(ambiguity_set) == d
        end
        @testset "ExpectedUtility" begin
            measure = ExpectedUtility(ambiguity_set, PieceWiseUtility(), WorstCase)
            model = Model(DEFAULT_SOLVER)
            w = @variable(model, [1:numA])
            fix.(w, ones(numA)/numA)
            er_pointer = PortfolioOpt.calculate_measure!(measure, w)

            @objective(model, Max, er_pointer)
            JuMP.optimize!(model)

            # has to be more concervative than the expected return under EllipticalSet
            @test value(er_pointer) <= mean(d)' * value.(w) - sqrt(value.(w)' * cov(d) * value.(w)) * ambiguity_set.γ1
        end
    end
end