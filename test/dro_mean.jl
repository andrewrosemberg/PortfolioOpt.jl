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
            measure = ExpectedUtility(ambiguity_set, PieceWiseUtility())
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
    @testset "DuWassersteinBall" begin
        s = DeterministicSamples(rand(rng, d, 100))
        @testset "constructor" begin
            @test DuWassersteinBall(s, 0.1, 1000.0, rand(numA, numA), 2.0) isa DuWassersteinBall
            @test DuWassersteinBall(s; ϵ=0.02,
                norm_cone=Inf,
                Λ=100.0,
                Q=rand(numA, numA)
            ) isa DuWassersteinBall
            @test_throws ArgumentError DuWassersteinBall(s, -0.1, 1000.0, rand(numA, numA), 2.0)
            @test_throws ArgumentError DuWassersteinBall(s, 0.1, -1000.0, rand(numA, numA), 2.0)
            @test_throws ArgumentError DuWassersteinBall(s, 0.1, 1000.0, rand(5, numA), 3.0)
        end
        @testset "distribution" begin
            @test PortfolioOpt.distribution(DuWassersteinBall(s)) == s
        end
        @testset "sample_size" begin
            @test PortfolioOpt.sample_size(DuWassersteinBall(s)) == PortfolioOpt.sample_size(s)
        end
        @testset "ExpectedReturn" begin
            measure = ExpectedReturn(DuWassersteinBall(s))
            model = Model(DEFAULT_SOLVER)
            w = @variable(model, [1:numA])
            fix.(w, ones(numA)/numA)
            er_pointer = PortfolioOpt.calculate_measure!(measure, w)

            @objective(model, Max, er_pointer)
            JuMP.optimize!(model)

            @test value(er_pointer) <= mean(s)' * value.(w)
        end
    end
end