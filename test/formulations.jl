@testset "formulations.jl" begin
    @testset "PieceWiseUtility" begin
        @testset "constructor" begin
            @test PieceWiseUtility() isa PieceWiseUtility
            @test PieceWiseUtility([1.0]) isa PieceWiseUtility
            @test PieceWiseUtility([1.0], [0.0]) isa PieceWiseUtility
        end
        @testset "coefficients" begin
            @test coefficients(PieceWiseUtility()) == [1.0]
            @test coefficients(PieceWiseUtility([1.0])) == [1.0]
            @test coefficients(PieceWiseUtility([1.0], [0.0])) == [1.0]
            @test coefficients(PieceWiseUtility([1.0, 2.0], [0.0, 1.0])) == [1.0, 2.0]
        end
        @testset "intercepts" begin
            @test intercepts(PieceWiseUtility()) == [0.0]
            @test intercepts(PieceWiseUtility([1.0])) == [0.0]
            @test intercepts(PieceWiseUtility([1.0], [0.0])) == [0.0]
            @test intercepts(PieceWiseUtility([1.0, 2.0], [0.0, 1.0])) == [0.0, 1.0]
        end
    end
    @testset "PortfolioRiskMeasure" begin
        Σ = rand(2, 2)
        Σ = Σ' * Σ
        d = MvNormal(rand(2), Σ)
        @testset "ExpectedReturn" begin
            @testset "constructor" begin
                @test ExpectedReturn(d) isa ExpectedReturn
                @test ExpectedReturn(d, EstimatedCase) isa ExpectedReturn
                @test ExpectedReturn(d, WorstCase) isa ExpectedReturn
            end
            @testset "ambiguityset" begin
                @test ambiguityset(ExpectedReturn(d)) == d
                @test ambiguityset(ExpectedReturn(d, EstimatedCase)) == d
                @test ambiguityset(ExpectedReturn(d, WorstCase)) == d
            end
        end
        @testset "Variance" begin
            @testset "constructor" begin
                @test Variance(d) isa Variance
                @test Variance(d, EstimatedCase) isa Variance
                @test Variance(d, WorstCase) isa Variance
            end
            @testset "ambiguityset" begin
                @test ambiguityset(Variance(d)) == d
                @test ambiguityset(Variance(d, EstimatedCase)) == d
                @test ambiguityset(Variance(d, WorstCase)) == d
            end
        end
        @testset "SqrtVariance" begin
            @testset "constructor" begin
                @test SqrtVariance(d) isa SqrtVariance
                @test SqrtVariance(d, EstimatedCase) isa SqrtVariance
                @test SqrtVariance(d, WorstCase) isa SqrtVariance
            end
            @testset "ambiguityset" begin
                @test ambiguityset(SqrtVariance(d)) == d
                @test ambiguityset(SqrtVariance(d, EstimatedCase)) == d
                @test ambiguityset(SqrtVariance(d, WorstCase)) == d
            end
        end
        @testset "ConditionalExpectedReturn" begin
            numS = 100
            α = 0.05
            cexp1 = ConditionalExpectedReturn(d, numS)
            cexp2 = ConditionalExpectedReturn(d, numS; α=0.01, R=WorstCase)
            @testset "constructor" begin
                @test cexp1 isa ConditionalExpectedReturn
                @test cexp2 isa ConditionalExpectedReturn
            end
            @testset "ambiguityset" begin
                @test ambiguityset(cexp1) == d
                @test ambiguityset(cexp2) == d
            end
            @testset "sample_size" begin
                @test sample_size(cexp1) == numS
                @test sample_size(cexp2) == numS
            end
            @testset "alpha_quantile" begin
                @test alpha_quantile(cexp1) == α
                @test alpha_quantile(cexp2) == 0.01
            end
        end
        @testset "ExpectedUtility" begin
            @testset "constructor" begin
                @test ExpectedUtility(d, PieceWiseUtility()) isa ExpectedUtility
                @test ExpectedUtility(d, PieceWiseUtility(), EstimatedCase) isa ExpectedUtility
                @test ExpectedUtility(d, PieceWiseUtility(), WorstCase) isa ExpectedUtility
            end
            @testset "ambiguityset" begin
                @test ambiguityset(ExpectedUtility(d, PieceWiseUtility())) == d
                @test ambiguityset(ExpectedUtility(d, PieceWiseUtility(), EstimatedCase)) == d
                @test ambiguityset(ExpectedUtility(d, PieceWiseUtility(), WorstCase)) == d
            end
            @testset "utility" begin
                @test utility(ExpectedUtility(d, PieceWiseUtility())) == PieceWiseUtility()
                @test utility(ExpectedUtility(d, PieceWiseUtility(), EstimatedCase)) == PieceWiseUtility()
                @test utility(ExpectedUtility(d, PieceWiseUtility(), WorstCase)) == PieceWiseUtility()
            end
        end
    end
end