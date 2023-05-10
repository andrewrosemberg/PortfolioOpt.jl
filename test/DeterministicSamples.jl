@testset "DeterministicSamples" begin
    @testset "rand(s, num_requested_samples)" begin
        s = DeterministicSamples([1.0 2.0 3.0; 4.0 5.0 6.0])
        @test rand(rng, s, 1) == [3.0; 6.0;;]
        @test rand(s, 1) == [3.0; 6.0;;]
        @test rand(s, 2) == [2.0 3.0; 5.0 6.0]
        @test rand(s, 3) == [1.0 2.0 3.0; 4.0 5.0 6.0]
        @test rand(s, 4) == [1.0 2.0 3.0 1.0; 4.0 5.0 6.0 4.0]
    end
    @testset "rand(s)" begin
        s = DeterministicSamples([1.0 2.0 3.0; 4.0 5.0 6.0])
        @test rand(s) == [1.0 2.0 3.0; 4.0 5.0 6.0]
        @test rand(s, nothing) == [1.0 2.0 3.0; 4.0 5.0 6.0]
        @test rand(rng, s, nothing) == [1.0 2.0 3.0; 4.0 5.0 6.0]
        s = DeterministicSamples([1.0 2.0 3.0; 4.0 5.0 6.0], 2)
        @test rand(s) == [2.0 3.0; 5.0 6.0]
    end
    @testset "mean" begin
        s = DeterministicSamples([1.0 2.0 3.0; 4.0 5.0 6.0])
        @test mean(s) == [2.0; 5.0]
    end
    @testset "cov" begin
        s = DeterministicSamples([1.0 2.0 3.0; 4.0 5.0 6.0])
        @test cov(s) == [1.0 1.0; 1.0 1.0]
    end
    @testset "samples_probability" begin
        s = DeterministicSamples([1.0 2.0 3.0; 4.0 5.0 6.0])
        @test PortfolioOpt.samples_probability(s) ≈ [1/3; 1/3; 1/3]
    end
    @testset "size and length" begin
        s = DeterministicSamples([1.0 2.0 3.0; 4.0 5.0 6.0])
        @test size(s) == (2, 3)
        @test size(s, 2) == 3
        @test size(s, 1) == 2
        @test length(s) == 2
        @test PortfolioOpt.sample_size(s) == 3
    end
end
