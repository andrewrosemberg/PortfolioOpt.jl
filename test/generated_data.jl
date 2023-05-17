function generate_gaussian_distribution(numA, rng=Random.GLOBAL_RNG)
    Σ = rand(rng, numA, numA)
    Σ = Σ' * Σ
    μ = rand(rng, numA)
    d = MvNormal(μ, Σ)
    return d
end
