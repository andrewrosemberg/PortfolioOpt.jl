struct DeterministicSamples{T<:Real} <: ContinuousMultivariateSampleable
    samples::Array{T,2}
    num_samples::Int
end

DeterministicSamples(samples, num_samples=size(samples, 2)) = DeterministicSamples(samples, num_samples)

Base.length(s::DeterministicSamples) = size(s.samples, 1)
Base.size(s::DeterministicSamples) = (length(s), s.num_samples)
Base.size(s::DeterministicSamples, dim::Int) = dim == 1 ? length(s) : s.num_samples
sample_size(s::DeterministicSamples) = s.num_samples
samples_probability(s::DeterministicSamples) = fill(1.0/sample_size(s), sample_size(s))

"""circular index"""
function cidx(i::Int, n::Int)
    return mod(i, n) == 0 ? n : mod(i, n)
end

function Distributions.rand(
    s::DeterministicSamples{T}, num_requested_samples::Int
) where {T<:Real}
    samples = s.samples
    x = Array{T, 2}(undef, length(s), num_requested_samples)
    num_samples = size(samples, 2)
    num_first_samples = min(num_requested_samples, num_samples)
    x[:, 1:num_first_samples] = samples[:, end-num_first_samples+1:end]
    num_last_samples = num_requested_samples - num_first_samples
    if num_last_samples > 0
        x[:, num_first_samples+1:end] = samples[:, cidx.(1:num_last_samples, num_samples)]
    end
    return x
end

function Distributions.rand(
    s::DeterministicSamples{T}, ::Nothing
) where {T<:Real}
    return rand(s, s.num_samples)
end

Distributions.rand(s::DeterministicSamples, num_requested_samples=nothing) = rand(s, num_requested_samples)
Distributions.rand(::AbstractRNG, s::DeterministicSamples{<:Real}, num_requested_samples::Int) = rand(s, num_requested_samples)
Distributions.rand(::AbstractRNG, s::DeterministicSamples{<:Real}, num_requested_samples::Nothing) = rand(s, num_requested_samples)

mean(s::DeterministicSamples) = Statistics.mean(s.samples, dims=2)[:,1]
cov(s::DeterministicSamples) = Statistics.cov(s.samples, dims=2)
