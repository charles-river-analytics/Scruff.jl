import Base.length

using Distributions: Categorical
using StatsFuns: logsumexp
using ..SFuncs: Cat

export 
    Particles,
    effective_sample_size,
    normalize_weights,
    log_prob_evidence,
    resample,
    probability,
    marginal

"Sample = Dict{Symbol, Any}"    
Sample = Dict{Symbol, Any}

"""
    struct Particles

A structure of particles containing a vector of `Samples` and of log_weights. 
"""
struct Particles
    samples::Vector{Sample}
    log_weights::Vector{Float64}
end

"""
    probability(parts::Particles, predicate::Sample -> Bool)::Float64

    Returns the probability that the predicate is satisfied
"""
function probability(parts::Particles, predicate::Function)::Float64
    sum = 0.0
    tot = 0.0
    for i in 1:length(parts.samples)
        w = exp(parts.log_weights[i])
        s = parts.samples[i]
        if predicate(s)
            sum += w
        end
        tot += w
    end
    return sum / tot
end

"""
    marginal(parts::Particles, x::Symbol)::Cat

Returns a Cat representing the marginal distribution over the given symbol according to parts
"""
function marginal(parts::Particles, x::Symbol)::Cat
    d = Dict{Any, Float64}()
    xs = [s[x] for s in parts.samples]
    lws = normalize_weights(parts.log_weights)
    for i in 1:length(parts.samples)
        d[xs[i]] = get(d, xs[i], 0.0) + exp(lws[i])
    end
    ks = Vector{Any}()
    ps = Vector{Float64}()
    for (k,p) in d
        push!(ks, k)
        push!(ps, p)
    end
    return Cat(ks, ps)
end

"""
    expected_sample_size(log_weights::Vector{Float64})

Effective sample size
"""
function effective_sample_size(lws::Vector{Float64})
    log_ess = (2 * logsumexp(lws)) - logsumexp(2 .* lws)
    return exp(log_ess)
end

"""
    normalize_weights(log_weights::Vector{Float64})

Normalize weights
"""
function normalize_weights(lws::Vector{Float64})
    return lws .- logsumexp(lws)
end

function log_prob_evidence(lws::Vector{Float64})
    return logsumexp(lws) - log(length(lws))
end

function resample(ps::Particles, target_num_particles::Int = length(ps.samples))
    lnws = normalize_weights(ps.log_weights)
    weights = exp.(lnws)
    selections = rand(Categorical(weights/sum(weights)), target_num_particles)
    samples = map(selections) do ind
        ps.samples[ind]
    end
    return Particles(samples, zeros(target_num_particles))
end
