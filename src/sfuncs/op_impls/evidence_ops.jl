#=
    OBSOLETE - SUBSUMED BY SCORE

    evidence_ops.jl : Support for operators that process evidence and interventions
=#

export
    evidence_message,
    evidence_entry,
    intervention_message,
    intervention_entry,
    NoiseEvidence
import LinearAlgebra

function evidence_entry(hard_evidence::T, value::T) where {T}
    return hard_evidence == value ? 1.0 : 0.0
end

function evidence_entry(soft_evidence::Dict{T, Float64}, value::T) where {T}
    return get(soft_evidence, value, 0.0)
end

struct NoiseEvidence
    mean ::Union{Array{Float64,1}, Float64}
    std ::Union{Array{Float64,2}, Float64}
end

function evidence_entry(noise_evid::NoiseEvidence, value::Float64)
    val = 1.0/(noise_evid.std * sqrt(2*pi)) * exp(-0.5 * ((noise_evid.mean - value)/noise_evid.std)^2)
    return val
end

function evidence_entry(noise_evid::NoiseEvidence, value::Array{Float64,1})
    k = length(value)
    val = exp.(-0.5 * transpose(value - noise_evid.mean) * LinearAlgebra.inv(noise_evid.std) * (value - noise_evid.mean))/sqrt((2*pi)^k * LinearAlgebra.det(noise_evid.std))
    return val
end

function evidence_entry(f::Function , value) ::Float64 # return f(v), has to be Float64
    return f(value)
end

function intervention_entry(hard_evidence::T, value::T) where {T}
    return hard_evidence == value ? 1.0 : 0.0
end

function evidence_message(::SFunc{I,O,P}, range::Vector{O}, evidence::Score{O})::Score{O} where {I,O,P}
    n = length(range)
    ps = Array{Float64, 1}(undef, n)
    for i = 1:n
        ps[i] = get_score(evidence, range[i])
    end
    return SoftScore{O}(range, ps)
end

function intervention_message(SFunc, range, intervention)
    n = length(range)
    ps = Array{Float64, 1}(undef, n)
    for i = 1:n
        ps[i] = intervention_entry(intervention, range[i])
    end
    return Cat(range, ps)
end
