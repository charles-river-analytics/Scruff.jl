export
    Invertible

"""
    struct Invertible{I,O} <: SFunc{Tuple{I},O,Nothing}

An invertible sfunc, with both a `forward` and a `inverse` function provided.

# Additional supported operators
- `support`
- `support_quality`
- `sample`
- `logcpdf`
- `bounded_probs`
- `make_factors`
- `compute_pi`
- `send_lambda`
"""
struct Invertible{I,O} <: SFunc{Tuple{I},O}
    forward :: Function
    inverse :: Function
end

@impl begin
    struct InvertibleSupport end
    function support(sf::Invertible{I,O}, parranges, size, curr) where {I,O}
        invsup = map(sf.forward, parranges[1])
        return invsup
    end
end

@impl begin
    struct InvertibleSupportQuality end

    function support_quality(::Invertible{I,O}, parranges) where {I,O}
        return :CompleteSupport
    end
end

@impl begin
    struct InvertibleSample end
    function sample(sf::Invertible{I,O}, i::Tuple{I}) where {I,O}
        return sf.forward(i[1])
    end
end

@impl begin
    struct InvertibleLogcpdf end
    function logcpdf(sf::Invertible{I,O}, i::Tuple{I}, o::O) where {I,O}
        return sf.forward(i[1]) == o ? 0.0 : -Inf
    end
end

@impl begin
    struct InvertibleBoundedProbs end

    function bounded_probs(sf::Invertible{I,O}, range::Vector{<:O}, 
            parranges) where {I,O}
        result = Float64[]
        for i in parranges[1]
            for o in range
                p = sf.forward(i) == o ? 1.0 : 0.0
                push!(result, p)
            end
        end
        return (result, result)
    end
end

@impl begin
    struct InvertibleMakeFactors end

    function make_factors(sf::Invertible{I,O}, range::Vector{<:O}, parranges, 
            id, parids) where {I,O}
        dims = (length(parranges[1]), length(range))
        keys = (parids[1], id)
        entries = bounded_probs(sf, range, parranges)[1]
        facts = [Factor(dims, keys, entries)]
        return (facts, facts)
    end
end

@impl begin
    struct InvertibleComputePi end

    function compute_pi(sf::Invertible{I,O}, range::Vector{<:O}, parranges, 
            incoming_pis) where {I,O}
        ps = [cpdf(incoming_pis[1], (), sf.inverse(o)) for o in range]
        return Cat(range, ps)
    end
end

@impl begin
    struct InvertibleSendLambda end

    function send_lambda(sf::Invertible{I,O}, lambda, range, parranges, incoming_pis, 
            parent_idx) where {I,O}
        @assert parent_idx == 1
        ls = [get_log_score(lambda, sf.forward(x)) for x in parranges[1]]
        return LogScore(parranges[1], ls)
    end
end
#=
@impl begin
    struct InvertibleInitialStats end
    initial_stats(::Invertible) = nothing
end

@impl begin
    struct InvertibleAccumulateStats end
    accumulate_stats(::Invertible, existing_stats, new_stats) = nothing
end

@impl begin
    struct InvertibleExpectedStats end
    expected_stats(::Invertible, range, parranges, incoming_pis, child_lambda) = 
        nothing
end

@impl begin
    struct InvertibleMaximizeStats end
    maximize_stats(::Invertible, stats) = nothing
end
=#
            






