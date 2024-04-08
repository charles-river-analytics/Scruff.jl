export
    Generate


"""
    Generate{O} <: SFunc{Tuple{Dist{O}}, O}

Generate a value from its `Dist` argument.

This helps in higher-order programming. A typical pattern will be to create an sfunc that produces a `Dist`,
and then generate many observations from the `Dist` using `Generate`.

# Additional supported operators
- `support`
- `support_quality`
- `sample`
- `logcpdf`
- `compute_pi`
- `send_lambda`
- `make_factors`
"""
struct Generate{O} <: SFunc{Tuple{Dist{O}}, O}
end

@impl begin
    struct GenerateSupport end
    function support(::Generate{O}, 
                    parranges::NTuple{N,Vector}, 
                    size::Integer, 
                    curr::Vector{<:O}) where {O,N}

        result = Vector{O}()
        for sf in parranges[1]
            append!(result, support(sf, (), size, curr))
        end
        return unique(result)
    end
end

@impl begin
    struct GenerateSupportQuality end

    function support_quality(::Generate{O}, parranges) where O
        q = support_quality_rank(:CompleteSupport)
        for sf in parranges[1]
            imp = get_imp(MultiInterface.get_policy(), Support, sf, (), 0, O[])
            q = min(q, support_quality_rank(support_quality(imp, sf, ())))
        end
        return support_quality_from_rank(q)
    end
end

@impl begin
    struct GenerateSample end
    function sample(::Generate{O}, input::Tuple{<:Dist{O}})::O where O
        return sample(input[1], ())
    end
end

@impl begin
    struct GenerateLogcpdf end
    function logcpdf(::Generate{O}, i::Tuple{<:Dist{O}}, o::O)::AbstractFloat where O
        return logcpdf(i[1], (), o)
    end
end

# WARNING: THIS LOGIC DOES NOT WORK WITH MORE THAN ONE PARENT
@impl begin
    struct GenerateComputePi end
    function compute_pi(::Generate{O},
                     range::__OptVec{<:O}, 
                     parranges::NTuple{N,Vector}, 
                     incoming_pis::Tuple)::Dist{<:O} where {N,O}

        sfrange = parranges[1]
        sfpi = incoming_pis[1]
        result = zeros(Float64, length(range))
        for sf in sfrange
            p1 = cpdf(sfpi, (), sf)
            p2 = compute_pi(sf, range, (), ())
            p3 = [p1 * cpdf(p2, (), x) for x in range]
            result .+= p3
        end
        return Cat(range, result)
    end
end

# WARNING: THIS LOGIC DOES NOT WORK WITH MORE THAN ONE PARENT
@impl begin
    struct GenerateSendLambda end

    function send_lambda(::Generate{O},
                       lambda::Score{<:O},
                       range::__OptVec{<:O},
                       parranges::NTuple{N,Vector},
                       incoming_pis::Tuple,
                       parent_idx::Integer)::Score where {N,O}
        @assert parent_idx == 1
        sfrange::Vector{typeof(parranges[1][1])} = parranges[1]
        sfpi = incoming_pis[1]
        resultprobs = Vector{Float64}()
        for sf in sfrange
            resultpieces = Vector{Float64}()
            ypi = compute_pi(sf, range, (), ())
            for y in range
                push!(resultpieces, logcpdf(ypi, (), y) + get_log_score(lambda, y))
            end
            push!(resultprobs, logsumexp(resultpieces))
        end
        result :: LogScore{typeof(sfrange[1])} = LogScore(sfrange, resultprobs)
        return result
    end
end

@impl begin
    struct GenerateMakeFactors end

    function make_factors(::Generate{O}, range::Vector{<:O}, parranges::Tuple{<:Vector{<:Dist{O}}}, id::Int, parids::Tuple{Int}) where O
        dims = (length(parranges[1]), length(range))
        keys = (parids[1], id)
        entries = Float64[]
        for u in parranges[1]
            for x in range
                p = cpdf(u, (), x)
                push!(entries, p)
            end
        end
        fact = Factor(dims, keys, entries)
        return ([fact], [fact])
    end
end
