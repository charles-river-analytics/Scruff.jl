export Apply

"""
    Apply{J, O} <: SFunc{Tuple{SFunc{J, O}, J}, O}

Apply represents an sfunc that takes two groups of arguments.  The first group is a single
argument, which is an sfunc to apply to the second group of arguments.

# Additional supported operators
- `support`
- `support_quality`
- `sample`
- `logcpdf`
- `compute_pi`
- `send_lambda`

# Type parameters
- `J`: the input type of the *sfunc* that may be applied; that *sfunc* is the input type of the `Apply`
- `O`: the output type of the *sfunc* that may be applied, which is also the output type of the `Apply`
"""
struct Apply{J <: Tuple, O} <: SFunc{Tuple{SFunc{J, O}, J}, O}
end

@impl begin
    struct ApplySupport end
    function support(::Apply{J,O}, 
                    parranges::NTuple{N,Vector}, 
                    size::Integer, 
                    curr::Vector{<:O}) where {J<:Tuple,O,N}

        result = Vector{O}()
        for sf in parranges[1]
            append!(result, support(sf, (parranges[2],), size, curr))
        end
        return unique(result)
    end
end

@impl begin
    struct ApplySupportQuality end

    function support_quality(::Apply{J,O}, parranges) where {J,O}
        q = support_quality_rank(:CompleteSupport)
        for sf in parranges[1]
            imp = get_imp(MultiInterface.get_policy(), Support, sf, parranges[2], 0, O[])
            q = min(q, support_quality_rank(support_quality(imp, sf, parranges[2])))
        end
        return support_quality_from_rank(q)
    end
end

@impl begin
    struct ApplySample end
    function sample(::Apply{J,O}, input::Tuple{SFunc{J,O}, J})::O where {J<:Tuple,O}
        return sample(input[1], input[2])
    end
end

@impl begin
    struct ApplyLogcpdf end
    function logcpdf(::Apply{J,O}, i::Tuple{SFunc{J,O}, J}, o::O)::AbstractFloat where {J<:Tuple,O}
        return logcpdf(i[1], i[2], o)
    end
end

# WARNING: THIS LOGIC DOES NOT WORK WITH MORE THAN ONE PARENT
@impl begin
    struct ApplyComputePi end
    function compute_pi(::Apply{J,O},
                     range::__OptVec{<:O}, 
                     parranges::NTuple{N,Vector}, 
                     incoming_pis::Tuple)::Dist{<:O} where {N,J<:Tuple,O}

        sfrange = parranges[1]
        argsrange = parranges[2]
        sfpi = incoming_pis[1]
        argspi = incoming_pis[2]
        result = zeros(Float64, length(range))
        for sf in sfrange
            p1 = cpdf(sfpi, (), sf)
            p2 = compute_pi(sf, range, (argsrange,), (argspi,))
            p3 = [p1 * cpdf(p2, (), x) for x in range]
            result .+= p3
        end
        return Cat(range, result)
    end
end

# WARNING: THIS LOGIC DOES NOT WORK WITH MORE THAN ONE PARENT
@impl begin
    struct ApplySendLambda end

    function send_lambda(::Apply{J,O},
                       lambda::Score{<:O},
                       range::__OptVec{<:O},
                       parranges::NTuple{N,Vector},
                       incoming_pis::Tuple,
                       parent_idx::Integer)::Score where {N,J<:Tuple,O}
        @assert parent_idx == 1 || parent_idx == 2
        sfrange = parranges[1]
        argsrange = parranges[2]
        sfpi = incoming_pis[1]
        argspi = incoming_pis[2]

        if parent_idx == 2
            # For each x, we must sum over the sfunc argument compute P(y|x) for each possible sfunc
            result = Vector{Float64}()
            for args in argsrange
                resultpieces = Vector{Float64}()
                for sf in sfrange
                    sp = logcpdf(sfpi, (), sf)
                    for y in range
                        a = isa(args, Tuple) ? args : tuple(args)
                        push!(resultpieces, sp + logcpdf(sf, a, y) + get_log_score(lambda, y))
                    end
                end
                push!(result, logsumexp(resultpieces))
            end
            return LogScore(argsrange, result)

        else # parent_idx == 1
            # This is simpler; we must sum over the arguments, which is achieved by the embedded compute_pi
            result = Vector{Float64}()
            for sf in sfrange
                resultpieces = Vector{Float64}()
                ypi = compute_pi(sf, range, (argsrange,), (argspi,))
                for y in range
                    push!(resultpieces, logcpdf(ypi, (), y) + get_log_score(lambda, y))
                end
                push!(result, logsumexp(resultpieces))
            end
            return LogScore(sfrange, result)
        end
    end
end
