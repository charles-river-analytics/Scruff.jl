export
  Det,
  ExplicitDet

import StatsBase
import StatsFuns

"""
    abstract type Det{I, O} <: SFunc{I, O}
`Det` defines an *sfunc* that represents a deterministic function `I -> O`.  When
a `Det` is subtyped, a function `apply(d::Det, i::I)::O` or `apply(d::Det, i::I...)::O`
must also be implemented.  It has no parameters.
# Additional supported operators
- `support`
- `support_quality`
- `sample`
- `logcpdf`
- `bounded_probs`
- `make_factors`
- `compute_pi`
- `send_lambda`

# Type parameters
- `I`: the input type(s) of the `Det`
- `O`: the output type(s) of the `Det`
"""
abstract type Det{I, O} <: SFunc{I, O} end

"""
    struct ExplicitDet{I, O} <: Det{I, O}
`ExplicitDet` is a `Det` that contains a field `f::Function`.  It also has an `apply` method
that simply delegates to the `ExplicitDet`'s function:
```
    apply(d::ExplicitDet, i...) = d.f(i...)
```
```
julia> d = ExplicitDet{Tuple{Vararg{Real}},Real}(sum)
ExplicitDet{Tuple{Vararg{Real, N} where N}, Real}(sum)
```
"""
struct ExplicitDet{I, O} <: Det{I, O}
    f :: Function
end

apply(d::ExplicitDet, i::Tuple) = d.f(i...)
apply(d::ExplicitDet, i...) = d.f(i...)

function Det(I, O, f)
    return ExplicitDet{I,O}(f)
end

@impl begin
    struct DetSupport end

    function support(sf::Det{I,O}, 
                    parranges::NTuple{N,Vector}, 
                    size::Integer, 
                    curr::Vector{<:O}) where {I,O,N}

        ps = cartesian_product(parranges)
        result = unique([apply(sf, p...) for p in ps])

        if (length(result) > size) # do downsampling but ensure that curr is included in the range
            # include curr in the result, it is OK if it > size
            if (!isempty(curr)) # curr is provided
                curr = unique(curr)
                curr_size= length(curr)
                if(curr_size >= size)
                    result = curr
                else
                    sample_size = size - curr_size # number of samples
                    result_without_curr =  setdiff(result, curr) # what is being sampled
                    result = StatsBase.sample(result_without_curr, sample_size, replace = false) # samples
                    result = vcat(curr, result) # concatanate curr and samples
                end
            else # curr is not provided
                result = StatsBase.sample(result, size, replace = false) # samples
            end
        end

        sort!(result)
        return result
    end
end

@impl begin
    struct DetSupportQuality end

    function support_quality(::Det, parranges)
        :CompleteSupport
    end
end

@impl begin
    struct DetSample end
    function sample(sf::Det{I,O}, x::I)::O where {I,O}
        if isa(x, Tuple)
            apply(sf, x...)
        else
            apply(sf, x)
        end
    end
end

@impl begin
    struct DetLogcpdf end
    function logcpdf(sf::Det{I,O}, inp::I, o::O)::AbstractFloat where {I,O}
        y = apply(sf, inp...)
        return y == o ? 0.0 : -Inf
    end        
end


# We cannot define complete for Det in an abstract way, because it depends on whether the ranges of the parents
# is complete.
@impl begin
    struct DetBoundedProbs end

    function bounded_probs(sf::Det{I,O}, 
                         range::__OptVec{<:O}, 
                         parranges::NTuple{N,Vector})::
                Tuple{Vector{<:AbstractFloat}, Vector{<:AbstractFloat}} where {I,O,N}
        ps = cartesian_product(parranges)
        result = Float64[]
        for p in ps
            x = apply(sf, p...)
            for r in range
                push!(result, x == r ? 1.0 : 0.0)
            end
        end
        return (result, result)
    end
end

@impl begin
    struct DetMakeFactors end

    function make_factors(sf::Det{I,O},
                        range::__OptVec{<:O}, 
                        parranges::NTuple{N,Vector}, 
                        id, 
                        parids::Tuple)::Tuple{Vector{<:Scruff.Utils.Factor}, Vector{<:Scruff.Utils.Factor}} where {I,O,N}

        entries = bounded_probs(sf, range, parranges)
        keys = [i for i in parids]
        push!(keys, id)
        keys = Tuple(keys)
        dims = [length(r) for r in parranges]
        push!(dims, length(range))
        dims = Tuple(dims)
        factors = [Factor(dims, keys, entries[1])]
        return (factors, factors)
    end
end

@impl begin
    struct DetComputePi end

    function compute_pi(sf::Det{I,O},
                     range::__OptVec{<:O}, 
                     parranges::NTuple{N,Vector}, 
                     incoming_pis::Tuple)::Dist{<:O} where {N,I,O}

        pinds = cartesian_product([collect(1:length(r)) for r in parranges])
        result = zeros(Float64, length(range))
        result_pieces = [Vector{Float64}() for i in range]
        prng = 1:length(parranges)
        for pind in pinds
            xs = [parranges[i][pind[i]] for i in prng]
            x = apply(sf, xs...)
            idx = indexin([x], range)[1]
            if !isnothing(idx)
                pi = sum([logcpdf(incoming_pis[i], (), xs[i]) for i in prng])
                push!(result_pieces[idx], pi)
            end
        end
        result = [exp(StatsFuns.logsumexp(pieces)) for pieces in result_pieces]
        return Cat(range, normalize(result))
    end
end

@impl begin
    struct DetSendLambda end

    function send_lambda(sf::Det{I,O},
                       lambda::Score{<:O},
                       range::__OptVec{<:O},
                       parranges::NTuple{N,Vector},
                       incoming_pis::Tuple,
                       parent_ix::Integer)::Score where {N,I,O}

        # For a particular parent, we consider all possible values of the other parents.
        # We make a joint argument p, and compute pi(other parents) * lambda(f(p)).
        # Need to make sure the target parent range is a Vector{T} rather than a Vector{Any}
        T = typejoin([typeof(x) for x in parranges[parent_ix]]...)
        target_parrange :: Vector{T} = parranges[parent_ix]
        otherranges = [r for r in parranges]
        deleteat!(otherranges, parent_ix)
        otherinds = cartesian_product([collect(1:length(r)) for r in otherranges])
        result_pieces = [Vector{Float64}() for u in target_parrange]

        for i in 1:length(target_parrange)
            parval = target_parrange[i]
            for otherind in otherinds
                fullval = []
                pi = 0.0
                for j = 1:parent_ix - 1
                    u = parranges[j][otherind[j]]
                    push!(fullval, u)
                    pi += logcpdf(incoming_pis[j], (), u)            
                end
                push!(fullval, parval)
                for j = parent_ix + 1:length(parranges)
                    u = parranges[j][otherind[j-1]]
                    push!(fullval, u)
                    pi += logcpdf(incoming_pis[j], (), u)
                end
                x = apply(sf, fullval...)
                idx = indexin([x], range)[1]
                if !isnothing(idx)
                    push!(result_pieces[i], pi + get_log_score(lambda, range[idx]))
                end
            end
        end
        result = [StatsFuns.logsumexp(pieces) for pieces in result_pieces]
        return LogScore(target_parrange, result)
    end
end
#=
@impl begin
    struct DetInitialStats end
    initial_stats(sf::Det) = nothing
end
@impl begin
    struct DetAccumulateStats end
    accumulate_stats(sf::Det, existing_stats, new_stats) = nothing
end
@impl begin
    struct DetExpectedStats end
    function expected_stats(sf::Det{I,O},
                          range::__OptVec{<:O}, 
                          parranges::NTuple{N,Vector},
                          pis::NTuple{M,Dist},
                          child_lambda::Score{<:O}) where {I,O,N,M}
        nothing
    end
end
@impl begin
    struct DetMaximizeStats end
    maximize_stats(sf::Det, stats) = nothing
end
=#

