#=
    unconditional.jl : General representation of sfuncs that don't depend on anything.
=#

export Unconditional

"""
    abstract type Unconditional{T} <: SFunc{Tuple{}, T}

`Unconditional` is a general representation of an *sfunc* that does not depend 
on anything.  It has no input.

# Type parameters
- `T`: the input type(s) of the `Unconditional`
- `P`: the parameter type(s) of the `Unconditional`
"""
abstract type Unconditional{T} <: SFunc{Tuple{}, T} end

@impl begin
    struct UnconditionalSample end
    function sample(sf::Unconditional{T}, i::Tuple{})::T
        sample(sf, x) # should never be called
    end
end

@impl begin
    struct UnconditionalLogcpdf end
    function logcpdf(sf::Unconditional{T}, i::Tuple{}, o::T)
        logcpdf(sf, i, o) # should never be called
    end
end

@impl begin
    struct UnconditionalMakeFactors
        numpartitions::Int64 = 100
    end

    function make_factors(sf::Unconditional{T}, 
            range::Vector{T}, 
            parranges::NTuple{N,Vector}, 
            id, 
            parids::Tuple)::Tuple{Vector{<:Scruff.Utils.Factor}, Vector{<:Scruff.Utils.Factor}} where {T,N}
        
        (lowers, uppers) = bounded_probs(sf, range, ())
        keys = (id,)
        dims = (length(range),)
        return ([Factor(dims, keys, lowers)], [Factor(dims, keys, uppers)])
    end
end

# compute_pi(u::Unconditional, range, parranges, incoming_pis) = compute_pi(u, range)

# unconditional sfuncs do not have parents, so send_lambda trivially sends an empty message
@impl begin 
    struct UnconditionalSendLambda end
    function send_lambda(sf::Unconditional{T},
                    lambda::Score{T},
                    range::Vector{T},
                    parranges::NTuple{N,Vector},
                    incoming_pis::Tuple,
                    parent_idx::Integer)::Score where {T,N}
        SoftScore(Float64[], Float64[])
    end
end