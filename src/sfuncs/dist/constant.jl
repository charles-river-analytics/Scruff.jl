export
    Constant

"""
    mutable struct Constant{O} <: Dist{O,Nothing}

`Constant` represents an sfunc that always produces the same value.  It has no
inputs and no parameters.

# Additional supported operators
- `support`
- `support_quality`
- `sample`
- `logcpdf`
- `bounded_probs`
- `compute_pi`

# Type parameters
- `O`: the output type(s) of the `Constant`
"""
mutable struct Constant{O} <: Dist{O}
    "the constant value to be returned"
    x :: O
end

@impl begin
    struct ConstantSupport end
    
    function support(sf::Constant{O},
            parranges::NTuple{N,Vector}, 
            size::Integer, 
            curr::Vector{<:O}) where {O,N} 
        [sf.x]
    end
end

@impl begin
    struct ConstantSupportQuality end
    function support_quality(::Constant, parranges)
        :CompleteSupport
    end
end

@impl begin
    struct ConstantSample end

    function sample(sf::Constant{O}, i::Tuple{})::O where {O}
        sf.x
    end
end

@impl begin
    struct ConstantLogcpdf end

    function logcpdf(sf::Constant{O}, i::Tuple{}, o::O)::AbstractFloat where {O}
        return o == sf.x ? 0.0 : -Inf
    end
end

@impl begin
    struct ConstantBoundedProbs end

    function bounded_probs(sf::Constant{O},
            range::Vector{<:O}, 
            parranges::NTuple{N,Vector})::Tuple{Vector{<:AbstractFloat}, Vector{<:AbstractFloat}} where {O,N}
    
        p = sf.x in range ? [1.0] : [0.0]
        return (p, p)
    end
end

#=
@impl begin
    struct ConstantInitialStats end

    initial_stats(sf::Constant) = nothing
end
=#
@impl begin
    struct ConstantComputePi end

    function compute_pi(sf::Constant{O}, 
            range::Vector{<:O}, 
            parranges::NTuple{N,Vector}, 
            incoming_pis::Tuple)::Dist{<:O} where {O,N}

        ps = [x == sf.x ? 1.0 : 0.0 for x in range]
        Cat(range, ps)
    end
end

# STATS
@impl begin
    struct ConstantAccumulateStats end

    function accumulate_stats(::Constant, existing_stats, new_stats)
        nothing
    end
end

@impl begin
    struct ConstantExpectedStats end

    function expected_stats(::Constant{O}, 
            range::Vector{}, 
            parranges::NTuple{N,Vector},
            pis::NTuple{M,Dist},
            lambda::Score{<:O}) where {O,N,M}

        nothing
    end
end

@impl begin
    struct ConstantMaximizeStats end

    function maximize_stats(::Constant, stats)
        nothing
    end
end
# STATS END