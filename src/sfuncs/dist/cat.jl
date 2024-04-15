export 
    Cat,
    Categorical,
    Discrete

import ..Utils.normalize
import Distributions

const Categorical{P, Ps} = DistributionsSF{Distributions.Categorical{P, Ps}, Int}
Categorical(p::Ps) where {P, Ps <: AbstractVector{P}} = Categorical{Ps, P}(p)

const Discrete{T, P, Ts, Ps} = DistributionsSF{Distributions.DiscreteNonParametric{T, P, Ts, Ps}, T}
function Discrete(xs::Xs, ps::Ps) where {X, Xs <: AbstractVector{X}, P, Ps <: AbstractVector{P}} 
    # Handle duplicates
    sort_order = sortperm(xs)
    xs = xs[sort_order]
    ps = ps[sort_order]

    for i=1:(length(xs) - 1)
        if xs[i] == xs[i + 1]
            ps[i] += ps[i + 1]
            ps[i + 1] = 0
        end
    end
    keep = ps .> 0
    xs = xs[keep]
    ps = ps[keep]
      
    return Discrete{X, P, Xs, Ps}(xs, ps)
end

@doc """
    mutable struct Cat{O} <: Dist{O, Vector{Real}}

`Cat` defines an sfunc that represents a set of categorical output values that are not conditioned
on any input.  Its parameters are always of type `Vector{Real}`, which is the probability of each
output value.

# Supported operations
- `support`
- `support_quality`
- `sample`
- `cpdf`
- `bounded_probs`
- `compute_pi`
- `f_expectation`

# Type parameters
- `O`: the output type of the `Cat`
"""
mutable struct Cat{O} <: Dist{O}
    range :: Vector{O}
    inversemap :: Dict{O, Int}
    params :: Vector{<:Real}
    original_range :: Vector{O}
    """
        Cat(r::Vector{O}, ps::Vector{<:Real}) where O

    `Cat`'s constructor

    # Arguments
    - `r::Vector{O}`: the set of possible output values
    - `ps::Vector{<:Real}`: the set of probabilities for each value in `r` 
      (will be normalized on call to `sample`)
    """
    function Cat(range::Vector{O}, params::Vector{<:Real}) where O
        @assert length(range) == length(params)
        # Handle repeated values correctly
        d = Dict{O, Float64}()
        for (x,p) in zip(range, params)
            d[x] = get(d, x, 0) + p
        end
        r = collect(keys(d))
        ps = [d[x] for x in r]
        inversemap = Dict([x => i for (i,x) in enumerate(r)]) 
        new{O}(r,inversemap,ps,range)
    end

    function Cat(d::Dict{O, <:Real}) where O
        Cat(collect(keys(d)), collect(values(d)))
    end

    """
        Cat(rps::Vector{Pair{O,<:Real}}) where O
    `Cat` constructor
    # Arguments
    - rps::Vector{Pair{O,Float64}}: a set of `Pair`s `output_value=>probability_of_value`
    """
    function Cat(rps::Vector{<:Pair{O,<:Real}}) where O
        range = first.(rps)
        probs = last.(rps)
        return Cat(range, normalize(probs))
    end
end

@impl begin
    struct CatSupport end
    function support(sf::Cat{O}, 
                     parranges::NTuple{N,Vector}, 
                     size::Integer, 
                     curr::Vector{<:O}) where {O,N}
        sf.range
    end
end

@impl begin
    struct CatSupportQuality end
    function support_quality(::Cat, parranges)
        :CompleteSupport
    end
end
    
@impl begin
    struct CatSample end
    function sample(sf::Cat{O}, i::Tuple{})::O where {O}
        i = rand(Distributions.Categorical(sf.params))
        return sf.range[i]
    end
end

@impl begin
    struct CatCpdf end
    function cpdf(sf::Cat{O}, i::Tuple{}, o::O)::AbstractFloat where {O}
        if o in sf.range
            return sf.params[sf.inversemap[o]]
        else
            return 0.0
        end
    end
end

@impl begin
    struct CatBoundedProbs end
    function bounded_probs(sf::Cat{O}, 
                        range::Vector{O}, 
                        ::NTuple{N,Vector})::Tuple{Vector{<:AbstractFloat}, 
                        Vector{<:AbstractFloat}} where {O,N}

        ps = [x in keys(sf.inversemap) ? sf.params[sf.inversemap[x]] : 0.0 for x in range]
        (ps, ps)
    end
end

@impl begin
    struct CatComputePi end

    function compute_pi(sf::Cat{O},
                    range::Vector{O}, 
                    ::NTuple{N,Vector}, 
                    ::Tuple)::Cat{O} where {N,O}

        ps = bounded_probs(sf, range, ())[1]
        Cat(range, ps)
    end
end

#=
@impl begin
    struct CatInitialStats end

    function initial_stats(sf::Cat)
        zeros(Float64, length(sf.original_range))
    end
end

@impl begin
    struct CatAccumulateStats end

    function accumulate_stats(sf::Cat, existing_stats, new_stats)
        existing_stats .+ new_stats
    end
end

@impl begin
    struct CatExpectedStats end

    function expected_stats(sf::Cat{O}, 
            range::Vector{O}, 
            ::NTuple{N,Vector},
            ::NTuple{M,Dist},
            lambda::Score{<:O}) where {O,N,M}
        orig = sf.original_range
        ps = zeros(Float64, length(orig))
        for (i,x) in enumerate(orig)
            if x in sf.range
                ps[i] = sf.params[sf.inversemap[x]]
            end
        end
        ls = [get_score(lambda, r) for r in orig]
        return ps .* ls
    end
end

@impl begin
    struct CatMaximizeStats end

    function maximize_stats(sf::Cat, stats)
        normalize(stats) 
    end
end
=#

@impl begin
    struct CatFExpectation end
    function f_expectation(sf::Cat, ::Tuple{}, fn::Function)
        sum = 0.0
        total = 0.0
        for (i,x) in enumerate(sf.range)
            sum += fn(x) * sf.params[i]
            total += sf.params[i]
        end
        return sum / total
    end
end
