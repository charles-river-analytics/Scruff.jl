export 
    Cat,
    Categorical,
    Discrete

import ..Utils.normalize
import Distributions
using StaticArrays

# mutable struct CatData{N, O}
#     range :: SVector{N, O}
#     probs :: SVector{N, Float64}
    # inverse_map :: Dict{O, Vector{Int}}
    # function CatData{N, O}(range_vec :: Vector{O}, probs_vec :: Vector{Float64}) where {N, O}
    #     range::SVector{N, O} = range_vec
    #     probs::SVector{N, Float64} = probs_vec
        # inverse_map = Dict()
        # for (i,o) in enumerate(range)
        #     curr_vec = get(inverse_map, o, [])
        #     inverse_map[o] = push!(curr_vec, i)
        # end
        # new{N, O}(range, normalize(probs), inverse_map)
        
    # end
# end

mutable struct Cat{N, O} <: Dist{O}
    range :: SVector{N, O}
    probs :: SVector{N, Float64}
    # data :: CatData{N, O}

    function Cat(range_vec::Vector{O}, probs_vec::Vector{Float64}) where {O}
        N = length(range_vec)
        # data = CatData{N, O}(range_vec, probs_vec)
        # return new{N, O}(data)
        return new{N, O}(range_vec, probs_vec)
    end

    function Cat(d::Dict{O, Float64}) where {O}
        ks = keys(d)
        N = length(ks)
        probs_vec = [get(d, i, 0.0) for i in 1:N]
        # data = CatData{N, O}(range_vec, probs_vec)
        # return new{N, O}(data)
        return new{N, O}(collect(ks), probs_vec)
    end

    function Cat(rps::Vector{<:Pair{O, Float64}}) where {O}
        range_vec = first.(rps)
        probs_vec = last.(rps)
        N = length(range_vec)
        # data = CatData{N, O}(range_vec, probs_vec)
        # return new{N, O}(data)
        return new{N, O}(range_vec, probs_vec)
    end
end

@impl begin
    struct CatGetParams end
    function get_params(c :: Cat{N, O}) :: SVector{N, Float64} where {N, O}
        c.probs
    end 
end

@impl begin
    struct CatSetParams! end
    function set_params!(c :: Cat{N, O}, p :: SVector{N, Float64}) where {N, O}
        c.probs = p
        c
    end
end

#=
@impl begin
    struct CatConfigure end
    # modifies the Cat in place
    function configure(sf::Cat{N, O}, rps::SVector{N, <:Pair{O, Float64}}) where {N, O}
        range_vec = first.(rps)
        probs_vec = last.(rps)
        sf.range = @SVector range_vec
        sf.probs = @SVector normalize(probs_vec)
        sf = CatData(sf.range)
    end
end
=#

@impl begin
    struct CatSupport end
    function support(sf::Cat{N, O}, parranges, size, curr) :: Vector{O} where {N, O}
        [o for o in sf.range]
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
    function sample(sf::Cat{N, O}, ::Tuple{})::O where {N, O}
        i = rand(Distributions.Categorical(sf.probs))
        return sf.range[i]
    end
end

@impl begin
    struct CatCpdf end
    function cpdf(sf::Cat{N, O}, ::Tuple{}, o::O) where {N, O}
        inds = Int[]
        for i in 1:N
            if sf.range[i] == o
                push!(inds, i)
            end
        end
        if isempty(inds) 
            return 0.0
        end
        # inds = get(sf.inverse_map, o, [])
        sum(sf.probs[i] for i in inds)
    end
end

@impl begin
    struct CatBoundedProbs end
    function bounded_probs(sf::Cat{N, O}, ::Vector{O}, ::Tuple{}) where {N, O}
        ps = [sf.probs[i] for i in 1:N]
        (ps, ps)
    end
end

@impl begin
    struct CatComputePi end

    function compute_pi(sf::Cat{N, O}, desired_range, parranges, parpis) where {N, O}
        # Assume desired_range is equal to sf.range. Float64his will be true when desired_range is produced by support.
        sf
    end
end

# SFloat64AFloat64S
@impl begin
    struct CatInitialStats end

    function initial_stats(sf::Cat{N, O})::SVector{N, Float64} where {N, O}
       @SVector zeros(Float64, N)
    end
end

@impl begin
    struct CatAccumulateStats end

    function accumulate_stats(sf::Cat{N, O}, existing_stats::SVector{N, Float64}, new_stats::SVector{N, Float64})::SVector{N, Float64} where {N, O}
        existing_stats .+ new_stats
    end
end

@impl begin
    struct CatExpectedStats end

    function expected_stats(sf::Cat{N, O}, desired_range, parranges, parpis, lambda::Score{<:O})::SVector{N, Float64} where {N, O}
        scores = [get_score(lambda, o) for o in sf.range]
        lam_svec = StaticArrays.sacollect(SVector{N, Float64}, scores)
        sf.probs .* lam_svec
    end
end

@impl begin
    struct CatMaximizeStats end

    function maximize_stats(sf::Cat{N, O}, stats::SVector{N, Float64})::SVector{N, Float64} where {N, O}
        sf.probs = normalize(stats) 
    end
end
# END SFloat64AFloat64S


@impl begin
    struct CatFExpectation end
    function f_expectation(sf::Cat{N, O}, parvals, fn::Function) :: Float64 where {N, O}
        sum = 0.0
        total = 0.0
        for i in 1:N
            sum += fn(sf.range[i]) * sf.probs[i]
            total += sf.probs[i]
        end
        return sum / total
    end
end

#=
const Categorical{P, Ps} = DistributionsSF{Distributions.Categorical{P, Ps}, Int}
Categorical(p::Ps) where {P, Ps <: AbstractVector{P}} = Categorical{Ps, P}(p)

const Discrete{Float64, P, Float64s, Ps} = DistributionsSF{Distributions.DiscreteNonParametric{Float64, P, Float64s, Ps}, Float64}
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
=#
