@impl begin
    struct DistMakeFactors
        numpartitions::Int64 = 100
    end

    function make_factors(sf::Dist{T}, 
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

@impl begin 
    struct DistSendLambda end
    
    function send_lambda(sf::Dist{T},
                       lambda::Score{<:T},
                       range::VectorOption{<:T},
                       parranges::NTuple{N,Vector},
                       incoming_pis::Tuple,
                       parent_idx::Integer)::Score where {N,T}

        SoftScore(Float64[], Float64[])
    end
end

include("distributions.jl")
include("cat.jl")
include("constant.jl")
include("flip.jl")
include("normal.jl")
include("uniform.jl")
