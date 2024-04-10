export Extend

"""
    struct Extend{I<:Tuple{Any},J,O} <: SFunc{J,O}

`Extend` defines an sfunc that extend the input of another sfunc.
Useful for defining Separable SFuncs.

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
- `I`: the input type(s) of the extended *sfunc*; it must be a tuple of length 1
- `J`: the input type(s) of the `Extend`
- `O`: the output type(s) of both the `Extend` and the extended *sfunc* 
"""
struct Extend{I<:Tuple{Any},J,O} <: SFunc{J,O}
    given :: SFunc{I,O} # I must be a Tuple type of length 1
    position :: Int
    """
        function Extend(J::DataType, given::S, position::Int) where {I<:Tuple{Any},O,S <: SFunc{I,O}}
    
    `Extend`'s constructor
    
    # Arguments
    - `J::DataType`: the input type of the `Extend`
    - `given::S`: the *sfunc* to extend
    - `position::Int`: the index of the !TODO!
    """
    function Extend(J::DataType, given::S, position::Int) where {I<:Tuple{Any},O,S <: SFunc{I,O}}
        new{I,J,O}(given, position)
    end
end

@impl begin
    struct ExtendSupport end
    
    function support(sf::Extend{I,J,O}, 
            parranges::NTuple{N,Vector}, 
            size::Integer, 
            curr::Vector{<:O}) where {I<:Tuple{Any},J,O,N}

        parrange = parranges[sf.position]
        return support(sf.given, Tuple([[p] for p in parrange]), size, curr)
    end

end

@impl begin
    struct ExtendSample end
    function sample(sf::Extend{I,J,O}, i::J)::O where {I<:Tuple{Any},J,O}
        parval = i[sf.position]
        return sample(sf.given, tuple(parval))
    end
end

@impl begin
    struct ExtendLogcpdf end
    function logcpdf(sf::Extend{I,J,O}, i::J, o::O)::AbstractFloat where {I<:Tuple{Any},J,O}
        parval = i[sf.position]
        return logcpdf(sf.given, tuple(parval), o)
   end
end

@impl begin 
    struct ExtendSupportQuality end
    
    function support_quality(sf::Extend{I,J,O}, fullparranges) where {I,J,O}
        parrange = fullparranges[sf.position]
        parranges = Tuple([[p] for p in parrange])
        imp = get_imp(MultiInterface.get_policy(), Support, sf.given, parranges, 0, O[])
        return support_quality(imp, sf.given, parranges)
    end
end

@impl begin
    mutable struct ExtendMakeFactors
        numpartitions::Int64 = 10
    end

    function make_factors(sf::Extend{I,J,O},
            range::VectorOption{<:O}, 
            parranges::NTuple{N,Vector}, 
            id, 
            parids::Tuple)::Tuple{Vector{<:Scruff.Utils.Factor}, Vector{<:Scruff.Utils.Factor}} where {I<:Tuple{Any},J,O,N}

        parrange = parranges[sf.position]
        parid = parids[sf.position]
        return make_factors(sf.given, range, (parrange,), id, (parid,))
    end
end

@impl begin
    struct ExtendComputePi end

    function compute_pi(sf::Extend{I,J,O},
            range::VectorOption{<:O}, 
            parranges::NTuple{N,Vector}, 
            incoming_pis::Tuple)::Dist{<:O} where {N,J,I<:Tuple{Any},O}
    
        thisparrange = parranges[sf.position]
        thisincoming_pis = incoming_pis[sf.position]
        return compute_pi(sf.given, range, (thisparrange,), (thisincoming_pis,))
    end
end

@impl begin
    struct ExtendSendLambda end

    function send_lambda(sf::Extend{I,J,O},
                       lambda::Score{<:O},
                       range::VectorOption{<:O},
                       parranges::NTuple{N,Vector},
                       incoming_pis::Tuple,
                       parent_idx::Integer)::Score where {I<:Tuple{Any},N,J,O}
    
        thisparrange = parranges[sf.position]
        thisincoming_pis = incoming_pis[sf.position]
        if parent_idx != sf.position
            # This parent is not relevant.
            # We need to return a constant: \sum_i \sum_x \pi_i P(x|i) \lambda(x)
            # where i ranges over fullparranges[e.position]
            cpieces = Vector{Float64}()
            for i in 1:length(thisparrange)
                pi = logcpdf(thisincoming_pis, (), thisparrange[i])
                for j in 1:length(range)
                    px = logcpdf(sf.given, Tuple(thisparrange[i]), range[j])
                    push!(cpieces, pi + px + get_log_score(lambda, range[j]))
                end
            end
            c = exp(StatsFuns.logsumexp(cpieces))
            l = SoftScore(parranges[parent_idx], fill(c, length(parranges[parent_idx])))
            return l
        else
            # We use parent_idx 1 since e.given has only one parent
            l = send_lambda(sf.given, lambda, range, (thisparrange,), (thisincoming_pis,), 1)
            return l
        end
    end
end

#=
@impl begin
    struct ExtendInitialStat end
    initial_stats(sf::Extend) = initial_stats(sf.given)
end

@impl begin
    struct ExtendAccumulateStats end
    function accumulate_stats(sf::Extend, existing_stats, new_stats)
        accumulate_stats(sf.given, existing_stats, new_stats)
    end
end

@impl begin
    struct ExtendExpectedStats end
    
    function expected_stats(sf::Extend{I,J,O},
                            range::VectorOption{<:O}, 
                            parranges::NTuple{N,Vector},
                            pis::NTuple{M,Dist},
                            child_lambda::Score{<:O}) where {I<:Tuple{Any},J,O,N,M}

        parrange = parranges[sf.position]
        parent_pis = pis[sf.position]
        return expected_stats(sf.given, range, tuple(parrange), tuple(parent_pis), child_lambda)
    end
end

@impl begin
    struct ExtendMaximizeStats end
    maximize_stats(sf::Extend, stats) = maximize_stats(sf.given, stats)
end
=#


