export Mixture

"""
    mutable struct Mixture{I,O} <: SFunc{I,O}

`Mixture` defines an *sfunc* representing mixtures of other *sfuncs*.  It contains a 
vector of *sfuncs* and a vector of probabilities that those *sfuncs* are selected, 
which indices are keys associating the two.  The output type of a `Mixture` is defined by
the output type of its internal components.  The parameters of a `Mixture` are
its probabilities followed by parameters of all its internal components, in order. 

# Additional supported operators
- `support`
- `support_quality`
- `sample`
- `cpdf`
- `expectation`
- `compute_pi`
- `send_lambda`

# Type parameters
- `I`: the input type(s) of the `Mixture`
- `O`: the shared output type(s) of its internal components and the output type(s) of the `Mixture`
"""
mutable struct Mixture{I,O} <: SFunc{I,O}
    components::Vector{<:SFunc{I,O}}
    probabilities::Vector{Float64}
end

@impl begin
    struct MixtureSupport end
    function support(sf::Mixture{I,O}, 
                    parranges::NTuple{N,Vector}, 
                    size::Integer, 
                    curr::Vector{<:O}) where {I,O,N}

        subsize = Int(ceil(size / length(sf.components)))
        ranges = [support(comp, parranges, subsize, curr) for comp in sf.components]
        result = []
        for range in ranges
            append!(result, range)
        end
        result = unique(result)
        tresult = convert(Vector{output_type(sf)}, result)
        # tresult = Vector{output_type(sf)}(undef, length(result))
        # copyto!(tresult, result)
        sort!(tresult)
        return tresult
    end
end

#=
@impl begin
    struct MixtureInitialStats end
    initial_stats(sf::Mixture) = [initial_stats(c) for c in sf.components]
end

@impl begin
    struct MixtureAccumulateStats end
    function accumulate_stats(sf::Mixture, existing_stats, new_stats) 
        [accumulate_stats(sf.components[i], existing_stats[i], new_stats[i]) for i in 1:length(sf.components)]
    end
end


@impl begin
    struct MixtureExpectedStats end
    function expected_stats(sf::Mixture{I,O},
                          range::VectorOption{<:O}, 
                          parranges::NTuple{N,Vector},
                          pis::NTuple{M,Dist},
                          child_lambda::Score{<:O}) where {I,O,N,M}
        # The statistics organize mutually exclusive cases.
        # Each case consists of a component i, a parent value p, and a child value c,
        # and represents the #(i selected, c | p, evidence).
        # This is equal to P(i selected | p, evidence) #(c | i selected, p, evidence).
        # P(i selected | p, evidence) is proportional to m.probabilities[i] * \sum_c #(c | i selected, p, evidence).
        # P(C | i selected, p, evidence) is equal to expected_stats(m.components[i], same arguments...)
        compstats = [expected_stats(comp, range, parranges, pis, child_lambda) for comp in sf.components]
        # This is bad. It assumes stats is a Dict, which is okay for tables
        # summed = [Dict((k,sum(v)) for (k,v) in stats) for stats in compstats]
        summed = [sum(sum(v) for v in values(stats)) for stats in compstats]
        pselected = sf.probabilities .* summed
        # return [mult_through(compstats[i], pselected[i]) for i in 1:length(m.components)]
        return [mult_through(compstats[i], sf.probabilities[i]) for i in 1:length(sf.components)]
    end
end

@impl begin
    struct MixtureMaximizeStats end
    function maximize_stats(sf::Mixture, stats)
        probparams = normalize([sum(sum(values(st))) for st in stats])
        compparams = [maximize_stats(sf.components[i], stats[i]) for i in 1:length(sf.components)]
        return (probparams, compparams...)
    end
end
=#

@impl begin
    struct MixtureSupportQuality end

    function support_quality(sf::Mixture{I,O}, parranges) where {I,O}
        q = support_quality_rank(:CompleteSupport)
        for comp in sf.components
            imp = get_imp(MultiInterface.get_policy(), Support, sf, parranges, 0, O[])
            q = min(q, support_quality_rank(support_quality(imp, comp, parranges)))
        end
        return support_quality_from_rank(q)
    end
end

@impl begin
    mutable struct MixtureMakeFactors
        numpartitions::Dict{SFunc, Int64} = Dict{SFunc, Int64}()
    end

    function make_factors(sf::Mixture{I,O},
                        range::VectorOption{<:O}, 
                        parranges::NTuple{N,Vector}, 
                        id, 
                        parids::Tuple)::Tuple{Vector{<:Scruff.Utils.Factor}, Vector{<:Scruff.Utils.Factor}} where {I,O,N}
        lfactors = Vector{Scruff.Utils.Factor}()
        ufactors = Vector{Scruff.Utils.Factor}()
        numcomps = length(sf.components)
        mixkey = nextkey()
        for (i,comp) in enumerate(sf.components)
            (lcompfactors, ucompfactors) = make_factors(comp, range, parranges, id, parids)
            function process(factors, target)
                for fact in factors
                    dims = [d for d in fact.dims]
                    push!(dims, numcomps)
                    dims = Tuple(dims)
                    keys = [k for k in fact.keys]
                    push!(keys, mixkey)
                    keys = Tuple(keys)
                    entries = Float64[]
                    for e in fact.entries
                        for j = 1:numcomps
                            push!(entries, i == j ? e : 1.0) # 1.0 means irrelevant
                        end
                    end
                    relevantfact = Factor(dims, keys, entries)
                    push!(target, relevantfact)
                end
            end
            process(lcompfactors, lfactors)
            process(ucompfactors, ufactors)
        end
        mixdims = (numcomps,)
        mixkeys = (mixkey,)
        mixentries = sf.probabilities
        mixfact = Factor(mixdims, mixkeys, mixentries)
        push!(lfactors, mixfact)
        push!(ufactors, mixfact)
        return (lfactors, ufactors)
    end
end

@impl begin
    struct MixtureComputePi end
    function compute_pi(sf::Mixture{I,O},
                     range::VectorOption{<:O}, 
                     parranges::NTuple{N,Vector}, 
                     incoming_pis::Tuple)::Dist{<:O} where {N,I,O}
        function f(i)
            cp = compute_pi(sf.components[i], range, parranges, incoming_pis)
            sf.probabilities[i] .* [cpdf(cp, (), x) for x in range]
        end
        scaled = [f(i) for i in 1:length(sf.components)]
        result = sum(scaled)
        return Cat(range, normalize(result))
    end
end

@impl begin
    struct MixtureSendLambda end
    function send_lambda(sf::Mixture{I,O},
                       lambda::Score{<:O},
                       range::VectorOption{<:O},
                       parranges::NTuple{N,Vector},
                       incoming_pis::Tuple,
                       parent_ix::Integer)::Score where {N,I,O}

        # Need to make sure the target parent range is a Vector{T} rather than a Vector{Any}
        T = typejoin([typeof(x) for x in parranges[parent_ix]]...)
        
        target_parrange :: Vector{T} = parranges[parent_ix]
        lams = [send_lambda(comp, lambda, range, parranges, incoming_pis, parent_ix) for comp in sf.components]
        scores = [[get_score(lams[j], target_parrange[i]) for i in 1:length(target_parrange)] for j in 1:length(sf.components)]
        scaled = sf.probabilities .* scores
        result = zeros(Float64, length(target_parrange))
        for sc in scaled
            result .+= sc
        end
        return SoftScore(target_parrange, result)
    end
end

# This does not seem to fit
@impl begin
    struct MixtureSample end
    function sample(sf::Mixture{I,O}, x::I)::O where {I,O}
        probs = sf.probabilities/sum(sf.probabilities)
        cat = Categorical(probs)
        which_component = rand(cat)
        component = sf.components[which_component]
        return sample(component, x)
    end
end

@impl begin
    struct MixtureCpdf end
    function cpdf(sf::Mixture{I,O}, i::I, o::O)::AbstractFloat where {I,O}
        complpdf = [cpdf(comp, i, o) for comp in sf.components]
        probs = sf.probabilities/sum(sf.probabilities)
        return sum(probs .* complpdf)
    end
end

@impl begin
    struct MixtureExpectation end
    
    function expectation(sf::Mixture{I,O}, x::I)::O where {I,O}
        probs = sf.probabilities/sum(sf.probabilities)
        cat = Categorical(probs)
        which_component = rand(cat)
        component = sf.components[which_component]
        return expectation(component, x)
    end
end
