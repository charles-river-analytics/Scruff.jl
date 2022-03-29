
export
    Serial

"""
    struct Serial{I,O} <: SFunc{I,O}

A sequence of sfuncs in series.

Although this could be implemented as a special case of NetworkSFunc,
the serial composition allows an easier and more efficient implementation of operations.
All but the first sfunc in the sequence will have a single input; the output of each
sfunc feeds into the input of the next sfunc.

To work properly, most of the operations on `Serial` need the support of the intermediate
sfuncs, given an input range. Rather than compute this each time, and to avoid having
the non-support operations take a size argument, support is memoized, and must be called
before other operations like logcpdf are called. The `support_memo` is a dictionary whose
keys are tuples of parent ranges and whose values are the support computed for those
parent ranges, along with the target size for which they were computed.
Storing the target size enables refinement algorithms that increase the size and improve 
the support.

# Additional supported operators
- `support`
- `support_quality`
- `sample`
- `cpdf`
- `bounded_probs`
- `make_factors`
- `compute_pi`
- `send_lambda`

# Type parameters
- I the input type of the first sfunc
- O the output type of the last sfunc
"""
struct Serial{I,O} <: SFunc{I,O}
    components :: NTuple{N,SFunc} where N
    support_memo :: Dict
    Serial(I,O,sfuncs) = new{I,O}(sfuncs, Dict())
end

@impl begin
    struct SerialSupport end
    function support(sf::Serial{I,O}, parranges, size, curr) where {I,O}
        if parranges in keys(sf.support_memo)
            (sup,sz) = sf.support_memo[parranges]
            if sz >= size
                return last(sup)
            end
        end

        # Adequate support has not been found - compute it now
        prs = parranges
        compsups = Vector[]
        for component in sf.components
            sup = support(component, prs, size, output_type(component)[])
            push!(compsups, sup)
            prs = (sup,)
        end
        for c in curr
            if !(c in sup) 
                push!(sup, c)
            end
        end
        sf.support_memo[parranges] = (tuple(compsups...), size)
        return sup
    end
end

@impl begin
    struct SerialSupportQuality end

    function support_quality(sf::Serial{I,O}, parranges) where {I,O}
        if !(parranges in keys(sf.support_memo))
            error("Support must be called before support_quality for Serial")
        end
        compsups = sf.support_memo[parranges][1]
        rank = support_quality_rank(:CompleteSupport)
        prs = parranges
        for (comp,sup) in zip(sf.components, compsups)
            q = support_quality_rank(support_quality(comp, prs))
            rank = min(rank, q)
            prs = (sup,)
        end
        return support_quality_from_rank(rank)
    end
end

@impl begin
    struct SerialSample end
    function sample(sf::Serial{I,O}, i::I) where {I,O}
        x = i
        for component in sf.components
            x = (sample(component, x),)
        end
        return x[1]
    end
end

function _checksup(sf, i)
    for k in keys(sf.support_memo)
        if all([i[j] in k[j] for j in 1:length(k)])
            return sf.support_memo[k][1]
        end
    end
    return nothing
end

@impl begin
    struct SerialCpdf end
    function cpdf(sf::Serial{I,O}, i::I, o::O) where {I,O}
        compsups = _checksup(sf, i)
        if isnothing(compsups)
            error("No support found for parent values in logcpdf for Serial")
        end
        us = [i]
        pis = [1.0]
        n = length(sf.components)
        for j in 1:n-1
            comp = sf.components[j]
            sup = compsups[j]
            newpis = zeros(Float64, length(sup))
            for (u,pi) in zip(us,pis)
                for l in 1:length(sup)
                    x = sup[l]
                    newpis[l] += cpdf(comp, u, x) * pi
                end
            end
            us = [(x,) for x in sup]
            pis = newpis
        end
        result = 0.0
        finalcomp = sf.components[n]
        for (u, pi) in zip(us,pis)
            result += cpdf(finalcomp, u, o) * pi
        end
        return result
    end
end

@impl begin
    struct SerialBoundedProbs end

    function bounded_probs(sf::Serial{I,O}, range::Vector{<:O}, 
            parranges) where {I,O}
        probs = Float64[]
        combos = cartesian_product(parranges)
        for i in combos
            for o in range
                push!(probs, cpdf(sf, tuple(i...), o))
            end
        end
        return (probs, probs)
    end
end

@impl begin
    struct SerialMakeFactors end

    function make_factors(sf::Serial{I,O}, range::Vector{<:O}, parranges, 
            id, parids) where {I,O}
        if !(parranges in keys(sf.support_memo))
            error("Support must be called before make_factors for Serial")
        end
        compsups = sf.support_memo[parranges][1]
        inids = parids
        outid = nextkey()
        prs = parranges
        lowers = Factor[]
        uppers = Factor[]
        for i in 1:length(sf.components)
            outid = i == length(sf.components) ? id : nextkey()
            (ls, us) = make_factors(sf.components[i], compsups[i], prs, outid, inids)
            inid = (outid,)
            prs = (compsups[i],)
            append!(lowers, ls)
            append!(uppers, us)
        end
        return(lowers, uppers)
    end
end

@impl begin
    struct SerialComputePi end

    function compute_pi(sf::Serial{I,O}, range::Vector{<:O}, parranges, 
            incoming_pis) where {I,O}
        ps = zeros(Float64, length(range))
        combos = cartesian_product(parranges)
        m = length(parranges)
        for combo in combos
            parpis = [cpdf(incoming_pis[i], (), combo[i]) for i in 1:m]
            ipi = reduce(*, parpis)
            for (j,o) in enumerate(range)
                ps[j] += cpdf(sf, tuple(combo...), o) * ipi
            end
        end
        return Cat(range, ps)
    end
end

function _incpis(components, supports, parranges, incoming_pis)
    incpis = Tuple[incoming_pis]
    prs = parranges
    for i in 1:length(components)-1
        comp = components[i]
        rng = supports[i]
        pi = compute_pi(comp, rng, prs, incpis[i])
        push!(incpis, (pi,))
        prs = (rng,)
    end
    return incpis
end

function _lambdas(components, supports, lambda, incpis)
    lambdas = Score[lambda]
    lam = lambda
    for i = length(components):-1:2
        comp = components[i]
        rng = supports[i]
        prs = (supports[i-1],)
        ipis = incpis[i]
        lam = send_lambda(comp, lam, rng, prs, ipis, 1)
        pushfirst!(lambdas, lam)
    end
    return lambdas
end

@impl begin
    struct SerialSendLambda end

    function send_lambda(sf::Serial{I,O}, lambda, range, parranges, incoming_pis, 
            parent_idx) where {I,O}
        if !(parranges in keys(sf.support_memo))
            error("Support must be called before send_lambda for Serial")
        end
        compsups = sf.support_memo[parranges][1]
        incpis = _incpis(sf.components, compsups, parranges, incoming_pis)
        lambdas = _lambdas(sf.components, compsups, lambda, incpis)
        return send_lambda(sf.components[1], lambdas[1], compsups[1], 
            parranges, incoming_pis, parent_idx)
    end
end
#=
@impl begin
    struct SerialInitialStats end
    initial_stats(sf::Serial) = map(initial_stats, sf.components)
end

@impl begin
    struct SerialAccumulateStats end
    accumulate_stats(sf::Serial, existing_stats, new_stats) = 
        [accumulate_stats(sf.components[i], existing_stats[i], new_stats[i])  
            for i in 1:length(sf.components)]
end

@impl begin
    struct SerialExpectedStats end
    function expected_stats(sf::Serial, range, parranges, incoming_pis, child_lambda)
        if !(parranges in keys(sf.support_memo))
            error("Support must be called before expected_stats for Serial")
        end
        compsups = sf.support_memo[parranges][1]
        incpis = _incpis(sf.components, compsups, parranges, incoming_pis)
        lambdas = _lambdas(sf.components, compsups, child_lambda, incpis)
        stats = Any[]
        prs = parranges
        for i in 1:length(sf.components)
            comp = sf.components[i]
            sup = compsups[i]
            ipi = incpis[i]
            lam = lambdas[i]
            push!(stats, expected_stats(comp, sup, prs, ipi, lam))
            prs = (sup,)
        end
        return tuple(stats...)    
    end
end

@impl begin
    struct SerialMaximizeStats end
    function maximize_stats(sf::Serial, stats) 
        ps = [maximize_stats(sf.components[i], stats[i]) for i in 1:length(sf.components)]
        return tuple(ps...)
    end
end
=#
            






