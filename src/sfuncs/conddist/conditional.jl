export
    Conditional,
    extend_tuple_type

using Folds
using StatsFuns

@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)
tuplejoin(t::Tuple) = t

"""
    extend_tuple_type(T1, T2)

Given two types `T1` and `T2`, concatenate the types into a single
tuple type.  

# Arguments
- `T1`: Any type
- `T2`: A tuple type

# Returns
- If `T1` is a tuple type, a tuple with the concatenation of the types in `T1` and `T2`
- If `T1` is not a tuple type, a tuple with `T1` prepended to the types in `T2`

# Examples
```
julia> extend_tuple_type(Int64, Tuple{Float64})
Tuple{Int64, Float64}

julia> extend_tuple_type(Tuple{Int64}, Tuple{Float64})
Tuple{Int64, Float64}

julia> extend_tuple_type(Tuple{Vector{Float64}}, Tuple{Symbol,Symbol})     
Tuple{Vector{Float64}, Symbol, Symbol}
```
"""
function extend_tuple_type(T1, T2) 
    if T1 <: Tuple
        Tuple{tuplejoin(fieldtypes(T1), fieldtypes(T2))...}
    else
        Tuple{T1, fieldtypes(T2)...}
    end
end

"""
    abstract type Conditional{I <: Tuple, J <: Tuple, K <: Tuple, O, S <: SFunc{J, O}} <: SFunc{K, O}

`Conditional` *sfuncs* represent the generation of an sfunc depending on the values of parents.  An
subtype of `Conditional` must provide a `gensf` method that takes an `I` and returns an 
`SFunc{J,O}` (**important** the generated SFunc must not appear outside the Conditional. 
It should not be a parent).

# Additional supported operators
- `support`
- `support_quality`
- `sample`
- `logcpdf`
- `make_factors`
- `compute_pi`
- `send_lambda`

# Type parameters
- `I`: the type of data used to generate the `Conditional`'s *sfunc*
- `J`: a tuple that represents the input types (the `I`) of the `Conditional`'s generated *sfunc*
- `K`: the input types of the `Conditional`; this is a tuple of types constructed from `I`, 
and `J` using `extend_tuple_types`
- `O`: the output type(s) of both the `Conditional` and the `Conditional`'s generated *sfunc*
- `S`: the type of the `Conditional`'s generated *sfunc*

"""
abstract type Conditional{I, J <: Tuple, K <: Tuple, O, S <: SFunc{J, O}} <: SFunc{K, O}
end

function split_pars(::Conditional{I}, pars) where {I}
    if I <: Tuple
        n1 = length(fieldnames(I))
        (Tuple(pars[1:n1]), Tuple(pars[n1+1:length(pars)]))
    else
        (pars[1], Tuple(pars[2:length(pars)]))
    end
end

@impl begin
    struct ConditionalSupport end
    
    function support(sf::Conditional{I,J,K,O}, 
                    parranges::NTuple{N,Vector}, 
                    size::Integer, 
                    curr::Vector{<:O}) where {I,J,K,O,N}
        (iranges, jranges) = split_pars(sf, parranges)
        irgs::Array{Array{Any, 1}, 1} = [r for r in iranges]
        isrange = cartesian_product(irgs)
        rng::Vector = isempty(curr) ? Vector{O}() : copy(curr)
        # inc = Int(ceil(size / length(isrange)))
        allcombranges= Dict{Any, Vector{O}}()

        # create complete range regardless of the size
        for is in isrange
            gsf = gensf(sf,tuple(is...))
            newrng = copy(support(gsf, jranges, size, curr))
            allcombranges[is] = unique(newrng)
        end

        # create range by mixing values from ranges created by each combination of parents values up to size
        allrng = collect(Iterators.flatten(values(allcombranges)))
        max_size = length(unique(allrng))
        while length(rng) < min(size, max_size)
            for is in isrange
                if(length(allcombranges[is]) > 0)
                    val = popfirst!(allcombranges[is])
                    push!(rng, val)
                end
            end
            rng = unique(rng)
        end
        return isempty(rng) ? O[] : rng
    end
end

@impl begin
    struct ConditionalSample end

    function sample(sf::Conditional{I,J,K,O}, i::K)::O where {I,J,K,O}
        (ivals, jvals) = split_pars(sf, i)
        sfg = gensf(sf,ivals)
        return sample(sfg, jvals)
    end
end

@impl begin
    struct ConditionalLogcpdf end

    function logcpdf(sf::Conditional{I,J,K,O}, i::K, o::O)::AbstractFloat where {I,J,K,O}
        (ivals, jvals) = split_pars(sf, i)
        sfgen = gensf(sf,ivals)
        return logcpdf(sfgen, jvals, o)
    end    
end

@impl begin
    struct ConditionalSupportQuality end

    function support_quality(sf::Conditional{I,J,K,O,S}, parranges) where {I,J,K,O,S}
        q = support_quality_rank(:CompleteSupport)
        (iranges, jranges) = split_pars(sf, parranges)
        isrange = cartesian_product(iranges)
        for is in isrange
            gsf = gensf(sf,tuple(is...))
            imp = get_imp(MultiInterface.get_policy(), Support, gsf, jranges, 0, O[])
            q = min(q, support_quality_rank(support_quality(imp, gsf, jranges)))
        end
        return support_quality_from_rank(q)
    end
end

@impl begin
    mutable struct ConditionalMakeFactors
        numpartitions::Int64 = 1
    end

    function make_factors(sf::Conditional{I,J,K,O},
                        range::__OptVec{<:O}, 
                        parranges::NTuple{N,Vector}, 
                        id, 
                        parids::Tuple)::Tuple{Vector{<:Scruff.Utils.Factor}, Vector{<:Scruff.Utils.Factor}} where {I,J,K,O,N}

        if any(isempty, parranges) # This is possible in lazy inference
            # Just return an empty factor so as not to break
            keys = (id,)
            dims = (0,)
            entries = Float64[]
            fs = [Factor(dims, keys, entries)]
            return (fs, fs)
        end

        # Here is a very general method for constructing factors for conditional sfuncs.
        # We introduce a switch variable.
        # For each of the generated sfuncs, we make its factors and extend them by saying
        # they are only relevant when the switch value corresponds to that choice.
        # We also add a factor saying that the switch variable takes on the appropriate value
        # for each value of the i inputs.
        # For this switch factor, we use another decomposition, relating the switch variable
        # to each of the i inputs separately. This avoids a quadratic blowup and brings the
        # complexity analogous to that of less general methods.
        # This can be improved by making a special case where there is only one i parent,
        # and returning a single factor.
        (iranges, jranges) = split_pars(sf, parranges)
        (iids, jids) = split_pars(sf, parids)
        icombos = cartesian_product(iranges)
        lfs = Factor[]
        ufs = Factor[]
        switchkey = nextkey()
        switchsize = length(icombos)
        for (switchval, ivals) in enumerate(icombos)
            function extend(factor)
                keys = [k for k in factor.keys]
                push!(keys, switchkey)
                keys = Tuple(keys)
                dims = [d for d in factor.dims]
                push!(dims, switchsize)
                dims = Tuple(dims)
                entries = Float64[]
                for e in factor.entries
                    for k in 1:switchsize
                        # 1.0 is the irrelevant value when factors are multiplied
                        push!(entries, k == switchval ? e : 1.0)
                    end
                end
                result = Factor(dims, keys, entries)
                return result
            end
            subsf = gensf(sf,tuple(ivals...))
            (sublfs, subufs) = make_factors(subsf, range, jranges, id, jids)
            append!(lfs, [extend(f) for f in sublfs])
            append!(ufs, [extend(f) for f in subufs])
        end

        for i in 1:length(iranges)
            switchfactorkeys = (parids[i], switchkey)
            parsize = length(parranges[i])
            switchfactordims = (parsize, switchsize)
            switchfactorentries = Float64[]
            for j in 1:parsize
                for ivals in icombos
                    # The effect of this is that the product will only be 1 for all the corresponding
                    # i values.
                    push!(switchfactorentries, j == indexin([ivals[i]], parranges[i])[1] ? 1.0 : 0.0)
                end
            end
            switchfact = Factor(switchfactordims, switchfactorkeys, switchfactorentries)
            push!(lfs, switchfact)
            push!(ufs, switchfact)
        end

        return (lfs, ufs)
    end
end

@impl begin
    struct ConditionalComputePi end

    function compute_pi(sf::Conditional{I,J,K,O},
                     range::__OptVec{<:O}, 
                     parranges::NTuple{N,Vector}, 
                     incoming_pis::Tuple)::Dist{<:O} where {N,I,J,K,O}

        (iranges, jranges) = split_pars(sf, parranges)
        # We need to correctly handle the case with duplicate parent values
        # logcpdf below will get the full incoming pi for all duplicate values
        irgs::Vector{Set{Any}} = [Set(r) for r in iranges]
        icombos = cartesian_product(irgs)
        (ipis, jpis) = split_pars(sf, incoming_pis)
        result_pieces = [Vector{Float64}() for x in range]
        
        for is in icombos
            gsf = gensf(sf,tuple(is...))
            ps = compute_pi(gsf, range, Tuple(jranges), jpis)
            if (!isempty(iranges))
                # ipi = sum([logcpdf(ipis[i], (), irgs[i][ind[i]]) for i in 1:length(iranges)])
                ipi = sum([logcpdf(ipis[j], (), is[j]) for j in 1:length(iranges)])
            else
                ipi = 1
            end
            for i in 1:length(range)
                push!(result_pieces[i], ipi + logcpdf(ps, (), range[i]))
            end
        end
        result = normalize([exp(StatsFuns.logsumexp(rps)) for rps in result_pieces])
        return Cat(range, result)
    end
end

@impl begin
    struct ConditionalSendLambda end

    function send_lambda(sf::Conditional{I,J,K,O},
                       lambda::Score{<:O},
                       range::__OptVec{<:O},
                       parranges::NTuple{N,Vector},
                       incoming_pis::Tuple,
                       parent_ix::Integer)::Score where {N,I,J,K,O}
        (iranges, jranges) = split_pars(sf, parranges)
        (ipis, jpis) = split_pars(sf, incoming_pis)
        iar :: Array{Array{Int, 1}} = collect(map(r -> collect(1:length(r)), iranges))
        iinds = cartesian_product(iar)
        jar :: Array{Array{Int, 1}} = collect(map(r -> collect(1:length(r)), jranges))
        jinds = cartesian_product(jar)
        result = zeros(Float64, length(parranges[parent_ix]))
        # Need to make sure the target parent range is a Vector{T} rather than a Vector{Any}
        T = typeof(parranges[parent_ix][1])
        target_parrange :: Vector{T} = parranges[parent_ix]

        if parent_ix <= length(iranges)
            rs::Array{Array{Int, 1}, 1} = [collect(1:length(rg)) for rg in iranges]
            deleteat!(rs, parent_ix)
            restranges = cartesian_product(rs)
            # We want to send a lambda message to the specific I input.
            # We can ignore the pi message from this input.
            # For each value of this input, we want to compute \sum_{i',j,o} \pi_J(j) P_ps.sf(o | j ; gen_params(ps, (i, i'))) \lambda(o).
            for i = 1:length(iranges[parent_ix])
                ival = iranges[parent_ix][i]
                ls = Vector{Float64}()
                for rest in restranges
                    ipi = 0
                    fullvals = []
                    for r = 1:parent_ix-1
                        ipi += logcpdf(ipis[r], (), parranges[r][rest[r]])
                        push!(fullvals, iranges[r][rest[r]])
                    end
                    push!(fullvals, ival)
                    for r = parent_ix+1:length(iranges)
                        ipi += logcpdf(ipis[r], (), parranges[r][rest[r-1]])
                        push!(fullvals, iranges[r][rest[r-1]])
                    end
                    gsf = gensf(sf,tuple(fullvals...))
                    for jind in jinds
                        jpi = 0
                        jval = []
                        for r = 1:length(jpis)
                            jpi += logcpdf(jpis[r], (), jind[r])
                            push!(jval, jranges[r][jind[r]])
                        end
                        jvaltypes = [typeof(j) for j in jval]
                        tjval = convert(Vector{typejoin(jvaltypes...)}, jval)
                        pi = ipi + jpi
                        for o in range
                            push!(ls, pi + logcpdf(gsf, Tuple(tjval), o) + get_log_score(lambda, o))
                        end
                    end
                end
                result[i] = StatsFuns.logsumexp(ls)
            end
        elseif parent_ix <= length(parranges)
            n = length(iranges)
            jix = parent_ix - n
                # We weight the lambda messages sent to the specific J parent by the pi message for the I parents
            result_pieces = [Vector{Float64}() for x in target_parrange]
            for iind in iinds
                ival = [iranges[r][iind[r]] for r in 1:n]
                gsf = gensf(sf,tuple(ival...))
                ipi = sum(logcpdf(ipis[r], (), iind[r]) for r in 1:n)
                lam = send_lambda(gsf, lambda, range, jranges, jpis, jix)
                for i in 1:length(target_parrange)
                    push!(result_pieces[i], ipi + get_log_score(lam, parranges[parent_ix][i]))
                end
            end
            result = [StatsFuns.logsumexp(rp) for rp in result_pieces]
        end

        return LogScore(target_parrange, result)
    end
end