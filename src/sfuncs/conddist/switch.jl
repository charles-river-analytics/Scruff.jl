export 
    Switch,
    LinearSwitch,
    If,
    choose

"""
    abstract type Switch{N, I, K, O} <: Det{K, O}

`Switch` defines an sfunc that represents choosing between multiple incoming (parent) Sfuncs 
based on a test.  A subtype of `Switch` must provide a `choose` function that takes the switch 
and an `i` and returns an integer between `1` and `N`.  This is an index into a 'parent array'.

`K` must be a flat tuple type consisting of `I` and `N` occurrences of `O`:  for example,
if I is `Int`

```
    K = extend_tuple_type(Tuple{Int}, NTuple{N, O})
```

If the subtype'd sfunc is not in the Scruff.SFuncs module, the system must 
`import Scruff.SFuncs: choose`.

# Additional supported operators
- `support`
- `support_quality`
- `compute_pi`
- `send_lambda`

# Type parameters
- `N`: the count of *sfuncs* from which to choose
- `I`: the type of the second argument of the `choose` function defined for the `Switch`
- `K`: the input type of the `Switch`; see above
- `O`: the output type of the `Switch` 

See also:  [`choose`](@ref), [`extend_tuple_type`](@ref)
"""
abstract type Switch{N, I, K, O} <: Det{K, O} end

"""
`choose` interface.  For every subtype of `Switch`, an implementation of this method must
be created, whose first parameter is the subtype, and the second parameter is of type `I`
for the parameter type in `Switch`.

For example, the definition of the `If` *sfunc* is as follows, where `choose` returns
either index `1` or index `2`.

```
struct If{O} <: Switch{2, Bool, Tuple{Bool, O, O}, O} end
choose(::If, b::Bool) = b ? 1 : 2
```
"""
function choose end

# apply(sw::Switch, i, hs...) = hs[choose(sw,i)] # implemented because subtype of `Det`
apply(sw::Switch, i, hs...) = hs[choose(sw,i)] # implemented because subtype of `Det`

struct LinearSwitch{N, K, O} <: Switch{N, Int, K, O} 
    n :: Int
    function LinearSwitch(N, O)
        K = extend_tuple_type(Tuple{Int}, NTuple{N, O})
        new{N, K, O}(N)
    end
end

choose(::LinearSwitch, i) = i

struct If{O} <: Switch{2, Bool, Tuple{Bool, O, O}, O} end

choose(::If, b) = b ? 1 : 2

# switch overrides the functions for det that use the Cartesian product of the parents.
# Since only one parent is ever active at a time, we don't need to compute the Cartesian product.
# This leads to exponential time savings in the number of parents.
@impl begin
    struct SwitchSupport end
    function support(sf::Switch{N,I,K,O}, 
                    parranges::NTuple{M,Vector}, 
                    size::Integer, 
                    curr::Vector{<:O}) where {I,K,O,N,M}
        result = Vector{output_type(sf)}()
        for i in parranges[1]
            h = choose(sf, i)
            append!(result, parranges[h + 1])
        end
        return unique(result)
    end                
end

@impl begin
    struct SwitchSupportQuality end
    
    function support_quality(sf::LinearSwitch, parranges)
        ivals = map(i -> choose(sf, i), parranges[1])
        if all(j -> j in ivals, 1:sf.n)
            return :CompleteSupport
        else
            return :BestEffortSupport
        end
    end

    function support_quality(::Union{<:Support,Nothing}, sf::If, parranges)
        ivals = map(i -> choose(sf, i), parranges[1])
        if all(j -> j in ivals, 1:2)
            return :CompleteSupport
        else
            return :BestEffortSupport
        end
    end
end


# We currently don't overwrite bounded_probs.
# bounded_probs will be replaced with an operation that produces factors.

@impl begin
    struct SwitchComputePi end

    function compute_pi(sf::Switch{N,I,K,O},
                     range::__OptVec{<:O}, 
                     parranges::NTuple{M,Vector}, 
                     incoming_pis::Tuple)::Dist{<:O} where {M,N,I,K,O}
        result = zeros(Float64, length(range))
        ipis = incoming_pis[1]
        for (i,ival) in enumerate(parranges[1])
            ipi = cpdf(ipis, (), parranges[1][i])
            h = choose(sf, ival)
            hpis = incoming_pis[h+1]
            hrange = parranges[h+1]
            for (j,jval) in enumerate(range)
                k = indexin([jval], hrange)[1]
                # the range of the parent can be a subset of the range of the switch
                hpi = isnothing(k) ? 0.0 : cpdf(hpis, (), parranges[h+1][k])
                result[j] += ipi * hpi
            end
        end
        return Cat(range, result)
    end
end

@impl begin
    struct SwitchSendLambda end

    function send_lambda(sf::Switch{N,I,K,O},
                       lambda::Score{<:O},
                       range::__OptVec{<:O},
                       parranges::NTuple{M,Vector},
                       incoming_pis::Tuple,
                       parent_ix::Integer)::Score where {M,N,I,K,O}

        # This helper function computes, for a particular choice, the sum of
        # pi times lambda values.
        function compute1(h)
            hrange = parranges[h+1]
            hpis = incoming_pis[h+1]
            tot = 0.0
            for (j,jval) in enumerate(hrange)
                # We need to keep track of indices properly.
                # We cannot assume that the range of the switch and the range passed into
                # send_lambda are the same, even though they are of the same type O.
                # In particular, choices might have different ranges from each other,
                # perhaps restricted or in a different order.
                k = indexin([jval], range)[1]
                tot += cpdf(hpis, (), parranges[h+1][j]) * get_score(lambda, range[k])
            end
            return tot
        end

        # We need to make sure the message is correctly typed to the output type of the appropriate parent
        # Need to make sure the target parent range is a Vector{T} rather than a Vector{Any}
        T = typeof(parranges[parent_ix][1])
        target_parrange :: Vector{T} = parranges[parent_ix]
        if parent_ix == 1
            # We send a lambda message to the selector parent
            # For a given value of the selector, the lambda value is the 
            # probability of the child lambda for the corresponding choice.
            # This is equal to the sum, over all values in the range of the
            # choice, of the incoming_pi * lambda for that value.
            return SoftScore(target_parrange, 
                map(i -> compute1(choose(sf, i)), target_parrange))

        else
            # The lambda message to a choice consists of two components:
            # A choice-specific component equal to the pi of the I value that leads to that choice, times lambda
            # A constant from all the other choices, equal to the sum, over all choices, of the pi of the I value for the choice
            # times the sum of pi * lambda values for that choice.
            con = 0.0
            spec = nothing
            ipis = incoming_pis[1]
            for (i,ival) in enumerate(parranges[1])
                h = choose(sf, ival)
                if h+1 != parent_ix
                    con += cpdf(ipis, (), ival)  * compute1(h)
                else
                    # lambda is in the order of range, but hrange might be a subset and in a different order
                    hrange = parranges[h+1]
                    qs = []
                    for (j,jval) in enumerate(hrange)
                        # k = indexin([jval], range)[1]
                        # push!(qs, lambda[k])
                        push!(qs, get_score(lambda, jval))
                    end
                    spec = cpdf(ipis, (), ival) .* qs
                end
            end

            return SoftScore(target_parrange, con .+ spec)
        end
    end
end