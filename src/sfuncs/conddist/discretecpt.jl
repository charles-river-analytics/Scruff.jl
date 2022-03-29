export
    DiscreteCPT

"""
    function DiscreteCPT(range::Vector{O}, paramdict::Dict{I, Vector{Float64}}) where {I <: Tuple, O}

Constructs an sfunc that represents a Discrete Conditional Probability table.

`DiscreteCPT`s are implemented as a `Table` with a `Cat`.

The `range` parameter defines all the possible outputs of the `DiscreteCPT`.  The `paramdict` 
parameter defines the input(s) and the actual CPT.  For example, 

```
    range = [1, 2]
    paramdict = Dict((1,1) => [0.3, 0.7], (1,2) => [0.6, 0.4], (2,1) =>[0.4, 0.6],
    (2,2) => [0.7, 0.3], (3,1) => [0.5, 0.5], (3,2) => [0.8, 0.2])
```

can create a `DiscreteCPT` which has two(2) inputs (the length of the key) and, given each input
as defined by the key, selects either `1` or `2` (the range) with the given probability.  So, if
the input is `(2,1)`, `1` is selected with probability `0.4` and `2` is selected with probability
`0.6`.

See also: [`Table`](@ref), [`Cat`](@ref)
"""
function DiscreteCPT(range::Vector{O}, paramdict::Dict{I, Vector{Float64}}) where {I <: Tuple, O}
    NumInputs = length(collect(keys(paramdict))[1])
    sfmaker(probs) = Cat(range, probs)
    return Table(Tuple{}, O, NumInputs, paramdict, sfmaker)
end

#=
# This form is provided for backward compatibility
# The ranges of the parents and child are all integer ranges from 1 to the number of values
function DiscreteCPT(params :: Array{Vector{Float64}, N}) where N
    rangesizes = size(params)
    parranges = [collect(1:r) for r in rangesizes]
    parcombos = cartesian_product(parranges)
    childrange = collect(1:length(params[1]))
    I = NTuple{N, Int}
    paramdict = Dict{I, Vector{Float64}}()
    for (i, combo) in enumerate(parcombos)
        paramdict[tuple(combo...)] = params[i]
    end
    return DiscreteCPT(childrange, paramdict)
end
=#