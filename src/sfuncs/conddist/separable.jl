export 
    Separable,
    SepCPTs

"SepCPTs = Array{Dict{I, Array{Float64, 1}} where I}"
SepCPTs = Array{Dict{I, Array{Float64, 1}} where I}

# TODO: Generalize this to separable additive decompositions in general, not just CPTs
"""
    function Separable(range::Vector{O}, probabilities :: Vector{Float64}, compparams :: SepCPTs) where O

Constructs an sfunc representing separable models, defined by additive decompositon of a conditional probability distribution into
separate distributions depending on each of the parents.

`Separable`s are implemented as a `Mixture` of `Extend` sfuncs that extend `DiscreteCPT`s.

To construct a `Separable`, this method is passed the `range` of output values, the `probabilities`
of each of the underlying `DiscreteCPT` (which are the internal sfuncs of the `Mixture`), and
the parameters for each of the `DiscreteCPT`s.  For example, 

```
alphas = [0.2, 0.3, 0.5]
cpd1 = Dict((1,) => [0.1, 0.9], (2,) => [0.2, 0.8])
cpd2 = Dict((1,) => [0.3, 0.7], (2,) => [0.4, 0.6], (3,) => [0.5, 0.5])
cpd3 = Dict((1,) => [0.6, 0.4], (2,) => [0.7, 0.3])
cpds :: Array{Dict{I,Array{Float64,1}} where I,1} = [cpd1, cpd2, cpd3]
s = Separable([1, 2], alphas, cpds)
```

See also:  [`Mixture`](@ref), [`Extend`](@ref), [`DiscreteCPT`](@ref), [`Table`](@ref)
"""
function Separable(range::Vector{O}, probabilities :: Vector{Float64}, compparams :: SepCPTs) where O
    N = length(compparams)
    @assert length(probabilities) == N

    IS = []
    for i = 1:N
        ks = collect(keys(compparams[i]))
        push!(IS, typeof(ks[1][1]))
    end
    J = Tuple{IS...}

    # Explicit typing is necessary to ensure that the cpts passed to Mixture all have the same type.
    function make_cpt(i,I)::SFunc{J,O}
        cpt::Table{1, I, Tuple{}, I, O, <:Dist{O}} =
           DiscreteCPT(range, compparams[i])
        extended::SFunc{J,O} = Extend(J, cpt, i)
        return extended
    end

    cpts::Vector{SFunc{J,O}} = 
        [make_cpt(i,Tuple{IS[i]}) for i in 1:N]
    return Mixture(cpts, probabilities)
end
