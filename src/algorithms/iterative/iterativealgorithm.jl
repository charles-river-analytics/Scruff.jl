export
    IterativeAlgorithm,
    prepare,
    refine

"""
    abstract type IterativeAlgorithm <: InstantAlgorithm

Algorithm that runs iteratively on an `InstantNetwork`.

The algorithm should support two methods: `prepare` and `refine`.

An IterativeAlgorithm is also trivially an InstantAlgorithm where
`Infer` is implemented by calling `prepare` and `refine` once.
"""
abstract type IterativeAlgorithm <: InstantAlgorithm end

"""
    prepare(algorithm::IterativeAlgorithm, runtime::InstantRuntime
        evidence::Dict{Symbol, <:Score}, 
        interventions::Dict{Symbol, <:Dist},
        placeholder_beliefs::Dict{Symbol, <:Dist})

Prepare the inference algorithm for iteration.

Stores the algorithm state in `runtime`. 

# Arguments
- `algorithm`: The iterative algorithm to run.
- `runtime`: The runtime in which to run the algorithm.
- `evidence`: The supplied evidence, which defaults to `Dict()`. 
- `interventions`: The supplied interventions, which defaults to `Dict()`. 
- `placeholder_beliefs`: The beliefs associated with the placeholders in the 
network, which default to `Dict()`. 
"""
function prepare(algorithm::InstantAlgorithm, runtime::InstantRuntime, placeholder_beliefs::Vector{<:Dist},
    evidence::Vector{Tuple{Symbol, <:Score}}, interventions::Vector{Tuple{Symbol, <:Dist}})
end

"""
    refine(algorithm::IterativeAlgorithm, runtime::InstantRuntime)

Perform the next iteration of the algorithm.

Uses the algorithm state stored in `runtime` and stores the next state in `runtime`.
"""
function refine(algorithm::IterativeAlgorithm, runtime::InstantRuntime) end

function infer(algorithm::IterativeAlgorithm, runtime::InstantRuntime, 
    evidence::Dict{Symbol, <:Score}, interventions::Dict{Symbol, <:Dist}, placeholder_beliefs::Dict{Symbol, <:Dist})
    prepare(algorithm, runtime, placeholder_beliefs, evidence, interventions)
    refine(algorithm, runtime)
end
