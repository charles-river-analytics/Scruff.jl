export
    InstantAlgorithm,
    infer

"""
    InstantAlgorithm

    Algorithm that runs once on an `InstantNetwork`.
"""
abstract type InstantAlgorithm <: Algorithm end

# FIXME:
# placeholder_beliefs is intended to provide information about the distribution of placeholders to an algorithm.
# The current interface allows marginal distributions over placeholders to be passed, which is fine for an
# algorithm like BP, but if an algorithm needs joint information it has to get it in a different way.
# In particular, during particle filtering, the importance sampling algorithm needs joint samples of the previous state.
# Fortunately, it can get it out of the :particles state in the runtime, but this is clunky and not in line with the
# intentions of the general infer method.
# We should introduce sfuncs that define distributions over dictionaries, and provide operators to produce joint samples
# or compute marginals over individual variables, as they are able to support them. placeholder_beliefs should be
# such an sfunc.
"""
    infer(algorithm::InstantAlgorithm, runtime::InstantRuntime,
        evidence::Dict{Symbol, <:Score},
        interventions::Dict{Symbol, <:Dist},
        placeholder_beliefs::Dict{Symbol, <:Dist})

Run the inference algorithm.

Stores the results in `runtime`. The format of these results is up to 
`algorithm`, but they should be usable by queries with this algorithm.

# Arguments
- `algorithm`: The instant algorithm to run.
- `runtime`: The runtime in which to run the algorithm.
- `evidence`: The supplied evidence, which defaults to `Dict()`. 
- `interventions`: The supplied interventions, which defaults to `Dict()`. 
- `placeholder_beliefs`: The beliefs associated with the placeholders in the 
network, which default to `Dict()`. Instant algorithms might require that a belief be
supplied for every placeholder in `network`.
"""
