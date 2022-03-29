export
    LazyInference,
    LazyState

"""
    mutable struct LazyState

Maintains the state of a lazy algorithm

# Fields

- `previous_algorithm`: The last instant algorithm used, if any
- `evidence`: The evidence supplied in `prepare`
- `interventions`: The interventions supplied in `prepare`
- `placeholder_beliefs`: The placeholder beliefs supplied in `prepare`
- `next_size`: The range size to use in the next call to `refine`
- `next_depth`: The depth to use in the next call to `refine`
- `next_iteration`: The number of the next iteration
- `is_complete`: A flag indicating whether the netwowk has been fully expanded
- `order`: The order of nodes used in computations
"""
mutable struct LazyState
    previous_algorithm :: Union{InstantAlgorithm, Nothing}
    evidence :: Dict{Symbol, Score}
    interventions :: Dict{Symbol, Dist}
    placeholder_beliefs :: Dict{Symbol, Dist}
    next_size :: Int
    next_depth :: Int
    next_iteration :: Int
    is_complete :: Bool
    order :: Vector{Node}
end

"""
    LazyState(ns, nd, ni, ic)

Intantiate `LazyState` with `next_size`, `next_depth`, `next_iterator`, and `is_complete`. 
"""
LazyState(ns, nd, ni, ic) = LazyState(nothing, Dict{Symbol,Score}(), Dict{Symbol,Dist}(), Dict{Symbol,Dist}(), ns, nd, ni, ic, Node[])

"""
    struct LazyInference <: IterativeAlgorithm

An iterative algorithm that expands recursively and increases the ranges of instances on every iteration.
"""
struct LazyInference <: IterativeAlgorithm
    algorithm_maker :: Function # A function that takes the current depth and range size and returns the instant algorithm to use
    increment :: Int
    start_size :: Int
    max_iterations :: Int
    start_depth :: Int
    state :: LazyState

    """
    function LazyInference(maker::Function; increment = 10, start_size = increment, max_iterations = 100, start_depth = 1)
    
    # Arguments
    - `maker``:  A function that takes a range size and expansion depth and returns an `InstantAlgorithm`
    - `increment`:  The increment to range size on every iteration
    - `start_size`:  The starting range size
    - `max_iterations`:  The maximum number of refinement steps
    - `start_depth`:  The depth of recursive expansion in the first iteration
    """
    function LazyInference(maker::Function;
        increment = 10, start_size = increment, max_iterations = 100, start_depth = 1)
        new(maker, increment, start_size, max_iterations, start_depth, 
            LazyState(start_size, start_depth, 1, false))
    end
end

function answer(query::Query, lazyalg::LazyInference, runtime::InstantRuntime, target::VariableInstance)
    state = lazyalg.state
    return answer(query, state.previous_algorithm, runtime, target)
end

function prepare(alg::LazyInference, runtime::InstantRuntime,
    evidence::Dict{Symbol, <:Score} = Dict{Symbol, Score}(), 
    interventions::Dict{Symbol, <:Dist} = Dict{Symbol, Dist}(),
    placeholder_beliefs = get_placeholder_beliefs(runtime))
    ensure_all!(runtime)
    net = get_network(runtime)
    # The evidence, interventions, and placeholder_beliefs are punted to refine to pass to the underlying algorithm
    state = alg.state
    state.evidence = evidence
    state.interventions = interventions
    state.placeholder_beliefs = placeholder_beliefs
    state.order = topsort(get_initial_graph(net))
    clear_analysis!()
end

function refine(lazyalg::LazyInference, runtime::InstantRuntime)
    state = lazyalg.state
    if !(state.is_complete || state.next_iteration > lazyalg.max_iterations)
        set_ranges!(runtime, state.evidence, state.next_size, state.next_depth, state.order)
        inferencealg = lazyalg.algorithm_maker(state.next_size, state.next_depth)
        state.previous_algorithm = inferencealg
        infer(inferencealg, runtime, 
            state.evidence, state.interventions, state.placeholder_beliefs)
        state.is_complete = _complete(runtime)
        state.next_iteration += 1
        state.next_depth += 1
        state.next_size += lazyalg.increment
    end
end

