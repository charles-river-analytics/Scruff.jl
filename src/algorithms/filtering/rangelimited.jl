export 
    RangeLimited,
    limit_and_set_beliefs!

mutable struct RangeLimited <: Filter
    inner_filter :: Filter
    limits :: Dict{Symbol, Integer}
end

function init_filter(rl::RangeLimited, dr::DynamicRuntime)
    init_filter(rl.inner_filter, dr)
end

function filter_step(rl::RangeLimited, dynrun::DynamicRuntime{T}, variables::Vector{<:Variable}, time::T, evidence::Dict{Symbol, Score}) where T
    wf = rl.inner_filter
    instrun = create_instant_runtime(wf, dynrun, variables, time)
    infer_with_instant_runtime(wf, dynrun, instrun, time, evidence)
    limit_and_set_beliefs!(instrun, rl.limits, T) # Intercept ordinary inference with the inner filter here to limit the ranges.
    restore_dynamic_runtime(wf, dynrun, instrun, time)
end

function limit_and_set_beliefs!(runtime::Runtime, limits::Dict{Symbol, Integer}, timetype) 
    network = get_network(runtime)
    ranges = Dict{Symbol, Vector{T} where T}()
    for node in topsort(get_initial_graph(network))
        set_range_and_belief!(runtime, network, node, ranges, limits, timetype)
    end
end

function set_range_and_belief!(runtime, network, node, ranges, limits, timetype) 
    # The keys of limits are based on the plain dynamic names and don't have the underscores of instant names
    (key, _) = dynamic_name_and_time(node, timetype)
    if key in keys(limits)
        instance = current_instance(runtime, node)
        name = get_name(node)
        set_range_and_belief_limited!(runtime, instance, name, ranges, limits[key])
    end
end

function set_range_and_belief_limited!(runtime, instance, name, ranges, limit)
    if has_belief(runtime, instance)
        belief = get_belief(runtime, instance)
        samples = limited_support(belief, (), limit)
        ranges[name] = samples 
        new_belief = Cat(samples, [1.0 / length(samples) for s in samples])
        post_belief!(runtime, instance, new_belief)
    end
end

function answer(q::Query, rl::RangeLimited, r::Runtime, i::VariableInstance)
    is = VariableInstance[i]
    answer(q, rl.inner_filter, r, is)
end

function probability(rl::RangeLimited, runtime::Runtime, item::Queryable, predicate::Function)
    probability(rl.inner_filter, runtime, item, predicate)
end

function probability(rl::RangeLimited, runtime::Runtime, item::Queryable{O}, value::O) where O
    probability(rl.inner_filter, runtime, item, value)
end

function expectation(rl::RangeLimited, run::Runtime, item::Queryable, fn::Function)::Float64
    expectation(rl.inner_filter, run, item, fn)
end
