export
    WindowFilter,
    SyncPF,
    AsyncPF,
    CoherentPF

"""
    struct WindowFilter <: Filter

    General construction for a filter based on a flexible windowing scheme.

#arguments
    window_creator Defines the method used to create windows
    inference_algorithm Defines the algorithm to use on a window
    postprocess! A postprocessing function, that takes the runtime and does any additional processing needed to carry to the next iteration. Defaults to doing nothing.
    
"""
mutable struct WindowFilter <: Filter
    window_creator :: WindowCreator
    inference_algorithm :: InstantAlgorithm
    postprocess! :: Function
    latest_window :: Union{InstantRuntime, Nothing}
    WindowFilter(wc, ia, pp) = new(wc, ia, pp, nothing)
    WindowFilter(wc, ia) = new(wc, ia, run -> nothing, nothing)
end

function init_filter(wf::WindowFilter, dynrun::DynamicRuntime)
    ensure_all!(dynrun, current_time(dynrun))
    instrun = initial_instant_runtime(dynrun)
    # We assume no evidence or interventions at time 0
    # TODO: Handle placeholder beliefs
    infer(wf.inference_algorithm, instrun)
    wf.latest_window = instrun
    retrieve_values_from_instant_runtime!(dynrun, instrun)
    _store_beliefs(wf, dynrun, instrun)
end

function _store_beliefs(wf::WindowFilter, dynrun::DynamicRuntime{T}, instrun::InstantRuntime) where T
    dynnet = get_network(dynrun)
    for instvar in get_variables(get_network(instrun))
        instinst = current_instance(instrun, instvar)
        belief = marginal(wf.inference_algorithm, instrun, instinst)
        (dynname, t) = dynamic_name_and_time(instvar, T)
        dyninst = get_instance(dynrun, get_node(dynnet, dynname), t)
        set_value!(dynrun, dyninst, :belief, belief)
    end
end

function filter_step(wf::WindowFilter, dynrun::DynamicRuntime{T}, variables::Vector{<:Variable}, time::T, evidence::Dict{Symbol, Score}) where T
    dynnet = get_network(dynrun)
    insts = create_window(wf.window_creator, dynrun, variables, time)
    instrun = instant_runtime_from_instances(dynrun, insts)
    # Apply the dynamic evidence to the instant runtime
    instev = Dict{Symbol, Score}()
    for (name, sc) in evidence
        instev[instant_name(name, time)] = sc
    end
    # Apply beliefs in the dynamic network as placeholder beliefs in the instant network.
    placeholder_beliefs = Dict{Symbol,Dist}()
    inst_phs = get_placeholders(get_network(instrun))
    for instnode in inst_phs
        (dynname, t) = dynamic_name_and_time(instnode, T)
        dynnode = get_node(dynnet, dynname)
        dyninst = get_instance(dynrun, dynnode, t)
        belief = get_value(dynrun, dyninst, :belief)
        placeholder_beliefs[get_name(instnode)] = belief
    end
    # TODO: Handle interventions
    instinterv = Dict{Symbol,Dist}()
    infer(wf.inference_algorithm, instrun, instev, instinterv, placeholder_beliefs)
    wf.latest_window = instrun
    retrieve_values_from_instant_runtime!(dynrun, instrun)
    set_time!(dynrun, time)
    _store_beliefs(wf, dynrun, instrun)
end

function answer(::Marginal, ::WindowFilter, dynrun::Runtime, target::VariableInstance) 
    return get_value(dynrun, target, :belief)
end

# function answer(query::Query, wf::WindowFilter, dynrun::Runtime, targets::Vector{VariableInstance}) 
#     # TODO: This code assumes that targets are in the latest_window, which might not be true
#     # for an asynchronous filter. We need to construct an instant window for the targets.
#     # For that, we need to put information from the dynamic window into the instant window,
#     # which is not done yet.
#     instrun = wf.latest_window
#     insttargets = VariableInstance[]
#     for target in targets
#         instname = instant_name(get_name(target), current_time(dynrun))
#         instnode = get_node(get_network(instrun), instname)
#         insttarget = current_instance(instrun, instnode)
#         push!(insttargets, insttarget)
#     end
#     answer(query, wf.inference_algorithm, wf.latest_window, insttargets)
# end
