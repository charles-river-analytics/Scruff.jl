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

function create_instant_runtime(wf, dynrun, variables, time) 
    insts = create_window(wf.window_creator, dynrun, variables, time)
    for inst in insts
        node = get_node(inst)
        ensure_instance!(dynrun, node, time)
    end
    instant_runtime_from_instances(dynrun, insts)
end

function compile_evidence(time, evidence)
    instev = Dict{Symbol, Score}()
    for (name, sc) in evidence
        instev[instant_name(name, time)] = sc
    end
    instev
end

function compile_placeholders(dynrun::DynamicRuntime{T}, instrun) where T
    dynnet = get_network(dynrun)
    instnet = get_network(instrun)
    placeholder_beliefs = Dict{Symbol,Dist}()
    inst_phs = get_placeholders(instnet)
    for instnode in inst_phs
        (dynname, t) = dynamic_name_and_time(instnode, T)
        dynnode = get_node(dynnet, dynname)
        if has_instance(dynrun, dynnode, t)
            dyninst = get_instance(dynrun, dynnode, t)
        else
            placeholder = Placeholder{output_type(dynnode)}(get_name(dynnode))
            dyninst = get_instance(dynrun, placeholder, t)
        end            
        belief = get_value(dynrun, dyninst, :belief)
        placeholder_beliefs[get_name(instnode)] = belief
    end
    placeholder_beliefs
end

function infer_with_instant_runtime(wf, dynrun, instrun, time, evidence)
    instev = compile_evidence(time, evidence)
    placeholder_beliefs = compile_placeholders(dynrun, instrun)
    # Apply beliefs in the dynamic network as placeholder beliefs in the instant network.
    # TODO: Handle interventions
    instinterv = Dict{Symbol,Dist}()
    infer(wf.inference_algorithm, instrun, instev, instinterv, placeholder_beliefs)
end

function restore_dynamic_runtime(wf, dynrun, instrun, time)
    wf.latest_window = instrun
    retrieve_values_from_instant_runtime!(dynrun, instrun)
    set_time!(dynrun, time)
    _store_beliefs(wf, dynrun, instrun)
end

function filter_step(wf::WindowFilter, dynrun::DynamicRuntime{T}, variables::Vector{<:Variable}, time::T, evidence::Dict{Symbol, Score}) where T
    instrun = create_instant_runtime(wf, dynrun, variables, time)

    infer_with_instant_runtime(wf, dynrun, instrun, time, evidence)

    restore_dynamic_runtime(wf, dynrun, instrun, time)
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
