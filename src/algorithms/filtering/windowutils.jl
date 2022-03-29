export 
    instant_runtime_from_instances, 
    retrieve_values_from_instant_runtime!,
    initial_instant_runtime,
    instant_name,
    dynamic_name_and_time,
    instant_node

"""
Create a name in an instant network corresponding to the given dynamic name and time.
"""
function instant_name(dynamic_name::Symbol, time::Number)::Symbol
    return Symbol(dynamic_name, "_", time)
end

"""
Create a dynamic name and time from an instant node. T is the time type.
"""
function dynamic_name_and_time(instant_node::Node, T = Int)::Tuple{Symbol, T} 
    instant_name = collect(repr(instant_node.name))
    # Need to handle two different representations of symbols
    if length(instant_name) >= 6 && instant_name[1:6] == ['S', 'y', 'm', 'b', 'o', 'l']
        instant_name = instant_name[9:length(instant_name)-2] # strip parens and quotes
    else
        instant_name = instant_name[2:length(instant_name)] # strip leading colon
    end
    len = length(instant_name)
    i = findlast(x -> x == '_', instant_name)
    dynamic_name = Symbol(String(instant_name[1:i-1]))
    time = parse(T, String(instant_name[i+1:len]))
    return (dynamic_name, time)
end

"""
Create an instant node from a dynamic variable instance.
"""
function instant_node(dyninst::VariableInstance)
    name = instant_name(get_name(dyninst), get_time(dyninst))
    model = SimpleModel(get_sfunc(dyninst))
    var = Variable(name, model)
    return var
end

"""
Create an instant node from a dynamic placeholder instance.
"""
function instant_node(dyninst::PlaceholderInstance{O}) where O
    name = instant_name(get_name(dyninst), get_time(dyninst))
    ph = Placeholder{O}(name)
    return ph
end


"""
    instant_runtime_from_instances(runtime::DynamicRuntime, instances::Vector{Instance})

Create an instant runtime from the given instances in the given dynamic runtime.

This runtime has an instant network that contains a variable for each instance in `insts`, 
tagged with the time of the instance.
The network also contains a placeholder for each instance in `placeholder_insts`.
The function also instantiates the variables in the instant runtime and stores any runtime 
values from the dynamic runtime with the corresponding instances in the instant runtime.
This function is useful for running instant algorithms on a time window 
for dynamic reasoning.
"""
function instant_runtime_from_instances(dynrun::DynamicRuntime, dyninsts::Vector{Instance})
    dynnet = get_network(dynrun)
    forward_index = Dict{Instance, Node}()
    back_index = Dict{Node, Instance}()
    placeholders = Placeholder[]
    variables = Variable[]
    nodes = Node[]

    for dyninst in dyninsts
        node = instant_node(dyninst)
        if get_node(dyninst) isa Variable
            push!(variables, node)
        else
            push!(placeholders, node)
        end
        forward_index[dyninst] = node
        back_index[node] = dyninst
        push!(nodes, node)
    end

    instgraph = VariableGraph()
    for node in nodes
        nodepars = Node[]
        dyninst = back_index[node]
        insttime = get_time(dyninst)
        dynnode = get_node(dyninst)
        dynpars = get_transition_parents(dynnet, dynnode)
        for dynpar in dynpars
            # Find the most recent parent instance equal or before this variable's instance
            parinst = latest_instance_before(dynrun, dynpar, insttime, dynpar != dynnode)
            if isnothing(parinst)
                error("Variable does not have parent in instances")
            elseif !(parinst in dyninsts)
                # should be a placeholder
                ph = Placeholder{output_type(dynpar)}(get_name(dynpar))
                parinst = PlaceholderInstance(ph, get_time(parinst))
            end
            push!(nodepars, forward_index[parinst])
        end
        instgraph[node] = nodepars
    end

    instnet = InstantNetwork(variables, instgraph, placeholders)
    instrun = Runtime(instnet)
    ensure_all!(instrun)

    for ((dyninst, valuename), value) in dynrun.values
        if dyninst in dyninsts
            instinst = current_instance(instrun, forward_index[dyninst])
            set_value!(instrun, instinst, valuename, value)
        end
    end

    for (k,v) in get_state(dynrun)
        set_state!(instrun, k, v)
    end

    return instrun
end

"""
    Creates an instant runtime for the first time step.
"""
function initial_instant_runtime(dynrun::DynamicRuntime)
    variables = Variable[]
    placeholders = Placeholder[]
    dynnet = get_network(dynrun)
    node_index = Dict{Symbol, Node}() # Maps names of dynamic nodes to instant instances
    instgraph = VariableGraph()
    for dynnode in topsort(get_initial_graph(dynnet))
        nodename = get_name(dynnode)
        dyninst = current_instance(dynrun, dynnode)
        instvar = instant_node(dyninst)
        if instvar isa Variable
            push!(variables, instvar)
        else
            push!(placeholders, instvar)
        end
        node_index[nodename] = instvar
        dynpars = get_initial_parents(dynnet, dynnode)
        instpars = Node[]
        for dynpar in dynpars
            instpar = node_index[get_name(dynpar)]
            push!(instpars, instpar)
        end
        instgraph[instvar] = instpars
    end
    instnet = InstantNetwork(variables, instgraph, placeholders)
    instrun = Runtime(instnet)
    for node in values(node_index)
        instantiate!(instrun, node, 0)
    end
    return instrun
end

"""
    retrieve_values_from_instant_runtime!(dynrun::DynamicRuntime, instrun::InstantRuntime)

    Retrieve values in a dynamic runtime from an instant runtime constructed
    using `instant_runtime_from_instances`.
"""
function retrieve_values_from_instant_runtime!(dynrun::DynamicRuntime{T}, 
        instrun::InstantRuntime) where T
    index = Dict{Instance, Instance}()
    for node in get_nodes(get_network(instrun))
        # If instrun has been constructed correctly from dynrun,
        # there should be no runtime errors in this code.
        instinst = current_instance(instrun, node)
        (dynname, dyntime) = dynamic_name_and_time(node, T)
        dynnode = get_node(get_network(dynrun), dynname)
        dyninst = get_instance(dynrun, dynnode, dyntime)
        index[instinst] = dyninst
    end

    # TODO: This assumes that the values in the dynamic runtime take the same value as in the instant runtime.
    # This is not necessarily the case, e.g. for :particles, which contains samples that are dictionaries from node name to 
    # node value. The node names in the instant runtime and dynamic runtime are different. 
    # For this, importance stores beliefs with nodes in the instant runtime that can get translated.
    # A better general method is needed. 
    for ((instinst, valuename), value) in instrun.values
        set_value!(dynrun, index[instinst], valuename, value)
    end
    for (k, v) in get_state(instrun)
        set_state!(dynrun, k, v)
    end
end    

