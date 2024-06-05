export
    parent_ranges,
    get_range,
    has_range,
    set_range!,
    set_ranges!,
    expander_range

#####################################################################
#                                                                   #
# Algorithm building blocks for setting and expanding the ranges of #
# all the current instances in a runtime.                           #
#                                                                   #
# order is a topological sort of the network in the runtime.        #
# size is a uniform target size for each instance.                  #
#                                                                   #
#####################################################################

"""
    parent_ranges(runtime::Runtime, var::Variable{I,J,O}, depth = typemax(Int)) where {I,J,O}

Returns the ranges of the parents of the given variable.

See [`Scruff.get_range`](@ref)
"""
function parent_ranges(runtime::Runtime, var::Variable{I,J,O}, depth = typemax(Int)) where {I,J,O}
    net = get_network(runtime)
    pars = get_parents(net, var)
    result = []
    for p in pars
        pinst = current_instance(runtime, p)
        push!(result, get_range(runtime, pinst, depth))
    end
    T = isempty(result) ? Vector{O} : typejoin([typeof(x) for x in result]...)
    return convert(Vector{T}, result)
end


"""
    RANGE

The constant key used to store the range of a specific variable instance 
"""
const RANGE = :__range__

"""
    set_range!(runtime::Runtime, inst::Instance{O}, range::Vector{<:O}, depth::Int = 1) where O

Sets the range value for the given instance. Defaults to depth of 1.
"""
function set_range!(runtime::Runtime, inst::Instance{O}, range::Vector{<:O}, depth::Int = 1) where O
    if has_value(runtime, inst, RANGE)
        curr = get_value(runtime, inst, RANGE)
        s = Tuple{Int, Vector{O}}[]
        i = 1
        while i <= length(curr)
            pair = curr[i]
            d = pair[1]
            if d > depth
                push!(s, pair)
                i += 1
            end
            push!(s, (depth, range))
            i = d == depth ? i+1 : i
            for j = i:length(curr)
                push!(s, curr[j])
            end
            set_value!(runtime, inst, RANGE, s)
            return
        end
        push!(s, (depth, range))
        set_value!(runtime, inst, RANGE, s)
    else
        set_value!(runtime, inst, RANGE, [(depth, range)])
    end
end

"""
    get_range(runtime::Runtime, inst::Instance, depth = max_value(Int))

Returns the range value for the given instance; this will return
`nothing` if no range has been set.

The depth specifies the maximum depth of range desired.
"""
function get_range(runtime::Runtime, inst::Instance, depth = typemax(Int))
    has_range(runtime, inst, depth) || return nothing
    rng = get_value(runtime, inst, RANGE)
    for i in 1:length(rng)
        (d,r) = rng[i]
        if d <= depth
            return r
        end
    end
    return nothing
end

function has_range(runtime::Runtime, inst::Instance, depth::Int = typemax(Int)) 
    has_value(runtime, inst, RANGE) || return false
    r = get_value(runtime, inst, RANGE)
    (d,_) = r[length(r)]
    return d <= depth
end

"""
    get_range(runtime::DynamicRuntime, v::Variable{I,J,O}, depth = 1) where {I,J,O}

Returns the range of the most recent instance of the given variable.
"""
function get_range(runtime::DynamicRuntime, v::Variable{I,J,O}, depth = 1) where {I,J,O}
    inst = current_instance(runtime, v)
    range = get_range(runtime, inst, depth)
    if range !== nothing
        return range
    elseif has_previous_instance(runtime, v)
        prev = previous_instance(runtime, v)
        if has_range(runtime, prev)
            return get_range(runtime, prev, depth)
        else
            return O[]
        end
    else
        return O[]
    end
end

"""
    get_range(runtime::InstantRuntime, v::Node{O}, depth = 1) where O

Returns the range of the given node.
"""
function get_range(runtime::InstantRuntime, v::Node{O}, depth = 1) where O
    inst = runtime.instances[v]
    range = get_range(runtime, inst, depth)
    if range !== nothing
        return range
    else
        return O[]
    end
end

"""
    set_ranges!(runtime::InstantRuntime, evidence = Dict{Symbol, Score}(),
        size = 10, depth = 1, 
        order = topsort(get_initial_graph(get_network(runtime))),
        placeholder_beliefs = get_placeholder_beliefs(runtime))

Set the ranges of all current instances in the runtime.

This method first checks whether ranges exist for the runtime at the desired depth,
with the desired range size, and with the same evidence. If so, it doesn't do anything.
If the depth is less than 1, it doesn't do anything.
Otherwise, it uses the support operator to compute ranges of variables in order.
Placeholders should have ranges set already in `placeholder_beliefs`.
For expanders, it recursively sets the ranges of the subnetworks at depth - 1.

Returns a flag indicating whether any instance has a changed range.
"""
function set_ranges!(runtime::InstantRuntime, 
    evidence::Dict{Symbol, <:Score} = Dict{Symbol, Score}(),
    size :: Int = 10, depth :: Int = 1, 
    order = topsort(get_initial_graph(get_network(runtime))),
    placeholder_beliefs = get_placeholder_beliefs(runtime))
    if depth < 1 
        return false 
    end
    nodes = get_nodes(get_network(runtime))
    # If we require greater depth or greater size or different evidence, we have to do it again
    if has_state(runtime, :nodes) && has_state(runtime, :range_size) && has_state(runtime, :range_depth) && has_state(runtime, :range_evidence) &&
        get_state(runtime, :range_size) >= size  && get_state(runtime, :range_depth) >= depth && get_state(runtime, :range_evidence) == evidence &&
        get_state(runtime, :nodes) == nodes
        return false
    else
        set_state!(runtime, :range_size, size)
        set_state!(runtime, :range_depth, depth)
        set_state!(runtime, :range_evidence, evidence)
        set_state!(runtime, :nodes, nodes)
    end 
    changed = false
    for v in order 
        O = output_type(v)
        rng::Vector{O} = O[]
        curr = get_range(runtime, v, depth)
        inst = current_instance(runtime, v)
        if v isa Placeholder
            pi = placeholder_beliefs[v.name]
            rng = support(pi, (), size, output_type(v)[])
            chng = rng != curr
        else
            sf = get_sfunc(inst)
            if length(curr) < size
                parranges = parent_ranges(runtime, v, depth)
                if isa(sf, Expander)
                    rc = expander_range(runtime, v, size, depth)
                    rng= rc[1]
                    chng = rc[2]
                else
                    rng = support(sf, tuple(parranges...), size, curr)
                    chng = rng != curr
                end
                if v.name in keys(evidence)
                    ev = evidence[v.name]
                    if ev isa HardScore 
                        if !(ev.value in rng)
                            push!(rng, ev.value)
                            chng = true
                        end
                    elseif ev isa LogScore
                        for k in keys(ev.logscores)
                            if !(k in rng)
                                push!(rng, k)
                                chng = true
                            end
                        end
                    end
                end 
                changed = changed || chng
            else
                rng = curr
            end
        end
        set_range!(runtime, inst, rng, depth)
    end
    if changed
        delete_state!(runtime, :solution)
    end
    return changed
end

"""
    function expander_range(runtime :: Runtime, v :: Variable,
        target_size :: Int, depth :: Int)

    Recursively compute the range of the expander and subnetworks up to the given depth.

Computing the range of the expander expands enough of the parent          
range to reach the desired target size, or expands all the parents fully.
"""
function expander_range(runtime :: Runtime, v :: Variable,
                        target_size :: Int, depth :: Int)
    changed = false
    net = runtime.network
    parents = get_parents(net, v)
    parent = parents[1]
    parrange = get_range(runtime, parent, depth)
    rangeset = Set()
    subranges = Dict()
    for p in parrange
        if expanded(runtime, v, p)
            subnet = expansion(runtime, v, p)
            subrun = subruntime(runtime, v, subnet)
            subout = current_instance(subrun, get_node(subnet, :out))
            if has_range(subrun, subout, depth - 1)
                subrange = get_range(subrun, subout, depth - 1)
                subranges[p] = subrange
                union!(rangeset, subrange)
            else
                subranges[p] = []
            end
        else
            subranges[p] = []
        end
    end
    todo = Dict()
    for p in parrange
        todo[p] = length(subranges[p])
    end
    while length(rangeset) < target_size && !isempty(todo)
        p = argmin(todo)
        delete!(todo, p)
        (subnet, chng) = expand!(runtime, v, p)
        changed = chng
        subrun = subruntime(runtime, v, subnet)
        if depth > 1
            order = topsort(get_initial_graph(subnet))
            outvar = get_node(subrun, :out)
            # Cannot have evidence on subnet
            if set_ranges!(subrun, Dict{Symbol, Score}(), target_size - length(rangeset), depth - 1, order)
                changed = true
            end
            inst = current_instance(subrun, outvar)
            rng = get_range(subrun, inst, depth - 1)
            union!(rangeset, rng)
        end
    end
    result = output_type(v)[]
    for x in rangeset
        push!(result, x)
    end
    sort!(result)
    return (result, changed)
end
