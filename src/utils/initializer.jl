export
    default_initializer

using ..Operators

"""
Deprecated
"""
function default_initializer(runtime::InstantRuntime, default_range_size::Int=10,
        placeholder_beliefs :: Dict = Dict())
    ensure_all!(runtime)
    for (pname, bel) in placeholder_beliefs
        node = get_node(runtime, pname)
        inst = current_instance(runtime, node)
        post_belief!(runtime, inst, bel)
    end
    order = topsort(get_initial_graph(get_network(runtime)))
    set_ranges!(runtime, Dict{Symbol, Score}(), default_range_size, 1, order, placeholder_beliefs)
end


#=
function default_initializer(runtime::DynamicRuntime{T}, default_time::T, default_range_size::Int=10) where {T}
    ensure_all!(runtime, default_time)
    net = get_network(runtime)
    ord = topsort(get_transition_graph(net))
    ranges = Dict{Symbol, Array{Any, 1}}()
    for var in ord
        inst = current_instance(runtime, var)
        sf = get_sfunc(inst)
        parranges = collect(map(p -> ranges[p.name], get_parents(net, var)))
        rng = support(sf, Tuple(parranges), default_range_size, Vector{output_type(sf)}())
        set_range!(runtime, inst, rng)
        ranges[var.name] = rng
    end
end

function default_initializer(runtime::DynamicRuntime{Int})
    default_initializer(runtime, 0)
end
=#

function  preserve_evidence(evid_i, typeOfEvidence)
    preserve_values = Vector{typeOfEvidence}()
    if isa(evid_i, Dict) # soft evidence
        push!(preserve_values, collect(keys(evid_i)))
    elseif isa(evid_i, NoiseEvidence) # noise evidence
        push!(preserve_values, evid_i.mean)
    elseif isa(evid_i, Function)# hard evidence
        # do nothing
    else # hard evidence
        push!(preserve_values, evid_i)
    end
    return preserve_values
end
