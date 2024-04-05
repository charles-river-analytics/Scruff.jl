export
    WindowCreator,
    create_window,
    SyncWindow,
    AsyncWindow,
    CoherentWindow

"""
    abstract type WindowCreator end

A type that identifies how to create windows of instances for filtering algorithms.

T represents the time type.

Must implement `create_window`.
"""

abstract type WindowCreator{T} end

struct SyncWindow <: WindowCreator{Int} end

"""
    create_window(::SyncWindow, runtime::Runtime, variables::Vector{<:Variable}, time::Int)::Vector{Instance}

Creates a window by instantiating all variables at all intermediate time steps from the earliest parent to the given time.
The `variables` argument is ignored.
"""
function create_window(::SyncWindow, runtime::Runtime, variables_to_sample::Vector{<:Variable}, time::Int)::Vector{Instance} 
    prevtime = time
    net = get_network(runtime)
    ord = topsort(get_transition_graph(net))
    for v in variables_to_sample
        pars = get_transition_parents(net, v)
        for p in pars
            time_offset = has_timeoffset(net, v, p)
            t = get_time(latest_instance_before(runtime, p, time, !time_offset))
            prevtime = min(prevtime, t)
        end
    end

    insts = Instance[]
    for n in ord
        ph = Placeholder{output_type(n)}(n.name)
        push!(insts, PlaceholderInstance(ph, prevtime))
    end
    for t in prevtime+1:time
        for n in ord
            push!(insts, ensure_instance!(runtime, n, t))
        end
    end
    return insts
end

struct AsyncWindow{T <: Number} <: WindowCreator{T} end

"""
    create_window(::AsyncWindow, runtime::Runtime, variables::Vector{<:Variable}, time::Int)::Vector{Instance}

Creates a window by instantiating only the given variables at the given time.
"""

function create_window(::AsyncWindow{T}, runtime::Runtime, variables::Vector{<:Variable}, time::T)::Vector{Instance} where T
    insts = Instance[]
    done = Set{Variable}()
    for v in variables
        for p in get_transition_parents(get_network(runtime), v)
            if !(p in done)
                parinst = latest_instance_before(runtime, p, time, true) # changed false to true!!!
                partime = get_time(parinst)
                ph = Placeholder{output_type(p)}(p.name)
                phinst = PlaceholderInstance(ph, partime)
                push!(insts, phinst)
                push!(done, p)
            end
        end
        push!(insts, ensure_instance!(runtime, v, time))
        push!(done, v)
    end
    return insts
end
        
"""
    struct CoherentWindow <: WindowCreator end
    
A variant of AsyncWindow that ensures that parent values are never stale for any variable that
gets updated in a filter step. In other words, if any parent of a direct parent has been updated more recently than a variable
to be updated, the direct parent is added to the variables to be updated. This condition then recurses for the direct parents.
"""
struct CoherentWindow{T <: Number} <: WindowCreator{T} end

function create_window(::CoherentWindow{T}, runtime::Runtime, variables::Vector{<:Variable}, time::T)::Vector{Instance} where T
    # Note: This method does not allow placeholder parents of dynamic variables
    net = get_network(runtime) 
    parents = get_transition_graph(net)
    fullvars = Set{Variable}()
    order = topsort(parents)
    times = Dict([(n, get_time(current_instance(runtime, n))) for n in order])

    function ensure(v)
        if !(v in fullvars)
            push!(fullvars, v)
            for anc in ancestors(parents, v, Set{Node}())
                for grandanc in get(parents, anc, [])
                    if times[grandanc] > times[anc]
                        ensure(anc)
                        break  
                    end
                end
            end
            times[v] = time # Need to do this to ensure instantiation through a chain of dependencies
        end
    end

    for var in variables
        ensure(var)
    end
    # We must make sure variables get sampled in the correct order to maintain coherence
    orderedvars = Variable[]
    for v in order
        if v in fullvars
            push!(orderedvars, v)
        end
    end

    create_window(AsyncWindow{T}(), runtime, orderedvars, time)
end
