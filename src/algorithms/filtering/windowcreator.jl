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
    previous_time = time
    net = get_network(runtime)
    order = topsort(get_transition_graph(net))

    function get_previous_instances()
        previous_instances = Dict{Symbol, Instance}()
        for node in order
            for parent in get_transition_parents(net, node)
                time_offset = has_timeoffset(net, node, parent)
                # time_offset = true
                # Assuming init_filter has been called, previous_instance should exist
                previous_instance = latest_instance_before(runtime, parent, time, !time_offset)
                if isnothing(previous_instance)
                    placeholder = Placeholder{output_type(parent)}(get_name(parent))
                    previous_instance = latest_instance_before(runtime, placeholder, time, !time_offset)
                end
                # previous_instance = get_instance(runtime, parent, time)
                previous_instances[get_name(previous_instance)] = previous_instance
            end
        end
        previous_instances
    end

    previous_instances = get_previous_instances()
    previous_time = time
    for inst in values(previous_instances)
        previous_time = min(previous_time, get_time(inst))
    end

    #   function get_window_start_time()
    #     for variable in variables_to_sample
    #         for parent in parents
    #             previous_time = min(previous_time, time)
    #         end
    #     end
    #     previous_time
    # end

    # previous_time = get_window_start_time()

    function create_placeholders()
        placeholders = Dict{Symbol, Instance}()
        for node in order
            name = get_name(node)
            placeholder = Placeholder{output_type(node)}(name)
            if has_instance(runtime, node, previous_time)
                remove_instance!(runtime, node, previous_time) # We must replace the variable with the placeholder, otherwise we run into problems
                # Careful: We must make sure placeholders can be used where variables are expected

            end
            new_instance = instantiate!(runtime, placeholder, previous_time)
            placeholders[name] = new_instance
        end
        placeholders
    end

    new_instances_dict = create_placeholders()
    all_instances = collect(values(new_instances_dict))

    function copy_values_into_placeholders()
        for ((previous_instance, value_name), value) in runtime.values
            name = get_name(previous_instance)
            # previous_instance = previous_instances[name]
            new_instance = new_instances_dict[name]
            set_value!(runtime, new_instance, value_name, value)
        end
    end

    copy_values_into_placeholders()
        
    function fill_in_gaps()
        for t in previous_time+1:time
            for node in order
                if !has_instance(runtime, node, t)
                    push!(all_instances, ensure_instance!(runtime, node, t))
                else
                    push!(all_instances, get_instance(runtime, node, t))
                end
            end
        end
    end    
            
    fill_in_gaps()

    all_instances
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
                time_offset = has_timeoffset(runtime.network, v, p)
                # time_offset = true
                parinst = latest_instance_before(runtime, p, time, !time_offset) 
                # parinst = get_instance(runtime, p, time)
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
