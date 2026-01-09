# Restarting from now means starting a new filter from the current state of an existing filter.
export restart_from_now

using DataStructures

# To investigate: Why does ordinary filter not have memory leak but deep copying does cause that?
# Does it have to do with garbage collection?

function restart_from_now(filter::F, current_runtime::DynamicRuntime{T}, network::DynamicNetwork):: DynamicRuntime{T} where {T <: Number, F <: Filter}
    # Step 1: Create a new DynamicRuntime with the given network. The starting time of the new network will be the current time of the original network.
    time = current_time(current_runtime)
    new_runtime = Runtime(network, time)

    # We assume that init_filter has been called in the previous runtime, with zero or more filter steps.
    # Therefore, there is no need to call init_filter again. Instead, we:

    # Step 2: Instantiate all the nodes in the network in the new runtime. Each node should be instantiated at its most recent time of instantiation in the original network.
    for node in network.variables
        if !has_instance(current_runtime, node, 0) # The runtime is allowed to have the instance at any time, starting from 0
            error("Filter has not been properly initialized")
        end
        instance = latest_instance_before(current_runtime, node, time, true)
        instantiate!(new_runtime, node, get_time(instance))
    end

    # Step 3: Copy the values associated with nodes in the new runtime from the previous runtime
    for variable in get_variables(network)
        current_inst = current_instance(current_runtime, variable)
        new_inst = current_instance(new_runtime, variable)
        for (key, value) in get_all_values(current_runtime, current_inst)
            set_value!(new_runtime, new_inst, key, value)
        end
    end

    # Step 4: Copy all the messages and global state (because they are not timed).
    current_env = get_env(current_runtime)
    current_state = get_state(current_env)
    for (key, value) in current_state
        set_state!(new_runtime, key, value)
    end
    current_messages = current_runtime.messages
    for ((source, key), dest_and_value) in current_messages
        for (dest, value) in dest_and_value
            set_message!(new_runtime, source, dest, key, value)
        end
    end

    new_runtime
end
