export
    Instance,
    VariableInstance,
    PlaceholderInstance,
    Env,
    Runtime,
    InstantRuntime,
    DynamicRuntime,
    collect_messages,
    current_instance,
    current_time,
    get_all_instances,
    get_belief,
    get_placeholder_beliefs,
    get_definition,
    get_env,
    get_evidence,
    get_instance,
    get_intervention,
    get_message,
    get_name,
    get_network,
    get_range,
    get_sfunc,
    get_state,
    get_time,
    get_value,
    get_node,
    has_belief,
    has_evidence,
    has_instance,
    has_intervention,
    has_previous_instance,
    has_range,
    has_state,
    has_value,
    previous_instance,
    rng,
    clear_state!,
    concat!,
    delete_evidence!,
    delete_state!,
    delete_value!,
    distribute_messages!,
    instantiate!,
    ensure_all!,
    ensure_instance!,
    post_belief!,
    post_evidence!,
    post_intervention!,
    latest_instance_before,
    remove_messages!,
    set_message!,
    set_range!,
    set_state!,
    set_time!,
    set_value!

using DataStructures
using Dates
using Random

using Scruff.Models

import Base.isless

"""
    Instance{O}

An abstract base type for variable and placeholder instances.

This is instance can have values associated with it in the runtime.
O is the output type of the node.
"""
abstract type Instance{O} end

"""
    VariableInstance

An instance of variable node at `time`.   `sf` is an sfunc generated from 
the variable's model.
"""
struct VariableInstance{O} <: Instance{O}
    node :: Variable{I,J,O} where {I,J}
    sf :: Union{SFunc{I,O}, SFunc{J,O}} where {I,J}
    time :: Any
end

"""
    PlaceholderInstance

An instance of placeholder `node` at `time`.  This instance has no sfunc, but can still take 
values in the runtime.
"""
struct PlaceholderInstance{O} <: Instance{O}
    node :: Placeholder{O}
    time :: Any
end

"""
    get_sfunc(i::VariableInstance)::SFunc

Get the instance's sfunc.
"""
get_sfunc(i::VariableInstance)::SFunc = i.sf

""" 
    get_node(i::Instance)::Node

Get the instance's node, whether it is a placeholder or a variable.
"""
get_node(i::Instance)::Node = i.node

"""
    get_time(i::Instance)

Get the instance's time.
"""
get_time(i::Instance) = i.time

"""
    get_name(i::Instance)::Symbol

Get the name of the instance's node.
"""
get_name(i::Instance):: Symbol = get_node(i).name

"""
    get_definition(i::VariableInstance)::D where {D<:ValueTyped}

Get the instance's variable's underlying model.
"""
get_definition(i::VariableInstance)::D where {D<:ValueTyped} = get_variable(i).model

"""
    get_model(i::VariableInstance)::D where {D<:ValueTyped}

Get the instance's variable's underlying model.
"""
get_model(i::VariableInstance)::D where {D<:ValueTyped} = get_definition(i)

output_type(i::Instance) = output_type(get_node(i))

"""
    struct Env

Holds all external state of a Runtime.  The `Env` supports the following methods:

```
get_state(env::Env) :: Dict{Symbol, Any}
has_state(env::Env, key::Symbol) :: Bool
get_state(env::Env, key::Symbol)
set_state!(env::Env, key::Symbol, value)
delete_state!(env::Env, key::Symbol)
clear_state!(env::Env)
clone(env::Env)
```
"""
struct Env
    state::Dict{Symbol, Any}
    Env() = new(Dict{Symbol, Any}())
end

get_state(env::Env) :: Dict{Symbol, Any} = env.state
has_state(env::Env, key::Symbol) :: Bool = haskey(get_state(env), key)
function get_state(env::Env, key::Symbol)
    has_state(env, key) || ArgumentError("symbol :$key does not exist in the environment")
    get_state(env)[key]
end

set_state!(env::Env, key::Symbol, value) = get_state(env)[key] = value
delete_state!(env::Env, key::Symbol) = pop!(get_state(env), key, nothing)
clear_state!(env::Env) = empty!(get_state(env))

clone(env::Env) :: Env = deepcopy(env)

# TODO what should happen if a global and local key conflicts?
function concat!(local_env::Env, global_env::Env) :: Env
    for (k,v) in get_state(clone(global_env))
        set_state!(local_env, k, v)
    end
    local_env
end

"""
    abstract type Runtime

A struct that contains the state of the compute graph. This code makes the assumption
that values are associated with instances but messages are passed between
variables and applied to the relevant instances later.  It has to be this way
because the receiving instance might not exist at the time the message is sent.
"""
abstract type Runtime end

"""
    struct InstantRuntime <: Runtime

A runtime that represents a network whose variables take on a single instance.  As
a convenience, the following methods create an `InstantRuntime`:

```
Runtime()
Runtime(net :: InstantNetwork)
Runtime(net :: DynamicNetwork)
```
"""
struct InstantRuntime <: Runtime
    env :: Env
    name :: Symbol
    network :: InstantNetwork
    instances :: Dict{Node, Instance}
    values :: Dict{Tuple{Instance, Symbol}, Any}
    messages :: Dict{Tuple{Node, Symbol}, Dict{Node, Any}}
end

"""
    struct DynamicRuntime <: Runtime

A runtime that represents a network whose variables take on many instances at different times `T`.
"""
struct DynamicRuntime{T} <: Runtime
    env :: Env
    name :: Symbol
    network :: DynamicNetwork
    instances :: Dict{Node, 
        SortedDict{T, Instance, Base.Order.ReverseOrdering}}
    values :: Dict{Tuple{Instance, Symbol}, Any}
    messages :: Dict{Tuple{Node, Symbol}, Dict{Node, Any}}
end


Runtime() = InstantRuntime(InstantNetwork(Variable[], Placeholder[], Placeholder[], 
    VariableGraph()))
Runtime(net :: InstantNetwork) = InstantRuntime(Env(), gensym(), net, 
    Dict(), Dict(), Dict())
Runtime(net :: DynamicNetwork) = Runtime(net, 0)
function Runtime(net :: DynamicNetwork, time::T) where {T} 
    rt = DynamicRuntime{T}(Env(), gensym(), net, Dict(), Dict(), Dict())
    set_time!(rt, time)
    return rt
end

function rng(r::Runtime)
    return Random.GLOBAL_RNG
end

get_env(runtime::Runtime) :: Env = runtime.env
get_name(runtime::Runtime) :: Symbol = runtime.name
get_network(runtime::Runtime) :: Network = runtime.network
get_nodes(runtime::Runtime) :: Set{Node} = get_nodes(get_network(runtime))
function get_node(runtime::Runtime, name::Symbol) :: Union{Placeholder, Variable, Nothing}
    get_node(get_network(runtime), name)
end

#=
    state functions
=#

get_state(runtime::Runtime) :: Dict{Symbol, Any} = get_state(get_env(runtime))
has_state(runtime::Runtime, key::Symbol) :: Bool = has_state(get_env(runtime), key)
get_state(runtime::Runtime, key::Symbol) = get_state(get_env(runtime), key)
set_state!(runtime::Runtime, key::Symbol, value) = set_state!(get_env(runtime), key, value)
delete_state!(runtime::Runtime, key::Symbol) = delete_state!(get_env(runtime), key)
clear_state!(runtime::Runtime) = empty!(get_state(runtime))


#=
    Managing Runtime time
=#
const TIME = :__simulated_time__

"""
    current_time(runtime::Runtime) -> T

Returns the currently set time of the given `Runtime`
"""
function current_time(runtime::DynamicRuntime{T}) :: T where {T}
    return get_state(runtime, TIME)
end

"""
    set_time(runtime::Runtime{T}, newtime::T) -> T

Sets the current time for the given `Runtime`
"""
set_time!(runtime::DynamicRuntime{T}, newtime::T) where {T} = set_state!(runtime, TIME, newtime)

#=
    Instantiating instances
=#

isless(::Nothing, ::Nothing) = false

"""
    instantiate!(runtime::InstantRuntime,variable::Variable,time = 0)
    instantiate!(runtime::InstantRuntime,placeholder::Placeholder,time = 0)
    instantiate!(runtime::DynamicRuntime{T}, node::Node,time::T = current_time(runtime))::Instance where {T}

Instantiate and return an instance for the given runtime at the given time; the default
time is the current time of the runtime in unix time (an `Int64`). For an `InstantRuntime`, 
there is only a single instance for each variable.
"""
function instantiate!(runtime::InstantRuntime,variable::Variable,time = 0) 
        # time argument is provided for uniformity but ignored
    return get!(runtime.instances, variable, 
        VariableInstance(variable, make_initial(variable.model, time), time))
end

function instantiate!(runtime::InstantRuntime,placeholder::Placeholder,time = 0)
    return get!(runtime.instances, placeholder, PlaceholderInstance(placeholder, time))
end

function instantiate!(runtime::DynamicRuntime{T}, node::Node,time::T = current_time(runtime))::Instance where {T}
    if haskey(runtime.instances, node)
        curr = (runtime.instances[node])
        @assert !in(time, keys(curr)) "variable $(variable.name) at time $(time) is already instantiated"
        parents = get_transition_parents(runtime.network, node)
        parenttimes = Vector{T}()
        for p in parents 
            parinst = latest_instance_before(runtime, p, time, p != node)
            if isnothing(parinst)
                error("In instantiate! for ", variable.name, ": parent ", p.name, " not instantiated at time ", time)
            end
            push!(parenttimes, get_time(parinst))
        end
        parenttimes = tuple(parenttimes...)
        if isa(node, Variable)
            sf = make_transition(node.model, parenttimes, time)
            inst = VariableInstance(node, sf, time)
        else
            inst = PlaceholderInstance(node, time)
        end
        curr[time] = inst
        return inst
    else
        curr = 
            SortedDict{T, Instance, Base.Order.ReverseOrdering}
                (Base.Order.ForwardOrdering())
        local inst::Instance
        if isa(node, Variable)
            sf = make_initial(node.model, time)
            inst = VariableInstance(node, sf, time)
        else
            inst = PlaceholderInstance(node, time)
        end
        instvec = Pair{T, Instance}[time => inst]
        runtime.instances[node] = 
            SortedDict{T, Instance, Base.Order.ReverseOrdering}(Base.Order.ReverseOrdering(), instvec)
        return inst
    end
end

"""
    ensure_all!(runtime::InstantRuntime, time=0) :: Dict{Symbol, Instance}
    ensure_all!(runtime::DynamicRuntime, time = current_time(runtime)) :: Dict{Symbol, Instance}

Instantiate all the variables in the network at the given time and returns them as a
dict from variable names to their corresponding instance; the default time is the current
time of the runtime in unix time.
"""
function ensure_all!(runtime::InstantRuntime, time=0) :: Dict{Symbol, Instance}
    insts = Dict{Symbol, Instance}()
    for node in get_nodes(runtime)
        insts[node.name] = instantiate!(runtime, node)
    end
    return insts
end
    
function ensure_all!(runtime::DynamicRuntime, time = current_time(runtime)) :: Dict{Symbol, Instance}
    insts = Dict{Symbol, Instance}()
    for node in get_nodes(runtime)
        insts[node.name] = ensure_instance!(runtime, node, time)
    end
    return insts
end

"""
    ensure_instance!(runtime::Runtime{T}, node::Node{O}, time::T = current_time(runtime))::Instance{O} where {O,T}

Returns an instance for the given variable at the given time, either by using an existing one or creating a new one.
"""
function ensure_instance!(runtime::Runtime, node::Node{O}, time = current_time(runtime))::Instance{O} where O
    if has_instance(runtime, node, time)
        return get_instance(runtime, node, time)
    else
        return instantiate!(runtime, node, time)
    end
end

"""
    current_instance(runtime::InstantRuntime, node::Node)
    current_instance(runtime::DynamicRuntime, node::Node)

Returns the current (last) instance for the given runtime and node; this method will
throw an exception if there is no current instance for the given node
"""
current_instance(runtime::InstantRuntime, node::Node) = runtime.instances[node]

function current_instance(runtime::DynamicRuntime, node::Node) :: Instance
    first(runtime.instances[node]).second
end

"""
    previous_instance(runtime::DynamicRuntime, node::Node)

Returns the previous instance for the given runtime and node.  This will throw
and exception if there is no previous instance.
"""
function previous_instance(runtime::DynamicRuntime, node::Node)
    insts = runtime.instances[node]
    key = keys[insts][2]
    return insts[key]
end

"""
    get_all_instances(runtime::DynamicRuntime, variable::Variable)
    get_all_instances(runtime::InstantRuntime, variable::Variable)

Returns all instances in the `runtime` for the given variable, in order, as a
`Vector{Instance}`.
"""
function get_all_instances(runtime::DynamicRuntime, variable::Variable)
    has_instance(runtime, variable) ? collect(values(runtime.instances[variable])) : Instance[]
end

function get_all_instances(runtime::InstantRuntime, variable::Variable)
    has_instance(runtime, variable) ? [runtime.instances[variable]] : Instance[]
end

"""
    function get_instance(runtime::DynamicRuntime{T}, node::Node, t::T)::Instance

Returns instance for the given variable at time `T`; throws an error if no
such instance exists.
"""
function get_instance(runtime::DynamicRuntime{T}, node::Node, t::T) :: Instance where {T}
    return runtime.instances[node][t]
end

"""
    latest_instance_before(runtime::DynamicRuntime{T}, node::Node, t::T, allow_equal::Bool) :: Union{Instance, Nothing} where T

Return the latest instance of the node before the given time.  The `allow_equal` flag indicates 
whether instances at a time equal to `t`` are allowed. 

If there is no such instance, returns `nothing`.

"""
function latest_instance_before(runtime::DynamicRuntime{T}, node::Node, t::T, allow_equal::Bool) :: Union{Instance, Nothing} where T
    if !(node in keys(runtime.instances))
        return nothing
    end
    smd = runtime.instances[node]
    for u in keys(smd)
        if allow_equal ? u <= t : u < t
            return smd[u]
        end
    end
    return nothing
end
    
"""
    has_instance(runtime::DynamicRuntime, node::Node, time = current_time(runtime))
    has_instance(runtime::InstantRuntime, node::Node)

Returns true if the given node has an instance in the given runtime at a time greater than or equal to the given time.
"""
function has_instance(runtime::DynamicRuntime, node::Node, time = current_time(runtime))
    haskey(runtime.instances, node) && time in keys(runtime.instances[node])
end

has_instance(runtime::InstantRuntime, node::Node) = haskey(runtime.instances, node)

"""
    has_previous_instance(runtime::DynamicRuntime, node::Node)

Checks if the specified node has an instance prior to the current one.
"""
function has_previous_instance(runtime::DynamicRuntime, node::Node)
    haskey(runtime.instances, node) && length(keys(runtime.instances, node)) > 1
end

#=
    Setting and getting values associated with instances
=#

"""
    set_value!(runtime::Runtime, instance::Instance, key::Symbol, value)

Set the value on an instance for the given key
"""
function set_value!(runtime::Runtime, instance::Instance, key::Symbol, value)
    runtime.values[(instance, key)] = value
end

"""
    get_value(runtime::Runtime, instance::Instance, key::Symbol)

Get the value on an instance for the given key; this will throw an exception
if the instance does not contain the given key
"""
function get_value(runtime::Runtime, instance::Instance, key::Symbol)
    result = runtime.values[(instance, key)]
    return result
end

"""
    delete_value!(runtime::Runtime{T}, instance::Instance, key::Symbol) where {T}

Deletes the mapping for the given instance and key in the runtime and returns it
"""
function delete_value!(runtime::Runtime, instance::Instance, key::Symbol)
    return pop!(runtime.values, (instance, key), nothing)
end

function has_value(runtime::Runtime, instance::Instance, key::Symbol)
    (instance, key) in keys(runtime.values)
end

#=
    Sending and receiving messages between variables
=#

function set_message!(runtime::Runtime, sender::Node, recipient::Node, key::Symbol, value)
    msgs = get(runtime.messages, (sender, key), Dict())
    msgs[recipient] = value
    runtime.messages[(sender, key)] = msgs
end

function get_message(runtime::Runtime, sender::Node, recipient::Node, key::Symbol)
    runtime.messages[(sender, key)][recipient]
end

function distribute_messages!(runtime::Runtime, sender::Node, recipients::Vector{Node}, 
     key::Symbol, values)
    msgs = get(runtime.messages, (sender, key), Dict())
    for (rcp, val) in zip(recipients, values)
        msgs[rcp] = val
    end
    runtime.messages[(sender, key)] = msgs
end

function collect_messages(runtime::Runtime, senders::Vector{Node}, recipient::Node, 
        key::Symbol)
    [get_message(runtime, sdr, recipient, key) for sdr in senders]
end

function remove_messages!(runtime::Runtime, sender::Node, key::Symbol)
    delete!(runtime.messages, (sender, key))
end

#=
    Invoking an operation on an instance
=#

function _unwrap(p)
    if p isa Tuple && length(p) == 1
        p[1]
    else
        p
    end
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
    BELIEF

The constant key used to store belief for a specific variable instance
"""
const BELIEF = :__belief__

"""
    post_belief!(runtime::Runtime, inst::Instance, belief)

Posts the given evidence for the given instance.
"""
function post_belief!(runtime::Runtime, inst::Instance, belief)
    set_value!(runtime, inst, BELIEF, belief)
end

"""
    get_belief(runtime::Runtime, inst::Instance)

Returns the last posted belief for the given instance; this will return
`nothing` if no belief has been posted
"""
function get_belief(runtime::Runtime, inst::Instance)
    has_value(runtime, inst, BELIEF) || return nothing
    return get_value(runtime, inst, BELIEF)
end

has_belief(runtime::Runtime, inst::Instance) = has_value(runtime, inst, BELIEF)

function get_placeholder_beliefs(runtime::Runtime)::Dict{Symbol,Dist}
    result = Dict{Symbol,Dist}()
    for ph in get_placeholders(get_network(runtime))
        i = current_instance(runtime, ph)
        if has_belief(runtime, i)
            result[ph.name] = get_belief(runtime, i)
        end
    end
    return result
end
"""
    EVIDENCE

The constant key used to store evidence for a specific variable instance
"""
const EVIDENCE = :__evidence__

"""
    post_evidence!(runtime::Runtime, inst::Instance, evidence)

Posts the given evidence for the given instance.
"""
function post_evidence!(runtime::Runtime, inst::Instance, evidence::Score)
    set_value!(runtime, inst, EVIDENCE, evidence)
end

"""
    get_evidence(runtime::Runtime, inst::Instance)

Returns the last posted evidence for the given instance; this will return
`nothing` if no evidence has been posted
"""
function get_evidence(runtime::Runtime, inst::Instance)
    has_value(runtime, inst, EVIDENCE) || return nothing
    return get_value(runtime, inst, EVIDENCE)
end

function delete_evidence!(runtime::Runtime, inst::Instance) 
    if has_value(runtime, inst, EVIDENCE)
        return delete_value!(runtime, inst, EVIDENCE)
    end
end

has_evidence(runtime::Runtime, inst::Instance) = has_value(runtime, inst, EVIDENCE)

"""
    INTERVENTION

The constant key used to store interventions for a specific variable instance
"""
const INTERVENTION = :__intervention__

"""
    post_intervention!(runtime::Runtime, inst::Instance, intervention::Dist)

Posts the given intervention for the given instance.
"""
function post_intervention!(runtime::Runtime, inst::Instance, intervention::Dist)
    set_value!(runtime, inst, INTERVENTION, intervention)
end

"""
    get_intervention(runtime::Runtime, inst::Instance)

Returns the last posted intervention for the given instance; this will return
`nothing` if no intervention has been posted
"""
function get_intervention(runtime::Runtime, inst::Instance)
    has_value(runtime, inst, INTERVENTION) || return nothing
    return get_value(runtime, inst, INTERVENTION)
end

has_intervention(runtime::Runtime, inst::Instance) = has_value(runtime, inst, INTERVENTION)

#=
    Define the interface for Algorithms
=#

"""
    initialize_algorithm(alg_fun::Function, runtime::Runtime)

Initializes the given algorithm's [`Runtime`](@ref).  By default, this does nothing.
"""
initialize_algorithm(alg_fun::Function, runtime::Runtime) = nothing

"""
    start_algorithm(alg_fun::Function, runtime::Runtime, args...)

Starts the algorithm in the current thread
"""
start_algorithm(alg_fun::Function, runtime::Runtime, args...) = nothing

# TODO implement ; should take a list of instance/evidence pairs
# (or rather, variable-name/time/evidence tuples)
function post_evidence!(alg_fun::Function, runtime::Runtime, inst::Instance, evidence)
end

