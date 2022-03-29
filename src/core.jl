using Dates, Random, UUIDs
using Logging

import Base: convert

export 
    Model,
    Network,
    InstantNetwork,
    DynamicNetwork,
    Placeholder,
    Node,
    SFunc,
    Dist, 
    Score,
    ValueTyped,
    Variable,
    VariableGraph,
    get_initial_graph,
    get_transition_graph,
    get_children,
    get_initial_children,
    get_transition_children,
    get_parents,
    get_initial_parents,
    get_transition_parents,
    get_sfunc,
    get_node,
    get_variables,
    get_placeholders,
    get_nodes,
    input_type,
    make_initial,
    make_transition,
    output_type

"""
    abstract type ValueTyped{O}

Supertype for typing all Scruff Variables; `O` is the 
actual type of the variable
"""
abstract type ValueTyped{O} end

"""
value_type(v::ValueTyped{O}) where {O}

return the actual type (i.e. `O`) of the `ValueTyped`
"""
value_type(v::ValueTyped{O}) where {O} = O

"""
    abstract type SFunc{I<:Tuple, O}

A Stochastic Function type with input variables defined by `I` and output type `O`.  
This is an abstract representation for a collection of operators with the same 
input and output types.

All sfuncs have the following operators defined:
- `compute_lambda`
- `compute_bel`
- `send_pi`
- `outgoing_pis`
- `outgoing_lambdas`

`SFunc` _also_ has both the operators `cpdf` and `logcpdf` defined in terms of the other. 
All sfuncs should implement one or the other of these operators.
"""
abstract type SFunc{I<:Tuple, O} end

Base.print_without_params(x::Type{<:SFunc}) = false

"""
    Dist{T} = SFunc{Tuple{}, T}

# Additional supported operators
- `make_factors`
- `send_lambda`
"""
Dist{T} = SFunc{Tuple{}, T}

"""
    Score{I} = SFunc{Tuple{I}, Nothing}

`Score` supports two (2) operators:  `get_score` and `get_log_score`.  `get_log_score`
is defined, by default, using `get_score.`  Every subtype of `Score` _must_ implement
`get_score`.
"""
Score{I} = SFunc{Tuple{I}, Nothing}

"""
    input_type(::Type{<:SFunc{I,O}}) where {I,O}
    input_type(::SFunc{I,O}) where {I,O}

Return the input type (i.e. `I`) of the `SFunc`
"""
input_type(::Type{<:SFunc{I,O}}) where {I,O} = I
input_type(::SFunc{I,O}) where {I,O} = I
"""
    output_type(::Type{<:SFunc{I,O}}) where {I,O}
    output_type(::SFunc{I,O}) where {I,O}
    
Return the output type (i.e. `O`) of the `SFunc`
"""
output_type(::Type{<:SFunc{I,O}}) where {I,O} = O
output_type(::SFunc{I,O}) where {I,O} = O

"""
    Placeholder{O} <: ValueTyped{O}

A type for typing Scruff variables that do not reference models
"""
struct Placeholder{O} <: ValueTyped{O}
    name::Symbol
end

"""
    Model{I, J, O} <: ValueTyped{O}

Supertype for all Scruff models.
The model represents a variable that varies over time and has output type `O`.
The model may return an initial sfunc with input type `I` using `make_initial`,
which takes the current time as argument,
and a transition sfunc with input type `J` using `make_transition`,
which takes both the parent times (a tuple of times of the same length as `J`)
and the current time as arguments.
These two functions need to be defined for every sfunc.

# Type parameters
- `I`: the input type to the `SFunc` returned by the model's `make_initial` function 
- `J`: the input type to the `SFunc` used during the `make_trasition` function call
- `O`: the actual type of the variables represented by this model
"""
abstract type Model{I, J, O} <: ValueTyped{O} end
# is_fixed(m) = false

function make_initial(m,t) end

function make_transition(m, parenttimes, time) end

(model::Model)(symbol::Symbol) = instantiate(model, symbol)

"""
    instantiate(model::Model, name::Symbol)

Create a new `Variable` with the given name for the given model.  Every
`Model` instance is also a function that, given a `Symbol`, will call 
this method.
"""
function instantiate(model::Model, name::Symbol)
    return Variable(name, model)
end

"""
    mutable struct Variable{I,J,O} <: ValueTyped{O}

A Variable describes the (time-series) of some set of random values.
It must be named, described by a model, and references to the model inputs must be defined.

For the type variables, see [`Model`](@ref)
"""
mutable struct Variable{I,J,O} <: ValueTyped{O}
    name::Symbol
    model::Model{I,J,O}
end

"""`Node{O} = Union{Placeholder{O}, Variable{I,J,O} where {I,J}}`"""
Node{O} = Union{Placeholder{O}, Variable{I,J,O} where {I,J}}

"""
    output_type(::Variable{I,J,O})
    output_type(::Placeholder{O})

returns the output type `O` of the given parameter
"""
function output_type(::Variable{I,J,O}) where {I,J,O}
    return O
end

function output_type(::Placeholder{O}) where O
    return O
end

get_name(node::Node) = node.name

"""VariableGraph = Dict{Node, Vector{Node}}"""
VariableGraph = Dict{Node, Vector{Node}}

"""
    abstract type Network{I,J,O} 

Collects variables, with placeholders for inputs, and defined outputs,
along with up to two directed graphs for initial and transition. 
Must implement `get_nodes`, `get_outputs`,
`get_initial_graph`, `get_transition_graph` (which returns a `VariableGraph`)

For the type parameters, see the underlying [`Model`](@ref) class for the mapped
[`Variable`](@ref)s.
"""
abstract type Network{I,J,O} end

"""
    get_initial_parents(n::Network, node::Node)::Vector{Node}

Returns a list of the initial parent nodes in the network from the network's
initial graph (i.e. the return graph from get_initial_graph(network)).
"""
function get_initial_parents(n::Network, node::Node)::Vector{Node}
    g = get_initial_graph(n)
    get(g, node, Node[])
end

"""
    get_transition_parents(n::Network, node::Node)::Vector{Node}

Returns a list of transitioned parent nodes in the network from the network's 
transition graph (i.e. the return graph from get_transition_graph(network))
"""
function get_transition_parents(n::Network, node::Node)::Vector{Node}
    g = get_transition_graph(n)
    get(g, node, Node[])
end

"""
    get_node(n::Network, name::Symbol)::Union{Node, Nothing}

Returns the node with the given name, or `nothing`.
"""
function get_node(n::Network, name::Symbol)::Union{Node, Nothing}
    vs = get_nodes(n)
    for v in vs
        if v.name == name
            return v
        end
    end
    return nothing
end

"""
    get_initial_children(n::Network, var::Node)::Vector{Node}

Returns a list of the initial child nodes in the network from the network's
initial graph (i.e. the return graph from get_initial_graph(network)).   
"""
function get_initial_children(n::Network, var::Node)::Vector{Node}
    [c for c in get_nodes(n) if var in get_initial_parents(n, c)]
end

"""
    get_transition_children(n::Network, var::Node)::Vector{Node}

Returns a list of transitioned child nodes in the network from the network's 
transition graph (i.e. the return graph from get_transition_graph(network))    
"""
function get_transition_children(n::Network, var::Node)::Vector{Node}
    [c for c in get_nodes(n) if var in get_transition_parents(n, c)]
end

"""
    complete_graph!(variables::Vector{<:Variable}, graph::VariableGraph)

Populate the `graph` with `v=>Node[]` for all `variables` not in the graph. 
"""
function complete_graph!(variables::Vector{<:Variable}, graph::VariableGraph)
    ks = keys(graph)
    for v in variables
        if !(v in ks)
            graph[v] = Node[]
        end
    end
    return graph
end

"""
    struct InstantNetwork{I,J,O} <: Network{I,Nothing,O}

A network that only supports initial models.  The `Nothing` in the supertype
shows that there is no transition to another time.

See [`Network`](@ref) for the type parameters
"""
struct InstantNetwork{I,O} <: Network{I,Nothing,O}
    variables::Vector{<:Variable}
    placeholders::Vector{<:Placeholder}
    outputs::Vector{<:Variable} # outputs is a subset of variables
    parents::VariableGraph
    function InstantNetwork(variables::Vector{<:Variable}, parents::VariableGraph,
                placeholders::Vector{<:Placeholder} = Placeholder[], outputs::Vector{<:Variable} = Variable[])
        I = Tuple{[value_type(p) for p in placeholders]...}
        O = Tuple{[output_type(v) for v in outputs]...}
        return new{I,O}(variables, placeholders, outputs, complete_graph!(variables, parents))
    end
end

get_variables(n::InstantNetwork) = n.variables
get_placeholders(n::InstantNetwork) = n.placeholders
get_nodes(n::InstantNetwork) = union(Set(n.variables), Set(n.placeholders))
get_initial_placeholders(n::InstantNetwork) = n.placeholders
get_transition_placeholders(::InstantNetwork) = error("InstantNetwork does not have a transition graph")
get_outputs(n::InstantNetwork) = n.outputs
get_graph(n::InstantNetwork) = n.parents
get_initial_graph(n::InstantNetwork) = n.parents
get_transition_graph(::InstantNetwork) = error("InstantNetwork does not have a transition graph")
get_parents(n::InstantNetwork, v) = get_initial_parents(n,v)
get_children(n::InstantNetwork, v) = get_initial_children(n,v)

"""
    DynamicNetwork{I,J,O} <: Network{I,J,O}

A network that can transition over time.  
See [`Network`](@ref) for the type parameters.
"""
struct DynamicNetwork{I,J,O} <: Network{I,J,O}
    variables::Vector{<:Variable}
    initial_placeholders::Vector{<:Placeholder}
    transition_placeholders::Vector{<:Placeholder}
    outputs::Vector{<:Variable} # outputs is a subset of variables
    initial_parents::VariableGraph
    transition_parents::VariableGraph
    function DynamicNetwork(variables::Vector{<:Variable}, 
                initial_parents::VariableGraph, transition_parents::VariableGraph,
                init_placeholders::Vector{<:Placeholder} = Placeholder[], 
                trans_placeholders::Vector{<:Placeholder} = Placeholder[], 
                outputs::Vector{<:Variable} = Variable[])
        I = Tuple{[value_type(p) for p in init_placeholders]...}
        J = Tuple{[value_type(p) for p in trans_placeholders]...}
        O = Tuple{[output_type(v) for v in outputs]...}
        return new{I,J,O}(variables, init_placeholders, trans_placeholders, outputs, 
            complete_graph!(variables, initial_parents), complete_graph!(variables, transition_parents))
    end
end

get_variables(n::DynamicNetwork) = n.variables
get_initial_placeholders(n::DynamicNetwork) = n.initial_placeholders
get_transition_placeholders(n::DynamicNetwork) = n.transition_placeholders
get_nodes(n::DynamicNetwork) = 
    union(Set(n.variables), Set(n.initial_placeholders), Set(n.transition_placeholders))
get_outputs(n::DynamicNetwork) = n.outputs
get_initial_graph(n::DynamicNetwork) = n.initial_parents
get_transition_graph(n::DynamicNetwork) = n.transition_parents

function assign_times(parents::Dict{Variable, Vector{Node}},
                      timed_varplaces::Dict{Node,Time})::Dict{Node,Time}
    # TODO be smarter. For now this is exhaustive at least, but exponentially slower than needed.
    # Unclear what the exact assumptions on model constraints should be
    # Easy improvement - select immediately identifiable vars with preference (i.e. FixedModel)

    untimed_vars = [var for var in parents.keys() if ~(var in keys(timed_varplaces))]
    if length(untimed_vars)==0
        return timed_varplaces
    end

    for untimed_var in untimed_vars
        new_time = assign_var(parents, timed_varplaces, untimed_var)
        if new_time
            sub_timed_varplaces = copy(timed_varplaces)
            sub_timed_varplaces[untimed_var] = new_time
            recursed = assign_times(parents, sub_timed_varplaces)
        else # Can't assign
            return false
        end

        if recursed
            return recursed
        end # Else try assigning another variable
    end
    
    return false
end

