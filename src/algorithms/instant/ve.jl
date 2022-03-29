export
    VE,
    ve

import Scruff.Operators.make_factors
import Scruff.SFuncs: Invertible, Serial
using Folds

"""
    VE(query_vars::Vector{Variable})

An instant algorithm that runs variable elimination.

# Arguments
- network
- `query_vars`: The variables to query, which are not eliminated
- depth: A depth of 1 means not to expand expanders, otherwise expands recursively to the given depth
- bounds: If true, return lower and upper bounds factors, otherwise just return a single factor
"""
struct VE <: InstantAlgorithm 
    query_vars::Vector{<:Variable}
    depth::Int
    bounds::Bool
    default_range_size::Int
    VE(q; depth = 1, bounds = false, range_size = 1000) = new(q,depth,bounds,range_size)
end

function answer(::Marginal, alg::VE, runtime::Runtime, inst::VariableInstance{O}) where O 
    inv = Invertible{Tuple{O},O}(x -> x[1], x -> (x,))
    jnt = joint(alg, runtime, [inst])
    if !alg.bounds
        s = Serial(Tuple{}, O, (jnt, inv))
        support(s, (), 10000, O[]) # need to precompute
        return s
    else 
        (lower, upper) = jnt
        sl = Serial(Tuple{}, O, (lower, inv))
        su = Serial(Tuple{}, O, (upper, inv))
        support(sl, (), 10000, O[]) # need to precompute
        support(su, (), 10000, O[]) # need to precompute
        return (sl, su)
    end
end

function answer(query::ProbValue{O}, alg::VE, runtime::Runtime, inst::VariableInstance{O}) where O
    x = query.value
    if !alg.bounds
        m = answer(Marginal(), alg, runtime, inst)
        return cpdf(m, (), x)
    else
        (lm,um) = answer(Marginal(), alg, runtime, inst)
        l = cpdf(lm, (), x)
        u = cpdf(um, (), x)
        if isapprox(l,u)
            return(l)
        else
            error("Lower and upper bounds are not equal - use ProbabilityBounds")
        end
    end
end

function answer(query::ProbabilityBounds{O}, alg::VE, runtime::Runtime, inst::VariableInstance{O}) where O
    (lm,um) = answer(Marginal(), alg, runtime, inst)
    r = query.range
    n = length(r)
    ls = zeros(Float64, n)
    us = zeros(Float64, n)
    # Typically, when a value is not in the support of an sfunc, we interpret its probability as zero
    # However, for upper bounds, if it is in range, we want the upper bound to be 1
    usup = support(um, (), 1000, O[])
    for i in 1:n
        x = r[i]
        ls[i] = cpdf(lm, (), x)
        us[i] = x in usup ? cpdf(um, (), x) : 1.0
    end
    lsum = sum(ls)
    usum = sum(us)
    resultls = zeros(Float64, n)
    resultus = zeros(Float64, n)
    for i in 1:n
        # The probability of a value cannot be less than 1 - the upper bounds of the other values
        # or more than 1 - the lower bounds of the other values
        resultls[i] = max(ls[i], 1 - (usum - us[i]))
        resultus[i] = min(us[i], 1 - (lsum - ls[i]))
    end
    return (resultls, resultus)
end

function joint(alg::VE, runtime::Runtime, insts::Vector{<:Instance})
    vars::Vector{Variable} = [i.node for i in insts]
    if any(v -> !(v in alg.query_vars), vars)
        error("Cannot query eliminated variables")
    end
    (jnt, ids) = get_state(runtime, :joint_belief)
    if !alg.bounds
        # no bounds, just a single result
        return marginalize(runtime, normalize(jnt), ids, alg.query_vars, vars, alg.depth)
    else 
        # jnt has lower and upper bounds to probabilities
        lower = marginalize(runtime, normalize(jnt[1]), ids, alg.query_vars, vars, alg.depth)
        upper = marginalize(runtime, normalize(jnt[2]), ids, alg.query_vars, vars, alg.depth)
        return (lower, upper)
    end
end

function marginalize(runtime, factor::Factor, keys::Dict{<:Node,Int}, query_vars::Vector{<:Variable}, vars_to_remain::Vector{<:Variable}, depth::Int)
    function get_var(k)
        for v in query_vars
            if keys[v] == k
                return v
            end
        end
    end
    index = [get_var(k) for k in factor.keys]
    ranges = [get_range(runtime, current_instance(runtime, v), depth) for v in index]
    selector = [v in vars_to_remain for v in index]
    combos = cartesian_product(ranges)
    result = Dict{Tuple, Float64}()
    for (combo, entry) in zip(combos, factor.entries)
        vals = []
        for i in 1:length(index)
            if selector[i]
                push!(vals, combo[i])
            end
        end
        tup = tuple(vals...)
        result[tup] = get(result, tup, 0.0) + entry
    end
    return Cat(result)
end

function infer(alg::VE, runtime::InstantRuntime,
    evidence::Dict{Symbol, Score} = Dict{Symbol, Score}(), 
    interventions::Dict{Symbol, Dist} = Dict{Symbol, Dist}(),
    placeholder_beliefs = get_placeholder_beliefs(runtime))
    network = get_network(runtime)
    if !(isempty(interventions))
        error("VE cannot handle interventions")
    end
    order = topsort(get_initial_graph(network))
    ensure_all!(runtime)
    for pname in keys(placeholder_beliefs)
        ph = get_node(network, pname)
        inst = current_instance(runtime, ph)
        pi = placeholder_beliefs[pname]
        set_range!(runtime, inst, support(pi, NTuple{0,Vector}(), 
            alg.default_range_size, output_type(ph)[]), alg.depth)
    end
    set_ranges!(runtime, evidence, alg.default_range_size, 1, order)
    for (n,e) in evidence
        node = get_node(network, n)
        inst = current_instance(runtime, node)
        post_evidence!(runtime, inst, e)
    end
    jnt = ve(runtime, order, alg.query_vars; depth = alg.depth, placeholder_beliefs = placeholder_beliefs, bounds = alg.bounds)
    set_state!(runtime, :joint_belief, jnt)
end

#############################################
#                                           #
# Helper functions for variable elimination #
#                                           #
#############################################

# Create a map from variable names to factor keys
function factorkeys(order) :: Dict{Node, Int}
    result = Dict{Node, Int}()
    for var in order
        result[var] = nextkey()
    end
    return result
end

# Create lower and upper bound factors for the given variable.
# Uses predetermined ranges for the variable and its parents.
#
# TODO: Change this to use an operator that returns a set of factors rather than probabilities.
# This will enable local factor decompositions for sfuncs.
function lower_and_upper_factors(runtime::Runtime, fn::Function,
                                 ids::Dict{Node, Int}, var::Variable{I,J,O}, depth) where {I,J,O}
    range = get_range(runtime, var, depth)
    pars = get_parents(get_network(runtime), var) 
    parranges = []
    for v in pars
        push!(parranges, get_range(runtime, v, depth))
    end
    prtype = isempty(parranges) ? Vector{O} : typejoin([typeof(x) for x in parranges]...)
    parranges = convert(Vector{prtype}, parranges)

    inst = current_instance(runtime, var)
    sf = get_sfunc(inst)
    # @debug "lower_and_upper_factors" var=var.name sf=typeof(sf)
    parids = tuple([ids[p] for p in pars]...)
    # Since Expander needs a runtime to create factors, it has special purpose code
    # if is_fixed(var.model) && isa(make_sfunc(var, runtime), Expander)
    if isa(make_initial(var.model), Expander)
        (lowers, uppers) = expander_probs(runtime, fn, var, depth)
        keys = map(s -> ids[s], pars)
        dims = isempty(pars) ? Int[] : map(length, parranges)
        push!(keys, ids[var])
        push!(dims, length(range))
        kds = Tuple(dims)
        kts = Tuple(keys)
        return ([Factor(kds, kts, lowers)], [Factor(kds, kts, uppers)])
    else
        facts = make_factors(sf, range, Tuple(parranges), ids[var], parids)
        return facts
    end
end

function evidence_factor(runtime, var::Variable, ids::Dict)
    inst = current_instance(runtime, var)
    evidence = get_evidence(runtime, inst)
    range = get_range(runtime, inst)
    dims = Tuple(length(range))
    keys = Tuple(ids[var])
    entries = Array{Float64, 1}(undef, length(range))
    for i = 1:length(range)
        entries[i] = get_score(evidence, range[i])
    end
    return Factor(dims, keys, entries)
end

# Make all the initial factors for the given network
function produce_factors(runtime::Runtime, fn::Function,
                      order::Vector{Node}, ids::Dict, placeholder_beliefs, depth)
    lowers = []
    uppers = []
    for node in order
        if node isa Variable
            (lower, upper) =
                lower_and_upper_factors(runtime, fn, ids, node, depth)
            append!(lowers, lower)
            append!(uppers, upper)
            inst = current_instance(runtime, node)
            if has_evidence(runtime, inst)
                evfact = evidence_factor(runtime, node, ids)
                push!(lowers, evfact)
                push!(uppers, evfact)
            end
        else
            bel = placeholder_beliefs[node.name]
            range = get_range(runtime, node, depth)
            (phlower, phupper) = make_factors(bel, range, (), ids[node], ())
            append!(lowers, phlower)
            append!(uppers, phupper)
        end
    end
    return (lowers, uppers)
end

function make_graph(factors)
    result = Graph()
    for fact in factors
        ids = fact.keys
        for (i,id) in enumerate(ids)
            size = fact.dims[i]
            add_node!(result, id, size)
            for other in ids
                if other != id
                    add_undirected!(result, id, other)
                end
            end
        end
    end
    return result
end

# Eliminate the variable with the given id by multiplying all factors
# mentioning the variable and summing the variable out of the result
function eliminate(var_id, factors)
    relevant = filter(f -> var_id in f.keys, factors)
    remaining = filter(f -> !(var_id in f.keys), factors)
    if !isempty(relevant)
        prodfact = relevant[1]
        for i in 2:length(relevant)
            prodfact = product(prodfact, relevant[i])
        end
        sumfact = sum_over(prodfact, var_id)
        # If the node being eliminated is completely isolated from the rest of the network,
        # sumfact will be empty and shouldn't be added
        if !isnothing(sumfact.dims)
            push!(remaining, sumfact)
        end
    end
    return remaining
end


########################################################
#                                                      #
# Run the variable elimination algorithm               #
#                                                      #
# Works with both discrete and continuos variables,    #
# using previously determined ranges.                  #
#                                                      #
# The second argument is a topologically ordered       #
# list of variables to include in the computation.     #
# The code assumes that for any variable, its parents  #
# are present and precede it in the order.             #
# The third argument is a list of variables to query   #
# i.e. not to eliminate. This must be nonempty.        #
#                                                      #
# Returns lower and upper bound factors,               #
# as well as a key to variable names from factor keys. #
#                                                      #
########################################################

function ve(runtime::Runtime, order::Vector{<:Node},
            query_vars::Vector{<:Variable}; depth = 1, placeholder_beliefs = Dict{Symbol,Dist}(), bounds = false)
    @assert !isempty(query_vars)
    ids :: Dict{Node, Int} = factorkeys(order)
    # Making values for an expander takes a function to solve subnetworks.
    # Here we create a function that passes the bounds flag for ve.
    f(runtime, order, query_vars, depth) = ve(runtime, order, query_vars; depth = depth, placeholder_beliefs = placeholder_beliefs, bounds = bounds)
    (lowers, uppers) = produce_factors(runtime, f, order, ids, placeholder_beliefs, depth)
    ve_graph = make_graph(lowers) # Assumes that lower and upper factors have same structure
    elim_order = greedy_order(ve_graph, map(v -> ids[v], query_vars))
    for var_id in elim_order
        lowers = eliminate(var_id, lowers)
        if bounds
            uppers = eliminate(var_id, uppers)
        end
    end
    lowerprod = lowers[1]
    for i in 2:length(lowers)
        lowerprod = product(lowerprod, lowers[i])
    end
    if bounds
        # if (length(lowers[1].entries) > 45 && length(uppers[1].entries) > 45)
        #     @info("$(typeof(lowers[1])) && $(typeof(uppers[1]))",
        #         lowers=lowers[1].entries[40:1:45], 
        #         uppers=uppers[1].entries[40:1:45])
        # end
        upperprod = uppers[1]
        for i in 2:length(uppers)
            upperprod = product(upperprod, uppers[i])
        end
        return ((lowerprod, upperprod), ids)
    else
        return (lowerprod, ids)
    end
end

function copy_graph(g :: Graph)
    ns = copy(g.nodes)
    es = Dict()
    ss = Dict()
    for n in ns
        es[n] = copy(g.edges[n])
        ss[n] = g.sizes[n]
    end
    return Graph(ns, es, ss)
end

##########################
#                        #
# Eliminating a variable #
#                        #
##########################

function unconnected_neighbors(g :: Graph, n :: Int)
    neighbors = g.edges[n]
    m = length(neighbors)
    result = []
    for i = 1:m
        n1 = neighbors[i]
        for j = i+1:m
            n2 = neighbors[j]
            if !(n1 in g.edges[n2]) || !(n2 in g.edges[n1])
                push!(result, (n1, n2))
            end
        end
    end
    return result
end

function eliminate(g :: Graph, n :: Int)
    ns = unconnected_neighbors(g, n)
    deleteat!(g.nodes, findfirst(m -> m == n, g.nodes))
    delete!(g.edges, n)
    delete!(g.sizes, n)
    for (m,ms) in g.edges
        if n in ms
            deleteat!(ms, findfirst(m -> m == n, ms))
        end
    end
    for (i, j) in ns
        add_undirected!(g, i, j)
    end
end

#############################################
#                                           #
# Greedily determining an elimination order #
# to minimize the number of edges added     #
#                                           #
#############################################

function cost(g :: Graph, n :: Int)
    return length(unconnected_neighbors(g, n))
end

function greedy_order(g :: Graph, to_leave :: Array{Int})
    candidates = filter(n -> !(n in to_leave), g.nodes)
    result = []
    h = copy_graph(g)
    while !isempty(candidates)
        costs = map(c -> cost(h, c), candidates)
        best = candidates[argmin(costs)]
        push!(result, best)
        eliminate(h, best)
        deleteat!(candidates, findfirst(n -> n == best, candidates))
    end
    return result
end

function greedy_order(g :: Graph)
    a :: Array{Int, 1} = []
    return greedy_order(g, a)
end
