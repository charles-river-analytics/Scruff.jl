export
    LoopyBP,
    loopy_BP

"""
    LoopyBP

An instant algorithm that runs loopy belief propagation.

# Arguments
- default_range_size: The size to use as default when calling `support` on a node.
- epsilon: The allowable difference between beliefs on successive iterations 
for termination.
- maxiterations: The maximum number of iterations to run. `infer` will terminate if 
this number of iterations is reached, even if it has not converged.
"""
struct LoopyBP <: BP
    default_range_size::Int
    epsilon::Float64
    maxiterations::Int
    LoopyBP(drs = 10, epsilon = 0.0001, maxiterations = 10) = new(drs, epsilon, maxiterations)
end

function loopy_BP(runtime::Runtime; default_range_size = 10, epsilon = 0.0001, maxiterations = 10)
    network = get_network(runtime)
    for node in topsort(get_initial_graph(network))
        remove_messages!(runtime, node, :pi_message)
        remove_messages!(runtime, node, :lambda_message)
    end
    run_bp(LoopyBP(default_range_size, epsilon, maxiterations), runtime)
end

function run_bp(algorithm::LoopyBP, runtime)
    network = get_network(runtime)
    ranges = Dict()
    nodes = get_nodes(network)

    for node in nodes
        inst = current_instance(runtime, node)
        rng = get_range(runtime, inst)
        ranges[node.name] = rng
    end

    for node in nodes
        for par in get_parents(network, node)
            if par isa Variable
                set_message!(runtime, par, node, :pi_message, 
                    ones(length(ranges[par.name])))
            end
        end
    end
    
    newbeliefs = initial_pass(runtime, network, ranges)
    conv = false
    iteration = 0

    while !conv && iteration < algorithm.maxiterations
        oldbeliefs = copy(newbeliefs)
        backward_pass(runtime, network, ranges)
        forward_pass(runtime, network, ranges, newbeliefs)
        conv = converged_loopy(get_variables(network), ranges, newbeliefs, oldbeliefs, algorithm.epsilon)
        iteration += 1
    end
end

function initialize(runtime, network, ranges)
end

function initial_pass(runtime, network, ranges)
    newbeliefs = Dict{Variable, Dist}()
    variables = [v for v in topsort(get_initial_graph(network)) if v isa Variable]
    for var in variables
        inst = current_instance(runtime, var)
        sf = get_sfunc(inst)
        pars = get_parents(network, var)
        incoming_pis :: Vector{Dist} =
            collect_messages(runtime, pars, var, :pi_message)
        range = ranges[var.name]
        parranges = [ranges[p.name] for p in pars]
        pi = compute_pi(sf, range, tuple(parranges...), tuple(incoming_pis...)) 
        # on the first pass, we interpret evidence as lambda message
        # coming from elsewhere, so include it in the pi
        if has_evidence(runtime, inst)
            evidence = get_evidence(runtime, inst)
            pi = Cat(range, [get_score(evidence, x) for x in range])
        end
        if has_intervention(runtime, inst)
            intervention = get_intervention(runtime, inst)
            pi = intervention
        end
        set_value!(runtime, inst, :pi, pi)
        newbeliefs[var] = pi
        for ch in get_children(network, var)
            set_message!(runtime, var, ch, :pi_message, pi)
        end
    end
    return newbeliefs
end

function backward_pass(runtime, network, ranges)
    variables = [v for v in topsort(get_initial_graph(network)) if v isa Variable]
    for var in reverse(variables)
        _backstep(runtime, var, ranges)
    end
end

function forward_pass(runtime, network, ranges, newbeliefs)
    variables = [v for v in topsort(get_initial_graph(network)) if v isa Variable]
    for var in variables
        _forwardstep(runtime, var, ranges, false)
        inst = current_instance(runtime, var)
        newbeliefs[var] = get_belief(runtime, inst)
    end
end

function converged_loopy(variables, ranges, new_beliefs, old_beliefs, epsilon::Float64)
    total_diff = 0.0
    total_len = 0
    for var in variables
        range = ranges[var.name]
        diffs = [abs(cpdf(new_beliefs[var], (), x) - cpdf(old_beliefs[var], (), x)) for x in range]
        total_diff += sum(diffs)
        total_len += length(range)
    end
    return total_diff / total_len < epsilon
end
