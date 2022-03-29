export 
    BP,
    infer

"""
    abstract type BP <: InstantAlgorithm

Belief Propagation algorithm
"""
abstract type BP <: InstantAlgorithm end

function answer(::Marginal, ::BP, runtime::Runtime, instance::VariableInstance, bounds = false)
    if bounds
        error("BP cannot provide bounded answers")
    end
    return get_belief(runtime, instance)
end

function infer(algorithm::BP, runtime::InstantRuntime,
    evidence::Dict{Symbol, Score} = Dict{Symbol, Score}(), 
    interventions::Dict{Symbol, Dist} = Dict{Symbol, Dist}(),
    placeholder_beliefs = get_placeholder_beliefs(runtime))
    net = get_network(runtime)
    ensure_all!(runtime)
    order = topsort(get_initial_graph(net))
    set_ranges!(runtime, evidence, algorithm.default_range_size, 1, order, placeholder_beliefs)
    for ph in get_placeholders(net)
        if !(ph.name in keys(placeholder_beliefs))
            error("Placeholder ", ph.name, " does not have a belief")
        end
        pi = placeholder_beliefs[ph.name]
        for ch in get_children(net, ph)
            set_message!(runtime, ph, ch, :pi_message, pi)
        end
    end
    for (n,e) in evidence
        v = get_node(net, n)
        inst = current_instance(runtime, v)
        post_evidence!(runtime, inst, e)
    end
    for (n,i) in interventions
        v = get_node(net, n)
        inst = current_instance(runtime, v)
        post_intervention!(runtime, inst, i)
    end
    run_bp(algorithm, runtime)
end

# Run a backward step of BP for a single variable.
function _backstep(runtime, var, ranges)
    network = get_network(runtime)
    inst = current_instance(runtime, var)
    evidence = get_evidence(runtime, inst)
    sf = get_sfunc(inst)
    chs = get_children(get_network(runtime), var)
    msgs = collect_messages(runtime, chs, var, :lambda_message)
    O = output_type(sf)
    range::Vector{<:O} = [convert(output_type(sf), r) for r in ranges[var.name]]       

    if isempty(msgs)
        ilams = Score{output_type(sf)}[]
        if !isnothing(evidence)
            push!(ilams, evidence)
        end
        lam = compute_lambda(sf, range, ilams)
    else
        incoming_lams = Score[]
        for m in msgs
            m1::Score = m
            push!(incoming_lams, m1)
        end
        if !isnothing(evidence)
            push!(incoming_lams, evidence)
        end
        lam = compute_lambda(sf, range, incoming_lams)
    end

    set_value!(runtime, inst, :lambda, lam)
    pars = get_parents(network, var)
    incoming_pis::Vector{Dist} = collect_messages(runtime, pars, var, :pi_message)
    parranges = [ranges[p.name] for p in pars]
    outgoing_lams = outgoing_lambdas(sf, lam, range, tuple(parranges...), tuple(incoming_pis...))
    distribute_messages!(runtime, var, pars, :lambda_message, outgoing_lams)
end

# Run a forward step of BP for a single variable
# The procedure is slightly different for the first pass, so use the firstpass flag
function _forwardstep(runtime, var, ranges, firstpass)
    network = get_network(runtime)
    inst = current_instance(runtime, var)
    sf = get_sfunc(inst)
    pars = get_parents(network, var)
    O = output_type(sf)
    range = convert(Vector{O}, ranges[var.name])
    parranges = [ranges[p.name] for p in pars]
    incoming_pis = tuple(collect_messages(runtime, pars, var, :pi_message)...)
    if has_intervention(runtime, inst)
        intervention = get_intervention(runtime, inst)
        pi = intervention
    elseif firstpass && has_evidence(runtime, inst) 
        # In the first pass, evidence is treated like lambda messages from elsewhere
        evidence = get_evidence(runtime, inst)
        pi = Cat(range, [get_score(evidence, x) for x in range])
    else
        pi = compute_pi(sf, range, tuple(parranges...), incoming_pis) 
    end
    set_value!(runtime, inst, :pi, pi)

    chs = get_children(network, var)
    if firstpass
        outpis = fill(pi, length(chs))
    else
        lam = get_value(runtime, inst, :lambda)
        bel = compute_bel(sf, range, pi, lam)
        post_belief!(runtime, inst, bel)
        incoming_lams::Vector{Score} = collect_messages(runtime, chs, var, :lambda_message)
        outpis = outgoing_pis(sf, range, bel, incoming_lams)
    end
    distribute_messages!(runtime, var, chs, :pi_message, outpis)
end