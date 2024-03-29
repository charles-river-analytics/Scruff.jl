export
    Importance,
    Rejection,
    LW,
    rejection_proposal,
    lw_proposal,
    make_custom_proposal

"""
    Importance <: InstantAlgorithm
    
Representation of an importance sampling algorithm.

# arguments
- proposal_function Specifies how the algorithm should make proposals. This is a function
that takes a runtime and an instance and returns a proposer. 
The proposer takes parent values and proposes a value for the instance along with a log
score.
- num_particles The number of completed particles to use. This is not necessarily the
number attempted. If there are rejections, the algorithm will continue to create particles
until `num_particles` have been completed. Warning: With impossible evidence, the process
will not terminate.
""" 
struct Importance <: InstantAlgorithm
    proposal_function :: Function   
    num_particles :: Int
end

function answer(pf::ProbFunction, ::Importance, runtime::Runtime, instances::Vector{VariableInstance}) 
    particles = get_state(runtime, :particles)
    score = 0.0
    total = 0.0
    for (s, lw) in zip(particles.samples, particles.log_weights)
        x = [s[get_name(inst)] for inst in instances]
        w = exp(lw)
        if pf.fn(x)
            score += w
        end
        total += w
    end
    return score / total
end

function answer(::Marginal, ::Importance, runtime::Runtime, instance::VariableInstance)
    O = output_type(get_node(instance))
    dict = Dict{O, Float64}()
    particles = get_state(runtime, :particles)
    for (s, lw) in zip(particles.samples, particles.log_weights)
        x = s[get_name(instance)]
        w = exp(lw)
        dict[x] = get(dict, x, 0.0) + w
    end
    z = sum(values(dict))
    vs = O[]
    ps = Float64[]
    for (v, p) in dict
        push!(vs, v)
        push!(ps, p/z)
    end
    return Cat(vs, ps)
end

const Reject = ErrorException("Reject")

"""
    rejection_proposal(::Runtime, instance::VariableInstance, parent_values::Tuple)

Return a proposer and scorer to implement standard rejection sampling from the prior.
It proposes a value for the `instance` from its sfunc, and scores it by the evidence,
if any. If the score is -Infinity, it throws a Reject exception.
"""
function rejection_proposal(runtime::Runtime, instance::VariableInstance)
    proposer(parent_values) = (sample(get_sfunc(instance), parent_values), 0.0)
end
    
function _get_hard_evidence(runtime, instance)::Union{HardScore, Nothing}
    if has_evidence(runtime, instance)
        ev = get_evidence(runtime, instance)
        if ev isa HardScore
            return ev
        end
    end
    return nothing
end

"""
    lw_proposal(runtime::Runtime, instance::VariableInstance, parent_values::Tuple)

Return a proposer and scorer to implement likelihood weighting.

This proposal scheme is the same as the prior proposal unless a variable has hard evidence.
In the case of hard evidence, the proposer sets the value of the variable to the evidence
value and scores it by the log conditional probability of the evidence given the parent
values.
"""
function lw_proposal(runtime::Runtime, instance::VariableInstance)
    evidence = _get_hard_evidence(runtime, instance)
    if !isnothing(evidence)
        return _hard_evidence_proposal(evidence, instance)
    else
        return rejection_proposal(runtime, instance)
    end
end

function _hard_evidence_proposal(evidence, instance)
    sf = get_sfunc(instance)
    x = evidence.value
    proposer(parent_values) = (x, logcpdf(sf, parent_values, x))
    return proposer
end

"""
    make_custom_proposal(custom_sfs::Dict{Symbol, SFunc})

Create a proposal function for a custom proposal scheme.

Returns a proposal function that can be provided to the Importance constructor.
Evidence is handled similar to `lw`, except that the custom proposal is used for soft
evidence.

# Arguments
- custom_sfs A dictionary mapping variable names to custom sfuncs used for their proposal.
Need not be complete; if a variable is not in this dictionary, its standard sfunc will be
used.
"""
function make_custom_proposal(custom_sfs::Dict{Symbol, SFunc})
    function proposal(runtime, instance)
        evidence = _get_hard_evidence(runtime, instance)
        if !isnothing(evidence)
            return _hard_evidence_proposal(evidence, instance)
        else    
            name = get_name(instance)
            if name in keys(custom_sfs)
                prior_sf = get_sfunc(instance)
                proposal_sf = custom_sfs[name]
                function proposer(parent_values) 
                    x = sample(proposal_sf, parent_values)
                    l = logcpdf(prior_sf, parent_values, x) - 
                            logcpdf(proposal_sf, parent_values, x)
                    return (x, l)
                end
                return proposer
            else
                return rejection_proposal(runtime, instance)
            end
        end
    end
    return proposal
end

Rejection(num_particles) = Importance(rejection_proposal, num_particles)

LW(num_particles) = Importance(lw_proposal, num_particles)

function _importance(runtime::Runtime, num_samples::Int, proposal_function::Function,
    samples::Vector{Dict{Symbol, Any}}, lws)
    net = runtime.network
    nodes = topsort(get_initial_graph(net))
    proposers = Function[]
    evidences = Score[]
    interventions = Union{Dist,Nothing}[]
    for v in nodes 
        if v isa Variable
            inst = current_instance(runtime, v)
            push!(proposers, proposal_function(runtime, inst))
            if has_evidence(runtime, inst)
                push!(evidences, get_evidence(runtime, inst))
            else
                push!(evidences, FunctionalScore{output_type(v)}(x -> 1.0))
            end
            if has_intervention(runtime, inst)
                push!(interventions, get_intervention(runtime, inst))
            else
                push!(interventions, nothing)
            end
        end
    end
    s = 1
    while s <= num_samples
        try
            vnum = 1
            for v in nodes 
                if v isa Variable
                    try
                      inst = current_instance(runtime, v)
                      if !isnothing(interventions[vnum])
                          iv = interventions[vnum]
                          samples[s][v.name] = sample(iv, ())
                      else
                          proposer = proposers[vnum]
                          pars = get_initial_parents(net, v)
                          parvals = tuple([samples[s][p.name] for p in pars]...)
                          (x, lw) = proposer(parvals)
                          pe = get_log_score(evidences[vnum], x)
                          if !isfinite(pe)
                              throw(Reject)
                          end
                          samples[s][v.name] = x
                          lws[s] += lw + pe
                      end
                    catch ex
                      @error("Error on variable $v")
                      rethrow(ex)
                    end
                      vnum += 1
                end
            end
            s += 1
        catch e
            if e != Reject
                rethrow(e)
            end
        end
    end
    for v in nodes
        if v isa Variable
            ps = Dict{output_type(v), Float64}()
            for i in 1:length(samples)
                x = samples[i][v.name]
                ps[x] = get(ps, x, 0.0) + exp(lws[i])
            end
            i = current_instance(runtime, v)
            set_value!(runtime, i, :belief, Cat(ps))
        end
    end

    log_prob_evidence = logsumexp(lws) - log(num_samples)
    set_state!(runtime, :log_prob_evidence, log_prob_evidence)
    set_state!(runtime, :particles, Particles(samples, lws))
end

function infer(algorithm::Importance, runtime::InstantRuntime,
    evidence::Dict{Symbol, Score} = Dict{Symbol, Score}(), 
    interventions::Dict{Symbol, Dist} = Dict{Symbol, Dist}(),
    placeholder_beliefs::Dict{Symbol, Dist} = get_placeholder_beliefs(runtime))
    net = get_network(runtime)
    nodes = get_nodes(net)
    ensure_all!(runtime)
    # See comment in instantalgorithm.jl
    # During particle filtering, importance needs joint samples of placeholders, not marginals
    # So we first check to see if particles already exists, and only add placeholder beliefs if
    # they're not found in existing particles
    samples = Dict{Symbol, Any}[]
    lws = Float64[]
    if has_state(runtime, :particles)
        particles = get_state(runtime, :particles)
        # Copy over only values of nodes in the current runtime
        for i in 1:algorithm.num_particles
            if length(particles.samples) > 0
                index_into_particles = (i-1) % length(particles.samples) + 1
                newsample = Dict{Symbol, Any}()
                oldsample = particles.samples[index_into_particles]
                for n in nodes
                    if n.name in keys(oldsample)
                        newsample[n.name] = oldsample[n.name]
                    end
                end
                push!(samples, newsample)
                push!(lws, particles.log_weights[index_into_particles])
            else
                push!(samples, Dict{Symbol, Any}())
                push!(lws, 0.0)
            end
        end
    else
        for i in 1:algorithm.num_particles
            push!(samples, Dict{Symbol, Any}())
        end
        lws = zeros(Float64, algorithm.num_particles)
    end

    placeholders = get_placeholders(net)
    for ph in placeholders
        pi = placeholder_beliefs[ph.name]
        for i in 1:algorithm.num_particles
            if !(ph.name in keys(samples[i]))
                samples[i][ph.name] = sample(pi, ())
            end
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
    _importance(runtime, algorithm.num_particles, algorithm.proposal_function, samples, lws)
end
