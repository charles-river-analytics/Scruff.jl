export 
    IterativeSampler

"""
    struct IterativeSampler <: IterativeAlgorithm

An iterative algorithm that uses a sampler to accumulate more samples on each refinement.
"""
struct IterativeSampler <: IterativeAlgorithm
    base_algorithm :: InstantAlgorithm 
end

function prepare(alg::IterativeSampler, runtime::InstantRuntime,
    evidence::Dict{Symbol, <:Score} = Dict{Symbol, Score}(), 
    interventions::Dict{Symbol, <:Dist} = Dict{Symbol, Dist}(),
    placeholder_beliefs = get_placeholder_beliefs(runtime))
    net = get_network(runtime)
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
    for (n,b) in placeholder_beliefs
        p = get_node(net, n)
        inst = current_instance(runtime, p)
        post_belief!(runtime, inst, b)
    end
    set_state!(runtime, :particles, Particles(Sample[], Float64[]))
end
   
function refine(alg::IterativeSampler, runtime::InstantRuntime)
    current_particles = get_state(runtime, :particles)
    infer(alg.base_algorithm, runtime)
    new_particles = get_state(runtime, :particles)
    all_samples = copy(current_particles.samples)
    append!(all_samples, new_particles.samples)
    all_log_weights = copy(current_particles.log_weights)
    append!(all_log_weights, new_particles.log_weights)
    set_state!(runtime, :particles, Particles(all_samples, all_log_weights))
end

answer(q::Query, alg::IterativeSampler, run::Runtime, inst::VariableInstance) = answer(q, alg.base_algorithm, run, inst)

answer(q::Query, alg::IterativeSampler, run::Runtime, insts::Vector{VariableInstance}) = answer(q, alg.base_algorithm, run, insts)

