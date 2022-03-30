
module NoveltyExample

using Scruff
using Scruff.Models
using Scruff.SFuncs
using Scruff.Algorithms

struct NoveltySetup
    known_sfs::Vector{Dist{Float64}}
    known_probs::Vector{Float64}
    novelty_prob::Float64
    novelty_prior_mean::Float64
    novelty_prior_sd::Float64
    novelty_sd::Float64
end

function novelty_network(setup::NoveltySetup, numobs::Int)::InstantNetwork
    known_sf = Cat(setup.known_sfs, setup.known_probs)
    known_model = SimpleModel(known_sf)
    known = known_model(:known)
    # known = Cat(setup.known_sfs, setup.known_probs)()(:known)
    is_novel = Flip(setup.novelty_prob)()(:is_novel)
    novelty_mean = Normal(setup.novelty_prior_mean, setup.novelty_prior_sd)()(:novelty_mean)
    novelty = Det(Tuple{Float64}, Dist{Float64}, m -> Normal(m[1], setup.novelty_sd))()(:novelty)
    behavior = If{Dist{Float64}}()()(:behavior)

    variables = [known, is_novel, novelty_mean, novelty, behavior]
    graph = VariableGraph(novelty => [novelty_mean], behavior => [is_novel, novelty, known])
    for i in 1:numobs
        obs = Generate{Float64}()()(obsname(i))
        push!(variables, obs)
        graph[obs] = [behavior]
    end

    return InstantNetwork(variables, graph)
end

obsname(i) = Symbol("obs", i)

function do_experiment(setup::NoveltySetup, obs::Vector{Float64}, alg::InstantAlgorithm)
    net = novelty_network(setup, length(obs))
    evidence = Dict{Symbol, Score}()
    for (i,x) in enumerate(obs)
        evidence[obsname(i)] = HardScore(x)
    end
    runtime = Runtime(net)
    infer(alg, runtime, evidence)

    is_novel = get_node(net, :is_novel)
    novelty_mean = get_node(net, :novelty_mean)
    println("Probability of novel = ", probability(alg, runtime, is_novel, true))
    println("Posterior mean of novel behavior = ", mean(alg, runtime, novelty_mean))
end

function setup(generation_sd::Float64, prob_novel::Float64)::NoveltySetup
    known = [Normal(0.0, generation_sd), Normal(generation_sd, generation_sd)]
    return NoveltySetup(known, [0.75, 0.25], prob_novel, 0.0, 10.0, generation_sd)
end
setup1 = setup(1.0, 0.1)
setup2 = setup(4.0, 0.1)
obs = [5.0, 6.0, 7.0, 8.0, 9.0]

println("Importance sampling")
println("Narrow generation standard deviation")
do_experiment(setup1, obs, LW(1000))
println("Broad generation standard deviation")
do_experiment(setup2, obs, LW(1000))

println("\nBelief propagation")
println("Narrow generation standard deviation")
do_experiment(setup1, obs, ThreePassBP())
println("Broad generation standard deviation")
do_experiment(setup2, obs, ThreePassBP())

println("\nBelief propagation with larger ranges")
println("Narrow generation standard deviation")
do_experiment(setup1, obs, ThreePassBP(25))
println("Broad generation standard deviation")
do_experiment(setup2, obs, ThreePassBP(25))

end
