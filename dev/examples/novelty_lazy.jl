module NoveltyLazy

using Scruff
using Scruff.SFuncs
using Scruff.Algorithms
import Scruff.Operators: expectation

struct NoveltySetup
    known_sfs::Vector{Dist{Float64}}
    known_probs::Vector{Float64}
    novelty_prob::Float64
    novelty_prior_mean::Float64
    novelty_prior_sd::Float64
    novelty_sd::Float64
end

function novelty_network(setup::NoveltySetup, numobs::Int)::InstantNetwork
    known = Cat(setup.known_sfs, setup.known_probs)()(:known)
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

function do_experiment(setup, obs)
    net = novelty_network(setup, length(obs))
    is_novel = get_node(net, :is_novel)
    novelty_mean = get_node(net, :novelty_mean)
    evidence = Dict{Symbol, Score}()
    for (i,x) in enumerate(obs)
        evidence[obsname(i)] = HardScore(x)
    end

    alg = LSFI([is_novel, novelty_mean]; start_size = 5, increment = 5)
    runtime = Runtime(net)
    prepare(alg, runtime, evidence)

    for i = 1:10
        println("Range size: ", alg.state.next_size)
        refine(alg, runtime)
        is_novel_lb = probability_bounds(alg, runtime, is_novel, [false, true])[1]
        println("Probability of novel = ", is_novel_lb[2])
        println("Posterior mean of novel behavior = ", mean(alg, runtime, novelty_mean))
    end
end

function setup(generation_sd::Float64, prob_novel::Float64)::NoveltySetup
    known = [Normal(0.0, generation_sd), Normal(generation_sd, generation_sd)]
    return NoveltySetup(known, [0.75, 0.25], prob_novel, 0.0, 10.0, generation_sd)
end
setup1 = setup(1.0, 0.1)
setup2 = setup(4.0, 0.1)
obs = [5.0, 6.0, 7.0, 8.0, 9.0]

println("Lazy Inference")
println("Narrow generation standard deviation")
do_experiment(setup1, obs)
println("\nBroad generation standard deviation")
do_experiment(setup2, obs)


end