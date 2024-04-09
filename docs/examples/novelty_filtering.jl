module NoveltyFiltering

using Scruff
using Scruff.Models
using Scruff.SFuncs
using Scruff.Algorithms
import Scruff: make_initial, make_transition

struct NoveltySetup
    known_velocities::Vector{Float64}
    known_probs::Vector{Float64}
    novelty_prob::Float64
    novelty_prior_mean::Float64
    novelty_prior_sd::Float64
    transition_sd::Float64
    observation_sd::Float64
end

struct PositionModel <: VariableTimeModel{Tuple{}, Tuple{Float64, Float64}, Float64} 
    setup::NoveltySetup
end
function make_initial(::PositionModel, ::Float64)::Dist{Float64}
    return Constant(0.0)
end
function make_transition(posmod::PositionModel, parenttimes::Tuple{Float64, Float64}, time::Float64)::SFunc{Tuple{Float64, Float64}, Float64}
    function f(pair)  
        (prevval, velocity) = pair
        Normal(prevval + t * velocity, t * posmod.setup.transition_sd)
    end
    t = time - parenttimes[1]
    return Chain(Tuple{Float64, Float64}, Float64, f)
end

function novelty_network(setup::NoveltySetup, numobs::Int)::DynamicNetwork
    known_velocity = StaticModel(Cat(setup.known_velocities, setup.known_probs))(:known_velocity)
    is_novel = StaticModel(Flip(setup.novelty_prob))(:is_novel)
    novel_velocity = StaticModel(Normal(setup.novelty_prior_mean, setup.novelty_prior_sd))(:novel_velocity)
    velocity = StaticModel(If{Float64}())(:velocity)
        
    position = PositionModel(setup)(:position)
    observation = SimpleModel(LinearGaussian((1.0,), 0.0, setup.observation_sd))(:observation)

    variables = [known_velocity, is_novel, novel_velocity, velocity, position, observation]
    initial_graph = VariableGraph(velocity => [is_novel, novel_velocity, known_velocity], observation => [position])
    transition_graph = VariableGraph(known_velocity => [known_velocity], is_novel => [is_novel], novel_velocity => [novel_velocity], 
                                     velocity => [velocity], position => [position, velocity], observation => [position])
    
    return DynamicNetwork(variables, initial_graph, transition_graph)
end

obsname(i) = Symbol("obs", i)

function do_experiment(setup::NoveltySetup, obs::Vector{Tuple{Float64, Float64}}, alg::Filter)
    net = novelty_network(setup, length(obs))
    runtime = Runtime(net, 0.0) # Set the time type to Float64 and initial time to 0
    init_filter(alg, runtime)

    is_novel = get_node(net, :is_novel)
    velocity = get_node(net, :velocity)
    observation = get_node(net, :observation)

    for (time, x) in obs
        evidence = Dict{Symbol, Score}(:observation => HardScore(x))
        println("Observing ", x, " at time ", time)
        # At a minimum, we need to include query and evidence variables in the filter step
        filter_step(alg, runtime, Variable[is_novel, velocity, observation], time, evidence)

        println("Probability of novel = ", probability(alg, runtime, is_novel, true))
        println("Posterior mean of velocity = ", mean(alg, runtime, velocity))
    end
end

# Known velocities are 0 and 1, novelty has mean 0 and standard deviation 10
setup = NoveltySetup([0.0, 1.0], [0.7, 0.3], 0.1, 0.0, 10.0, 1.0, 1.0)
obs1 = [(1.0, 2.1), (3.0, 5.8), (3.5, 7.5)] # consistent with velocity 2
obs2 = [(1.0, 4.9), (3.0, 17.8), (3.5, 20.5)] # consistent with velocity 6

println("Particle filter")
println("Smaller velocity")
# CoherentPF is a kind of asynchronous filter that makes sure all relevant variables are included in the filter step.
# In this example, it will ensure that position is included.
do_experiment(setup, obs1, CoherentPF(1000))
println("\nLarger velocity")
do_experiment(setup, obs2, CoherentPF(1000))

# The BP filter is included to show it's use, but it doesn't work well here.
# It only passes marginal distributions from one time step to the next,
# but with static variables, joint distributions are needed.
println("\nBP filter")
println("Smaller velocity")
do_experiment(setup, obs1, CoherentBP())
println("\nLarger velocity")
do_experiment(setup, obs2, CoherentBP())

end
