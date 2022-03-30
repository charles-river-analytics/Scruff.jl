module SoccerExample
using Scruff
using Scruff.SFuncs
using Scruff.Operators
using Scruff.Algorithms
using Scruff.Models
using Scruff.RTUtils
using Scruff.Utils

# define parameters
T = 4 # number of time steps

winning_range = [:us, :them, :none]

confident_range = [:yes, :no]
confident_prior = [0.4, 0.6]

possession_range = [:yes, :no]

goal_range = [:yes, :no]

scoreDiff_range = [:min5, :min4, :min3, :min2, :min1, :zero, :plus1, :plus2, :plus3, :plus4, :plus5]
scoreDiff_prior = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

sizeScore = length(scoreDiff_range)
sizeGoal = length(goal_range)
sizePossession = length(possession_range)

winningCPT = Dict(
(:min5,) => [0.0,  1.0, 0.0],
(:min4,) => [0.0,  1.0, 0.0],
(:min3,) => [0.0,  1.0, 0.0],
(:min2,) => [0.0,  1.0, 0.0],
(:min1,) => [0.0,  1.0, 0.0],
(:zero,) => [0.0,  0.0, 1.0],
(:plus1,) => [1.0,  0.0, 0.0],
(:plus2,) => [1.0,  0.0, 0.0],
(:plus3,) => [1.0,  0.0, 0.0],
(:plus4,) => [1.0,  0.0, 0.0],
(:plus5,) => [1.0,  0.0, 0.0])

confidentCPT = Dict( # confident(t-1) and winning
    (:yes, :us) =>[0.9, 0.1],
    (:no, :us) => [0.5, 0.5],
    (:yes, :them) => [0.5, 0.5],
    (:no, :them) => [0.1, 0.9],
    (:yes, :none) => [0.7, 0.3],
    (:no, :none) => [0.3, 0.7])

goalCPT = Dict( # possession and confident
    (:yes, :yes) => [0.04, 0.96],
    (:no, :yes) => [0.045, 0.955],
    (:yes, :no) => [0.01, 0.99],
    (:no, :no) => [0.02, 0.98])

possessionCPT = Dict(
    (:yes,) => [0.7, 0.3],
    (:no,) => [0.3, 0.7])

scoreDiffCPT = Dict{NTuple{3, Symbol},Array{Float64,1}}()
for i = 1:sizeScore
    v_nochange = zeros(sizeScore, 1)
    v_plus1 = zeros(sizeScore, 1)
    v_minus1 = zeros(sizeScore, 1)
    v_nochange[i] = 1.0
    if (i < sizeScore)
        v_plus1[i+1] = 1.0
    else
        v_plus1[i] = 1.0
    end
    if (i > 1)
        v_minus1[i-1] = 1.0
    else
        v_minus1[i] = 1.0
    end
    scoreDiffCPT[(scoreDiff_range[i], :yes, :yes)] = v_plus1[:, 1]
    scoreDiffCPT[(scoreDiff_range[i], :no, :yes)] = v_nochange[:, 1]
    scoreDiffCPT[(scoreDiff_range[i], :yes, :no)] = v_minus1[:, 1]
    scoreDiffCPT[(scoreDiff_range[i], :no, :no)] = v_nochange[:, 1]
end


function create_network_unrolling(T)
    scoreDiff_tmin1 = Cat(scoreDiff_range, scoreDiff_prior)()(:scoreDiff0)
    confident_tmin1 = Cat(confident_range, confident_prior)()(:confident0)
    vars = Variable[scoreDiff_tmin1, confident_tmin1]
    parents = VariableGraph()
    for i = 1:T
        winning_t = DiscreteCPT(winning_range, winningCPT)()(Symbol(string(:winning) * string(i)))
        confident_t = DiscreteCPT(confident_range, confidentCPT)()(Symbol(string(:confident) * string(i)))
        possession_t = DiscreteCPT(possession_range, possessionCPT)()(Symbol(string(:possession) * string(i)))
        goal_t = DiscreteCPT(goal_range, goalCPT)()(Symbol(string(:goal) * string(i)))
        scoreDiff_t = DiscreteCPT(scoreDiff_range, scoreDiffCPT)()(Symbol(string(:scoreDiff) * string(i)))
        push!(vars, winning_t)
        push!(vars, confident_t)
        push!(vars, possession_t)
        push!(vars, goal_t)
        push!(vars, scoreDiff_t)

        parents[winning_t] = [scoreDiff_tmin1]
        parents[confident_t] = [confident_tmin1, winning_t]
        parents[possession_t] = [confident_t]
        parents[goal_t] = [possession_t, confident_t]
        parents[scoreDiff_t] = [scoreDiff_tmin1, goal_t, possession_t]

        confident_tmin1 = confident_t
        scoreDiff_tmin1 = scoreDiff_t
    end
    network = InstantNetwork(vars, parents)
    return network
end


function create_dynamic_network()
    scoreDiff_init = Cat(scoreDiff_range, scoreDiff_prior)
    scoreDiff_cpt = DiscreteCPT(scoreDiff_range, scoreDiffCPT)
    scoreDiff_t = HomogeneousModel(scoreDiff_init, scoreDiff_cpt)(:scoreDiff)

    confident_init = Cat(confident_range, confident_prior)
    confident_cpt = DiscreteCPT(confident_range, confidentCPT)
    confident_t = HomogeneousModel(confident_init, confident_cpt)(:confident)

    winning_init = Constant(:none)
    #winning_init = DiscreteCPT(winning_range, winningCPT)
    winning_cpt = DiscreteCPT(winning_range, winningCPT)
    winning_t = HomogeneousModel(winning_init, winning_cpt)(:winning)

    possession_init = Constant(:yes)
    #possession_init = DiscreteCPT(possession_range, possessionCPT)
    possession_cpt = DiscreteCPT(possession_range, possessionCPT)
    possession_t = HomogeneousModel(possession_init, possession_cpt)(:possession)

    goal_init = Constant(:no)
    #goal_init = DiscreteCPT(goal_range, goalCPT)
    goal_cpt = DiscreteCPT(goal_range, goalCPT)
    goal_t = HomogeneousModel(goal_init, goal_cpt)(:goal)

    vars = Variable[scoreDiff_t, confident_t, winning_t, possession_t, goal_t]
    parents = VariableGraph()
    parents[winning_t] = [scoreDiff_t]
    parents[confident_t] = [confident_t, winning_t]
    parents[possession_t] = [confident_t]
    parents[goal_t] = [possession_t, confident_t]
    parents[scoreDiff_t] = [scoreDiff_t, goal_t, possession_t]

    network = DynamicNetwork(vars, VariableGraph(), parents)
    return network
end

function run_filtering_inference(network, alg)
    runtime = Runtime(network)
    init_filter(alg, runtime)
    vars = get_variables(network)
    score_var = get_node(network, :scoreDiff)
    idx = findall(x-> x.name==:scoreDiff, vars)
    vars_no_score = copy(vars)
    deleteat!(vars_no_score,idx)
    for i=1:T
        println("i=$i")
        if(i>1)
            filter_step(alg, runtime, Variable[score_var], i-1, Dict{Symbol, Score}())
        end
        if(i==3)
            filter_step(alg, runtime, vars_no_score, i, Dict{Symbol, Score}(:confident => HardScore(:yes), :goal => HardScore(:yes)))
        else
            filter_step(alg, runtime, vars_no_score, i, Dict{Symbol, Score}())
        end

        conf_i = current_instance(runtime, get_node(network,:confident))
        println("Confident at minute $i is : $(probability(alg, runtime, conf_i, :yes))")

        poss_i = current_instance(runtime, get_node(network, :possession))
        println("Possession at minute $i is : $(probability(alg, runtime, poss_i, :yes))")

        goal_i = current_instance(runtime, get_node(network, :goal))
        println("Goal at minute $i is : $(probability(alg, runtime, goal_i, :yes))")

        scoreDiff_i = current_instance(runtime, get_node(network, :scoreDiff))
        score_bels = [probability(alg, runtime, scoreDiff_i, score1) for score1 in scoreDiff_range]
        println("Score is [losing_by_5, losing_by_4, losing_by_3, losing_by_2, losing_by_1, even, winning_by_1, winning_by_2, winning_by_3, winning_by_4, winning_by_5] at minute $i is : $score_bels")

        winning_i = current_instance(runtime, get_node(network, :winning))
        println("Winning at minute $i is : $(probability(alg, runtime, winning_i, :us))")
    end

end
function run_static_inference(network, alg)
    runtime = Runtime(network) # create runtime
    
    # Set evidence.
    evid = HardScore(:yes)
    evidence = Dict{Symbol, Score}(:confident3 => evid, :goal3 => evid)
    
    # Perform inference.
    infer(alg, runtime, evidence)

    # Get updated beliefs 
    for i=1:T
        conf_i = current_instance(runtime, get_node(network, Symbol(string(:confident) * string(i))))
        println("Confident at minute $i is : $(probability(alg, runtime, conf_i, :yes))")

        poss_i = current_instance(runtime, get_node(network, Symbol(string(:possession) * string(i))))
        println("Possession at minute $i is : $(probability(alg, runtime, poss_i, :yes))")

        goal_i = current_instance(runtime, get_node(network, Symbol(string(:goal) * string(i))))
        println("Goal at minute $i is : $(probability(alg, runtime, goal_i, :yes))")

        scoreDiff_i = current_instance(runtime, get_node(network, Symbol(string(:scoreDiff) * string(i))))
        score_bels = [probability(alg, runtime, scoreDiff_i, score1) for score1 in scoreDiff_range]
        println("Score is [losing_by_5, losing_by_4, losing_by_3, losing_by_2, losing_by_1, even, winning_by_1, winning_by_2, winning_by_3, winning_by_4, winning_by_5] at minute $i is : $score_bels")

        winning_i = current_instance(runtime, get_node(network, Symbol(string(:winning) * string(i))))
        println("Winning at minute $i is : $(probability(alg, runtime, winning_i, :us))")
    end
end


## Static network
network = create_network_unrolling(T) #create network
println("##################################")
println("Running Three Pass Belief Propagation")
println("##################################")
run_static_inference(network, ThreePassBP()) # run three pass belief propagation
println("\n##################################")
println("Running Loopy Belief Propagation")
println("##################################")
run_static_inference(network, LoopyBP()) # run loopy belief propagation
# VE too slow for this example
#println("\n##################################")
# println("Running VE")
# println("##################################")
# run_static_inference(network, VE(get_variables(network))) # run VE
println("\n##################################")
println("Running Importance Sampling")
println("##################################")
prop = convert(Dict{Symbol, SFunc}, Dict(:scoreDiff1 => DiscreteCPT(scoreDiff_range, scoreDiffCPT)))
run_static_inference(network,  Importance(make_custom_proposal(prop), 1000))


## Dynamic network - must use async alorithms because of the nature of the example
network = create_dynamic_network()
println("##################################")
println("Running Particle Filtering")
println("##################################")
run_filtering_inference(network, AsyncPF(1000, 10, Int))

println("\n##################################")
println("Running Three Pass Belief Propagation Filtering")
println("##################################")
run_filtering_inference(network, AsyncBP(10, Int))

println("\n##################################")
println("Running Loopy Belief Propagation Filtering")
println("##################################")
run_filtering_inference(network, AsyncLoopy(10, Int))
end