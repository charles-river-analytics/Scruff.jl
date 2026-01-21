module SoccerExampleNew
using Scruff
using Scruff.SFuncs
using Scruff.Operators
using Scruff.Algorithms
using Scruff.Models
using Scruff.RTUtils
using Scruff.Utils
#using Plots

# define parameters
T = 90 # number of time steps
winning_range = [:us, :them, :none]
confident_range = [true, false]
confident_prior = [0.4, 0.6]
possession_range = [true, false]
goal_range = [true, false]

confidentCPT = Dict( # confident(t-1) and winning
    (true, :us) =>[0.9, 0.1],
    (false, :us) => [0.5, 0.5],
    (true, :them) => [0.5, 0.5],
    (false, :them) => [0.1, 0.9],
    (true, :none) => [0.7, 0.3],
    (false, :none) => [0.3, 0.7])

goalCPT = Dict( # possession and confident
    (true, true) => [0.04, 0.96],
    (false, true) => [0.045, 0.955],
    (true, false) => [0.01, 0.99],
    (false, false) => [0.02, 0.98])

possessionCPT = Dict(
    (true,) => [0.7, 0.3],
    (false,) => [0.3, 0.7])

function fun_winning(score_diff::Int)::Symbol
    if (score_diff>0)
        return :us
    elseif (score_diff == 0)
        return :none
    else
        return :them
    end
end

function fun_score_diff(scoreDiff_tmin1::Int, goal_t::Bool, possession_t::Bool)::Int
    if (goal_t && possession_t) # goal scored, we had the ball
        return scoreDiff_tmin1 + 1
    elseif (goal_t && !possession_t) # goal scored, they had the ball
        return scoreDiff_tmin1 - 1
    else
        scoreDiff_tmin1
    end
end

function create_network_unrolling(T)
    scoreDiff_tmin1 = Constant(0)()(:scoreDiff0)
    confident_tmin1 = Cat(confident_range, confident_prior)()(:confident0)
    vars = Variable[scoreDiff_tmin1, confident_tmin1]
    parents = VariableGraph()
    for i = 1:T
        winning_t = Det(Tuple{Int}, Symbol, fun_winning)()(Symbol(string(:winning) * string(i)))
        confident_t = DiscreteCPT(confident_range, confidentCPT)()(Symbol(string(:confident) * string(i)))
        possession_t = DiscreteCPT(possession_range, possessionCPT)()(Symbol(string(:possession) * string(i)))
        goal_t = DiscreteCPT(goal_range, goalCPT)()(Symbol(string(:goal) * string(i)))
        scoreDiff_t = Det(Tuple{Int, Bool, Bool}, Int, fun_score_diff)()(Symbol(string(:scoreDiff) * string(i)))
       
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
    scoreDiff_init = Constant(0)
    scoreDiff_trans = Det(Tuple{Int, Bool, Bool}, Int, fun_score_diff)
    scoreDiff_t = HomogeneousModel(scoreDiff_init, scoreDiff_trans)(:scoreDiff)

    confident_init = Cat(confident_range, confident_prior)
    confident_cpt = DiscreteCPT(confident_range, confidentCPT)
    confident_t = HomogeneousModel(confident_init, confident_cpt)(:confident)

    winning_init = Constant(:none)
    winning_trans = Det(Tuple{Int}, Symbol, fun_winning)
    winning_t = HomogeneousModel(winning_init, winning_trans)(:winning)

    possession_init = Constant(true)
    possession_cpt = DiscreteCPT(possession_range, possessionCPT)
    possession_t = HomogeneousModel(possession_init, possession_cpt)(:possession)

    goal_init = Constant(false)
    goal_cpt = DiscreteCPT(goal_range, goalCPT)
    goal_t = HomogeneousModel(goal_init, goal_cpt)(:goal)
    
    vars = Variable[scoreDiff_t, confident_t, winning_t, possession_t, goal_t]
    parents = VariableGraph()
    parents[winning_t] = [scoreDiff_t]
    parents[confident_t] = [confident_t, winning_t]
    parents[possession_t] = [confident_t]
    parents[goal_t] = [possession_t, confident_t]
    parents[scoreDiff_t] = [scoreDiff_t, goal_t, possession_t]

    parent_time_offset = VariableParentTimeOffset()
    push!(parent_time_offset, winning_t => scoreDiff_t)
    network = DynamicNetwork(vars, VariableGraph(), parents, parent_time_offset)
    return network
end

function run_filtering_inference(network, alg, game_observations)
    runtime = Runtime(network)
    init_filter(alg, runtime)
    vars = get_variables(network)
  
    ev_goal_times = game_observations[:goal]
    ev_possession_times = game_observations[:possession_us]

    confv = []
    possessionv = []
    winningv = []
    scorediffv= []
    goalv = []

    for i=1:T
        println("i=$i")
        # create evidence for time i 
        ev = Dict{Symbol, Score}()
        if (i in ev_goal_times)
            @info "Passing evidence for minute $i of goal scored"
            ev[:goal] = HardScore(true)
        end
        if (i in ev_possession_times)
            @info "Passing evidence for minute $i of possession us"
            ev[:possession] = HardScore(true)
        end

        filter_step(alg, runtime, vars, i, ev)

        conf_i = current_instance(runtime, get_node(network,:confident))
        p_conf_i = probability(alg, runtime, conf_i, true)
        println("Confident at minute $i is : $p_conf_i")
        push!(confv, p_conf_i)

        poss_i = current_instance(runtime, get_node(network, :possession))
        p_poss_i = probability(alg, runtime, poss_i, true)
        println("Possession at minute $i is : $p_poss_i")
        push!(possessionv, p_poss_i)

        goal_i = current_instance(runtime, get_node(network, :goal))
        p_goal_i = probability(alg, runtime, goal_i, true)
        println("Goal at minute $i is : $p_goal_i")
        push!(goalv, p_goal_i)

        scoreDiff_i = current_instance(runtime, get_node(network, :scoreDiff))
        mean_score = Scruff.Algorithms.mean(alg, runtime, scoreDiff_i)
        println("Score is $mean_score at minute $i")
        push!(scorediffv, mean_score)

        winning_i = current_instance(runtime, get_node(network, :winning))
        p_winning_i = probability(alg, runtime, winning_i, :us)
        println("Winning at minute $i is : $p_winning_i")
        push!(winningv, p_winning_i)
    end
    # fig = plot(layout = grid(5,1), legend=false)
    # timev = 1:T
    # plot!(fig[1], timev, winningv, title = "us winning") 
    # plot!(fig[2], timev, confv, title = "our confidence")
    # plot!(fig[3], timev, possessionv, title = "our posession") 
    # plot!(fig[4], timev, goalv, title="goal scored") 
    # plot!(fig[5], timev, scorediffv, title="score difference") 
    # display(fig)

end

function run_static_inference(network, alg)
    runtime = Runtime(network) # create runtime
    
    # Set evidence.
    evidence = Dict{Symbol, Score}(:goal20 => HardScore(true), :possession20 => HardScore(true))

    # Perform inference.
    infer(alg, runtime, evidence)

    # Get updated beliefs 
    for i=1:T
        conf_i = current_instance(runtime, get_node(network, Symbol(string(:confident) * string(i))))
        println("Confident at minute $i is : $(probability(alg, runtime, conf_i, true))")

        poss_i = current_instance(runtime, get_node(network, Symbol(string(:possession) * string(i))))
        println("Possession at minute $i is : $(probability(alg, runtime, poss_i, true))")

        goal_i = current_instance(runtime, get_node(network, Symbol(string(:goal) * string(i))))
        println("Goal at minute $i is : $(probability(alg, runtime, goal_i, true))")

        scoreDiff_i = current_instance(runtime, get_node(network, Symbol(string(:scoreDiff) * string(i))))
        mean_score = Scruff.Algorithms.mean(alg, runtime, scoreDiff_i)
        println("Score is $mean_score at minute $i")

        winning_i = current_instance(runtime, get_node(network, Symbol(string(:winning) * string(i))))
        println("Winning at minute $i is : $(probability(alg, runtime, winning_i, :us))")
    end
end

# Observations
game_observations = Dict{Symbol, Vector{Int}}(:goal => [20], :possession_us =>[20])

# Static network
network = create_network_unrolling(T) #create network
# println("##################################")
# println("Running Three Pass Belief Propagation")
# println("##################################")
# run_static_inference(network, ThreePassBP(100)) # run three pass belief propagation

# println("\n##################################")
# println("Running Loopy Belief Propagation")
# println("##################################")
# run_static_inference(network, LoopyBP()) # run loopy belief propagation
# VE too slow for this example
#println("\n##################################")
# println("Running VE")
# println("##################################")
# run_static_inference(network, VE(get_variables(network))) # run VE
println("\n##################################")
println("Running Likelihood Weighting")
println("##################################")
run_static_inference(network,  LW(1000))


# Dynamic network - must use async alorithms because of the nature of the example
network = create_dynamic_network()
println("##################################")
println("Running Particle Filtering")
println("##################################")
run_filtering_inference(network, SyncPF(1000), game_observations)

# println("\n##################################")
# println("Running Three Pass Belief Propagation Filtering")
# println("##################################")
# run_filtering_inference(network, AsyncBP(10, Int))

# println("\n##################################")
# println("Running Loopy Belief Propagation Filtering")
# println("##################################")
# run_filtering_inference(network, AsyncLoopy(10, Int))
end