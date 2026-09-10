module Experiments

using Scruff
using Scruff.Utils
using Scruff.RTUtils
using Scruff.Models
using Scruff.SFuncs
using Scruff.Algorithms
using Scruff.Operators
import Scruff: make_initial, make_transition
using StatsBase

function discrete_uniform(range::Array)
    len = length(range)
    Cat(range, fill(1/len, len))
end

possible_values = ["a", "b", "c"]
struct M1 <: VariableTimeModel{Tuple{}, Tuple{String}, String} end 
make_initial(::M1, t) = discrete_uniform(possible_values)
make_transition(::M1, parts, t) =
    Chain(Tuple{String}, String, tuple -> begin 
        parent = tuple[1]
        new_val = discrete_uniform(possible_values)
        Mixture([Constant(parent), new_val], [0.5, 0.5])
    end)

struct M2 <: VariableTimeModel{Tuple{}, Tuple{Float64}, Float64}
end
make_initial(::M2, t) = Normal(1.0, 0.1)
make_transition(::M2, parts, t) =
    Chain(Tuple{Float64}, Float64, tuple -> begin 
        parent = tuple[1]
        Normal(parent, 10.0)
    end)

mutable struct RuntimeContainer
    const alg::Algorithm
    const runtime::Runtime
    const network::Network
    var_dict::Dict{Symbol, Variable}
    time::Int64
end

function initialize_network(
    var_dict, graph, parent_time_offset=VariableParentTimeOffset(), 
    num_samples::Int64=100, range_limited_vars=Symbol[])
    variables = collect(values(var_dict))
    net = DynamicNetwork(variables, VariableGraph(), VariableGraph(graph), parent_time_offset)
    runtime = Runtime(net)
    # decide on algorithm - either AsyncPF or range limited BP
    # alg = AsyncPF(num_samples, num_samples, Int) 
    alg =  SyncBP(num_samples)
    # alg = create_range_limited_bp(num_samples, range_limited_vars)
    println(alg)
    init_filter(alg, runtime)
    return RuntimeContainer(alg, runtime, net, var_dict, 1)
end

function create_range_limited_bp(range_size, range_limited_vars)
    range_sizes = Dict{Symbol, Int64}()
    for var in range_limited_vars
        range_sizes[var] = range_size
    end
    RangeLimited(
        AsyncBP(range_size, Int),
        range_sizes,
    )
end

function run_inference(container::RuntimeContainer, evidence::Dict{Symbol, Score}, queries::Vector)
    alg = container.alg
    runtime = container.runtime
    t = container.time
    if !isa(alg, RangeLimited) && isa(alg.inference_algorithm, Importance)
        particles = get_state(runtime, :particles)
        newParticles = resample(particles)
        set_state!(runtime, :particles, newParticles)
    end
    variables = collect(values(container.var_dict))
    filter_step(alg, runtime, variables, t, evidence)
    results_dict = Dict{Symbol, Dict}()
    for var_name in queries 
        state_var = container.var_dict[var_name]
        belief_state = marginal(alg, runtime, current_instance(runtime, state_var))
        inverse_map = belief_state.__inversemap # need to use inverse map to map to probs
        state_probs = Dict()
        for (state, i) in inverse_map
            state_probs[state] = belief_state.params[i]
        end
        results_dict[var_name] = state_probs  
    end
    container.time += 1
    return results_dict
end

function main()
    m1 = M1()(:model1)
    m2 = M2()(:model2)
    var_dict = Dict{Symbol, Variable}(
        :model1 => m1,
        :model2 => m2
    )
    graph = VariableGraph(
        m1 => [m1],
        m2 => [m2]
    )
    num_samples = 100
    container = initialize_network(var_dict, graph, VariableParentTimeOffset(), num_samples, [:model1, :model2]) 
    for _ in 1:10
        @time result = run_inference(container, Dict{Symbol, Score}(:model1 => HardScore("a")), [:model1, :model2])
            # @test result[:model1] == Dict("c" => 0.0, "b" => 0.0, "a" => 1.0)
        # println(result[:model1])
        # println(sum(map(p -> p[1] * p[2], collect(result[:model2]))))
        # println(minimum(collect(keys(result[:model2]))))
        k = Float64.(collect(keys(result[:model2])))
        v = StatsBase.weights(Float64.(collect(values(result[:model2]))))
        println(sqrt(StatsBase.var(k, v)))
    end
    # result = run_inference(container, Dict{Symbol, Score}(:model1 => HardScore("a")), [:model1, :model2])
end


end