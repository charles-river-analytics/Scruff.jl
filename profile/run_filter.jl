function run_filter(n1, n2, algorithm, num_steps)
    (net, v1, v2) = make_two_node_dynamic_network(n1, n2)
    runtime = Runtime(net)
    init_filter(algorithm, runtime)
    for i in 1:num_steps
        filter_step(algorithm, runtime, Variable[v1, v2], 2, Dict{Symbol, Score}(:v2 => HardScore(2)))
    end
    probability(algorithm, runtime, v2, 2)
end