function run_instant(n1, n2, n3, n4, n5, algorithm)
    fivecpdnet = make_five_node_instant_network(n1, n2, n3, n4, n5)
    runtime = Runtime(fivecpdnet)
    default_initializer(runtime)
    evidence = Dict(:x1 => HardScore(2))
    infer(algorithm, runtime, evidence)
    inst5 = current_instance(runtime, get_node(get_network(runtime), :x5))
    probability(algorithm, runtime, inst5, 2)
end