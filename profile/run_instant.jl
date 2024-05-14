function run_instant(algorithm)
    x1m = Cat([1,2], [0.1, 0.9])()
    x2m = Cat([1,2,3], [0.2, 0.3, 0.5])()
    cpd2 = Dict((1,1) => [0.3, 0.7], (1,2) => [0.6, 0.4], (2,1) =>[0.4, 0.6],
                (2,2) => [0.7, 0.3], (3,1) => [0.5, 0.5], (3,2) => [0.8, 0.2])
    x3m = DiscreteCPT([1,2], cpd2)()
    x4m = DiscreteCPT([1,2], Dict((1,) => [0.15, 0.85], (2,) => [0.25, 0.75]))()
    x5m = DiscreteCPT([1,2], Dict((1,) => [0.35, 0.65], (2,) => [0.45, 0.55]))()

    x1 = x1m(:x1)
    x2 = x2m(:x2)
    x3 = x3m(:x3)
    x4 = x4m(:x4)
    x5 = x5m(:x5)

    fivecpdnet = InstantNetwork(Variable[x1,x2,x3,x4,x5], VariableGraph(x3=>[x2,x1], x4=>[x3], x5=>[x3]))

    runtime = Runtime(fivecpdnet)
    default_initializer(runtime)
    evidence = Dict(:x1 => HardScore(2))
    infer(algorithm, runtime, evidence)
    inst5 = current_instance(runtime, x5)
    probability(algorithm, runtime, inst5, 2)
end