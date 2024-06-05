random_probvec(n) = normalize([rand() for i in 1:n])

function make_cat(n)
    rangevec = [i for i in 1:n]
    probabilities = random_probvec(n)
    Cat(rangevec, probabilities)
end

function make_cpd_1parent(n, p)
    rangevec = [i for i in 1:n]
    parrangevec = [(j,) for j in 1:p]
    dict = Dict{Tuple{Int}, Vector{Float64}}()
    for tj in parrangevec
        dict[tj] = random_probvec(n)
    end
    DiscreteCPT(rangevec, dict)
end

function make_cpd_2parents(n, p1, p2)
    rangevec = [i for i in 1:n]
    combovec = [(i,j) for i in 1:p1 for j in 1:p2]
    dict = Dict{Tuple{Int, Int}, Vector{Float64}}()
    for k in combovec
        dict[k] = random_probvec(n)
    end
    DiscreteCPT(rangevec, dict)
end        

# Makes an instant network with five integer-ranged nodes: x1, x2, x3, x4, x5
# n1, n2, n3, n4, and n5 are the number of values in the range of the five nodes.
# x1 and x2 are Cat roots, x3 has a DiscreteCPT that depends on x1 and x2,
# and x4 and x5 both depend on x3 with a DiscreteCPT.
function make_five_node_instant_network(n1, n2, n3, n4, n5)
    x1m = SimpleModel(make_cat(n1))
    x2m = SimpleModel(make_cat(n2))
    x3m = SimpleModel(make_cpd_2parents(n3, n2, n1))
    x4m = SimpleModel(make_cpd_1parent(n4, n3))
    x5m = SimpleModel(make_cpd_1parent(n5, n3))
    x1 = x1m(:x1)
    x2 = x2m(:x2)
    x3 = x3m(:x3)
    x4 = x4m(:x4)
    x5 = x5m(:x5)
    InstantNetwork(Variable[x1,x2,x3,x4,x5], VariableGraph(x3=>[x2,x1], x4=>[x3], x5=>[x3]))
end