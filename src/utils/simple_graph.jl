import Base.copy

export
    Graph,
    add_node!,
    add_undirected!

"""
    struct Graph

A simple graph with nodes (Int), edges (outgoing), and a size
property for each node
"""
struct Graph
    nodes :: Array{Int, 1}
    edges :: Dict{Int, Array{Int, 1}}
    sizes :: Dict{Int, Int}
    Graph() = new([], Dict(), Dict())
    Graph(ns, es, ss) = new(ns, es, ss)
end

function add_node!(g :: Graph, n :: Int, size :: Int)
    if !(n in g.nodes)
        push!(g.nodes, n)
        g.edges[n] = []
    end
    g.sizes[n] = size
end

function add_undirected!(g :: Graph, n1 :: Int, n2 :: Int)
    if n1 in g.nodes && n2 in g.nodes
        if !(n2 in g.edges[n1])
            push!(g.edges[n1], n2)
        end
        if !(n1 in g.edges[n2])
            push!(g.edges[n2], n1)
        end
    end
end

