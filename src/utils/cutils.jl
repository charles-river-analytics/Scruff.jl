export
    cartesian_product,
    normalize,
    normalized_product,
    make_intervals,
    linear_value,
    bounded_linear_value,
    normal_density,
    memo,
    doop,
    mult_through,
    add_through,
    ancestors,
    topsort,
    converged_numeric

###############################################
#                                             #
# Cartesian product of arrays                 #
# Result contains every combination of inputs #
#                                             #
###############################################

"""
    cartesian_product(xs::Tuple)
    cartesian_product(xs::Array)

Given an array of arrays, returns the cartesian product of those arrays.

# Examples
```julia-repl
julia> cartesian_product([[1,2],[3,4]])
4-element Array{Array{Int64,1},1}:
 [1, 3]
 [1, 4]
 [2, 3]
 [2, 4]

julia> cartesian_product([[1,2],[3,4],[5,6]])
8-element Array{Array{Int64,1},1}:
 [1, 3, 5]
 [1, 3, 6]
 [1, 4, 5]
 [1, 4, 6]
 [2, 3, 5]
 [2, 3, 6]
 [2, 4, 5]
 [2, 4, 6]
```
"""
cartesian_product(xs::Tuple) = cartesian_product([x for x in xs])

function cartesian_product(xs :: Array)
        if isempty(xs)
        result = Array{Any, 1}[[]]
    else
        yss = cartesian_product(xs[2:end])
        result = Array{Any, 1}[]
        for x in xs[1]
            for ys in yss
                zs = copy(ys)
                pushfirst!(zs, x)
                push!(result, zs)
            end
        end
    end
    return result
end

########################################################
#                                                      #
# Normalize an array of non-negative reals to xum to 1 #
#                                                      #
########################################################

"""
    normalized_product(xss, n)

Compute the product of the given arrays of length n and normalize the result. 
Uses log computations to avoid underflow.
"""
function normalized_product(xss, n)
    rs = zeros(Float64, n)
    for xs in xss
        rs .+= [log(x) for x in xs]
    end
    z = max(rs...)
    rs .-= z
    ps = [exp(x) for x in rs]
    return normalize(ps)
end

#################################
#                               #
# Continuous variable utilities #
#                               #
#################################

"""
    make_intervals(range)

Given a range of values of a continuous variable, create interval bins
surrounding the values
"""
function make_intervals(range)
    srng = copy(range)
    sort!(srng)
    last = -Inf
    result = Tuple{Float64, Float64}[]
    for i = 2:length(srng)
        next = (srng[i-1] + srng[i]) / 2
        push!(result, (last, next))
        last = next
    end
    push!(result, (last, Inf))
    return result
end

"""
    linear_value(weights, bias, continuous_combo)

Weight and sum the `continuous_combo` with the given bias
"""
function linear_value(weights, bias, continuous_combo)
    result = bias
    for (weight, parent_value) in zip(weights, continuous_combo)
        result += weight * parent_value
    end
    return result
end

"""
    bounded_linear_value(weights, bias, interval_combo)

Weight and sum the upper and lower bounds in `interval_combo` with the
given bias
"""
function bounded_linear_value(weights, bias, interval_combo)
    lower = bias
    upper = bias
    # This code assumes that the intervals are correctly ordered
    for (weight, (lower_parent, upper_parent)) in zip(weights, interval_combo)
        if weight < 0
            lower += weight * upper_parent
            upper += weight * lower_parent
        else
            lower += weight * lower_parent
            upper += weight * upper_parent
        end
    end
    return (lower, upper)
end

"""
    normal_density(x, mean, variance)

Get the normal density of `x`
"""
function normal_density(x, mean, variance)
    d = sqrt(2 * pi * variance)
    e = -0.5 * (x-mean) * (x-mean) / variance
    return exp(e) / d
end

"""
    memo(f::Function)

returns a memoized one argument function
"""
function memo(f::Function)
    cache = Dict()
    function apply(arg)
        if arg in keys(cache)
            return cache[arg]
        else
            result = f(arg)
            cache[arg] = result
            return result
        end
    end
    return apply
end

####################################################################################
#                                                                                  #
# Functions for performing an arithmetic operation recursively on a data structure #
#                                                                                  #
####################################################################################

# This avoids having to write both forms of the function.
# Writing both forms could cause ambiguity.
# Will cause stack overflow if neither form is defined.
# Assumes op is commutative.
doop(x::Any, y::Any, op) = doop(y,x,op)

function doop(x::Float64, y::Float64, op::Function)
    return op(x,y)
end

function doop(x::Dict, y::Any, op) 
    result = Dict()
    xf = floatize(x)
    for (k,v) in xf
        result[k] = doop(v, floatize(y), op)
    end
    return result
end

function doop(x::Dict, y::Dict, op)
    result = Dict()
    xf = floatize(x)
    yf = floatize(y)
    for k in keys(xf)
        result[k] = doop(xf[k], yf[k], op)
    end
    return result
end

function doop(xs::Array, y::Any, op)
    result = floatize(xs)
    for i = 1:length(xs)
        result[i] = doop(xs[i], floatize(y), op)
    end
    return result
end

function doop(xs::Array, ys::Array, op)
    xf = floatize(xs)
    yf = floatize(ys)
    return [doop(xf[i], yf[i], op) for i in 1:length(xs)]
end

function doop(xs::Tuple, y::Any, op)
    xf = floatize(xs)
    return ntuple(i -> doop(xf[i], floatize(y), op), length(xf))
end

function doop(xs::Tuple, ys::Tuple, op)
    xf = floatize(xs)
    yf = floatize(ys)
    return ntuple(i -> doop(xf[i], yf[i], op), length(xs))
end

mult_through(x,y) = doop(x, y, (x,y) -> x*y)

add_through(x,y) = doop(x, y, (x,y) -> x+y)

floatize(x) = _transform(y -> Float64(y), x)
#=
 Topological sort 
=#

"""
    ancestors(graph :: Dict{U, Vector{U}}, var :: U, found:: Set{U}) :: Vector{U} where U

Find the ancestors of the given value x in the graph.  Found is a set of previously 
found ancestors, to handle cyclic graphs and avoid infinite loops
"""
# Need to handle cyclic graphs
function ancestors(graph :: Dict{U, Vector{U}}, var :: U, found:: Set{U}) :: Vector{U} where U
    result = Vector{U}()
    for par in get(graph, var, [])
        if isa(par, U) && !(par in found)
            push!(found, par) # Prevent infinite recursion
            append!(result, ancestors(graph, par, found))
            push!(result, par) # Guarantee that the ancestors appear before par
        end
    end
    return result
end

"""
    topsort(graph::Dict{T, Vector{T}}) :: Vector{T} where T

Performs a topological sort on the given graph. In a cyclic graph, the order of variables in the cycle
is arbitrary, but they will be correctly sorted relative to other variables.
"""
function topsort(graph::Dict{U, Vector{U}}) :: Vector{U} where U
    result = Vector{U}()
    found = Set{U}()
    for var in keys(graph)
        if !(var in found)
            push!(found, var)
            s = Set{U}([var])
            ancs = ancestors(graph, var, s) # guaranteed to be in topological order
            for anc in ancs
                if !(anc in found)
                    push!(found, anc)
                    push!(result, anc)
                end
            end
            push!(result, var) # after the ancestors
        end
    end
    return result
end

diff(x :: Number, y :: Number) = abs(x-y)

function diff(xs :: Dict, ys :: Dict)
    total = 0.0
    for (k,v) in keys(xs)
        total += diff(xs[k], ys[k])
    end
    return total
end

function diff(xs, ys)
    total = 0.0
    for i = 1:length(xs)
        total += diff(xs[i], ys[i])
    end
    return total
end

numparams(x :: Number) = 1

numparams(xs) = sum(map(numparams, xs))

function converged_numeric(oldp, newp, eps::Float64 = 0.01, convergebymax::Bool = false)
    totaldiff = 0.0
    num = 0
    if !(length(newp[k]) == length(oldp[k]))
        return false
    end
    d = diff(newp, oldp)
    n = numparams(newp)
    if convergebymax && d / n >= eps
        return false
    end
    totaldiff += d
    num += n
    return totaldiff / num < eps
end
