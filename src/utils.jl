module Utils

include("utils/factor.jl")
include("utils/cutils.jl")
include("utils/simple_graph.jl")

# Helper functions for normalize; may be useful elsewhere
function _entries(xs::Array)
    ys = map(_entries, xs)
    result = []
    for y in ys
        append!(result, y)
    end
    return result
end

_entries(d::Dict) = _entries(collect(map(_entries, values(d))))

_entries(xs::Tuple) = _entries([y for y in xs])

_entries(x::Number) = [x]

_entries(::Nothing) = []

_transform(f::Function, xs::Array) = [_transform(f,x) for x in xs]

_transform(f::Function, d::Dict) = Dict([(k,_transform(f,x)) for (k,x) in d])

_transform(f::Function, xs::Tuple) = tuple([_transform(f,x) for x in xs]...)

_transform(f::Function, x::Number) = f(x)

_transform(::Function, ::Nothing) = nothing


"""
    normalize(xs)

Normalize an array of non-negative reals to sum to 1
"""
function normalize(xs)
    tot = 0.0
    ys = _entries(xs)
    for y in ys
        if y < 0
            error("Negative probability in $(ys)")
        end
        tot += y
    end
    # In the case of learning, it is legitimately possible that a case never
    # happens, so all statistics are zero. Therefore we accept this case.
    if tot == 0.0
        f = x -> 1.0 / length(ys)
    else
        f = x -> x / tot
    end
    return _transform(f, xs) 
end


end
