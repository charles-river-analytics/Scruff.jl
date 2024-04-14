import Base.show
import Base.isapprox
using Folds

export
    Factor,

    show,
    product,
    sum_over,
    nextkey,
    normalize

"""
    struct Factor{N}

Representation of a factor over `N` instances.

# arguments
    dims a tuple of dimensions of the instances
    keys ids of the instances
    entries a vector of factor values
"""
struct Factor{N, T <: Real}
    dims :: NTuple{N, Int}
    keys :: NTuple{N, Int}
    entries :: Vector{T}
    function Factor(dims::Nothing, keys::Nothing, entries::Vector{T}) where T
        return new{0, T}((), (), entries)
    end
    function Factor(dims::NTuple{N, Int}, keys, entries::Vector{T}) where {N, T}
        return new{N, T}(dims, keys, entries)
    end
end

"""
    function normalize(f::Factor)

Return a new factor equal to the given factor except that entries sum to 1
"""
function normalize(f::Factor)
    z = sum(f.entries)
    new_entries = [e / z for e in f.entries]
    return Factor(f.dims, f.keys, new_entries)
end

"""
    nextkey()::Int

Produce a fresh instance id that does not conflict with an existing id.
"""
nextkey = begin
    count = 0
    f() = begin
        global count
        count += 1
        return count
    end
    f
end

function Base.isapprox(fact1 :: Factor, fact2 :: Factor, epsilon = 0.0001)
    if fact1.dims != fact2.dims || fact1.keys != fact2.keys
        return false
    end
    len = length(fact1.entries)
    return all(map(i -> isapprox(fact2.entries[i], fact1.entries[i]; atol = epsilon), 1:len))
end

###########################
#                         #
# Pretty printing factors #
#                         #
###########################
function show_boundary(N, space)
    total = (space + 1) * (N + 1) + 1
    for i in 1:total
        print("#")
    end
    println()
end

get_content(content :: String) = content

get_content(content) = sprint(show, content)

function show_spaced(space, content)
    c = get_content(content)
    init_gap = div(space - length(c), 2)
    for i in 1:init_gap
        print(' ')
    end
    print(c)
    for i in 1:space - length(c) - init_gap
        print(' ')
    end
end

function show_content(N, space, content)
    for i in 1:N
        print('#')
        show_spaced(space, content[i])
    end
    print('#')
    for i in 1:space
        print(' ')
    end
    print('#')
    println()
end

"""
    function show(f::Factor)

Print the factor in an easy to read format.
"""
function show(f :: Factor{0})
    # This factor should have been produced by summing out all the variables
    # and should have one row
    space = length(string(f.entries[1])) + 2
    show_boundary(0, space)
    print('#')
    show_spaced(space, f.entries[1])
    println('#')
    show_boundary(0, space)
end

function show(f :: Factor{N}) where N
    max1 = max(map(k -> length(string(k)), f.keys)...)
    max2 = max(map(k -> length(string(k)), f.entries)...)
    space = max(max1, max2) + 2
    show_boundary(N, space)
    show_content(N, space, f.keys)
    show_boundary(N, space)
    mults = Array{Int64}(undef, N)
    mults[N] = 1
    for k in N-1:-1:1
        mults[k] = mults[k+1] * f.dims[k+1]
    end
    for i in eachindex(f.entries)
        vs = []
        j = i-1
        for k in 1:N
            (r, j) = divrem(j, mults[k])
            print("#")
            show_spaced(space, r + 1)
        end
        print("#")
        show_spaced(space, f.entries[i])
        println("#")
    end
    show_boundary(N, space)
end

##################
#                #
# Factor product #
#                #
##################

# The product computation is written so that it can be compiled and optimized
# once the sizes of the input factors (N1 and N2),
# dimensions of the variables (D1 and D2),
# and join indices (J) are known.
struct ProdOp{N1, N2, D1, D2, J} end

function mults(dims)
    n = length(dims)
    result = Array{Int64}(undef, n)
    result[n] = 1
    for k in (n - 1):-1:1
        result[k] = result[k+1] * dims[k+1]
    end
    return result
end

function do_prod(N1, N2, J, f1, f2)
    # The result factor is constructed so that the variables in the second
    # input come first, in order, and then any variables in the first factor
    # that are not in the join come in order.
    # We construct DR, the dimensions of variables in the result factor,
    # as well as the keys of the result factor.
    # We work out the instructions,
    # which will be used to indicate which rows of the two input factors
    # any row of the result factor will be composed from.
    D1 = f1.dims
    D2 = f2.dims

    instructions = Vector{Vector{Tuple{Int, Int}}}(undef, N2)
    NR = N2
    rda = Vector{Int}([d for d in D2])
    rk = Vector{Int}([k for k in f2.keys])
    for j = 1:N2
        instructions[j] = [(2, j)] 
    end

    for (i, j) in J
        if j == 0
            NR += 1
            push!(instructions, [(1, i)])
            push!(rda, D1[i])
            push!(rk, f1.keys[i])
        else
            push!(instructions[j], (1, i))
        end
    end

    DR = NTuple{NR, Int}(rda)
    mults1 = mults(f1.dims)
    mults2 = mults(f2.dims)
    multsr = mults(DR)
    rkeys = NTuple{NR, Int}(rk)
    result = Array{Float64}(undef, Folds.reduce(*, DR))
    for i in eachindex(result)
        # Cartesian indices don't work here because you can't get inside them
        # So we have to build up the index ourselves
        idx1 = Array{Int16}(undef, N1)
        idx2 = Array{Int16}(undef, N2)
        j = i-1
        for k = 1:NR
            instr = instructions[k]
            (r, j) = divrem(j, multsr[k])
            e = r + 1
            if length(instr) == 2
                idx2[instr[1][2]] = e
                idx1[instr[2][2]] = e
            elseif instr[1][1] == 1
                idx1[instr[1][2]] = e
            else
                idx2[instr[1][2]] = e
            end
        end
        x1 = sum(map(q -> (idx1[q] - 1) * mults1[q], 1:N1)) + 1
        x2 = sum(map(q -> (idx2[q] - 1) * mults2[q], 1:N2)) + 1
        result[i] = f1.entries[x1] * f2.entries[x2]
    end

    return Factor(DR, rkeys, result)
end

function product(f1 :: Factor{N1}, f2 :: Factor{N2}) where {N1, N2}
    js = map(k -> findfirst(x -> x == k, f2.keys), f1.keys)
    ijs :: Array{Tuple{Int16, Int16}, 1} = Array{Tuple{Int16, Int16}, 1}(undef, length(f1.keys))
    for i = 1:length(f1.keys)
        j = length(f1.keys) - i + 1
        y = isnothing(js[j]) ? 0 : js[j]
        ijs[i] = (j, y)
    end
    matches = NTuple{N1, Tuple{Int16, Int16}}(ijs)
    return do_prod(N1, N2, matches, f1, f2)
end


###########################
#                         #
# Summing over a variable #
#                         #
###########################

struct SumOp{N, D, I} end

# Special case: Summing out the only variable in a factor
function do_sum(op :: SumOp{1, D, 1}, f :: Factor) where D
    x = sum(f.entries)
    dims = nothing
    keys = nothing
    return Factor(dims, keys, [x])
end

function do_sum(op :: SumOp{N, D, I}, f :: Factor) where {N, D, I}
    rdims = []
    rkeys = []
    for i in 1:I-1
        push!(rdims, f.dims[i])
        push!(rkeys, f.keys[i])
    end
    for i in I+1:N
        push!(rdims, f.dims[i])
        push!(rkeys, f.keys[i])
    end
    rdims = Tuple(rdims)
    rkeys = Tuple(rkeys)

    # The summation is efficeintly implemented by computing a pattern of
    # moves through the input factor's entries as we accumulate the result
    # factor's entries.
    # For any given entry of the result, we add a number of entries of the
    # input. inner_skip is the gap between those entries.
    # We also do inner_skip parallel sums per block, before skipping to the next
    # block. outer_skip is the amount to skip to the next block.
    inner_skip = 1
    for k = I+1:N
        inner_skip *= D[k]
    end
    outer_skip = inner_skip * D[I]
    result_size = inner_skip
    for k = 1:I-1
        result_size *= D[k]
    end
    result = Array{Float64}(undef, result_size)
    if result_size == 0 # This can legitimately happen for unexpanded Expander
        return Factor(rdims, rkeys, result)
    end

    section_start = 1
    result_index = 1
    for j = 1 : div(result_size, inner_skip)
        start = section_start
        for k = 1 : inner_skip
            total = 0.0
            orig_index = start
            for l = 1 : D[I]
                total += f.entries[orig_index]
                orig_index += inner_skip
            end
            result[result_index] = total
            result_index += 1
            start += 1
        end
        section_start += outer_skip
    end

    return Factor(rdims, rkeys, result)
end

function sum_over(f :: Factor{N}, key :: Int) where N
    for i = 1:N
        if f.keys[i] == key
            return do_sum(SumOp{N, f.dims, i}(), f)
        end
    end
end
