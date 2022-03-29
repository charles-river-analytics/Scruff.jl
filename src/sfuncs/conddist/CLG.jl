export 
    CLG

"""
    CLG(paramdict::Dict) 

Constructs an*sfunc representing a Conditional linear Gaussian.  These sfuncs may have 
both discrete and continuous parents.  For each combination of discrete parents, there is 
a `LinearGaussian` that depends on the continuous parents.

`CLG`s are implemented as a `Table` with a `LinearGaussian`.

The `paramdict` parameter defines the discrete and continuous parents, and the linear 
gaussean values where the length of a key is the count of the discrete inputs, the 
length of the tuple in a value is the count of continuous inputs, and the rest of the
values are used to build the parameters for `CLG` itself.  For example,

```
    Dict((:x,1) => ((-1.0, 1.0, 2.0), 3.0, 1.0), 
         (:x,2) => ((-2.0, 4.0, 2.0), 3.0, 1.0),
         (:x,3) => ((-3.0, 2.0, 2.0), 3.0, 1.0), 
         (:y,1) => ((-4.0, 5.0, 2.0), 3.0, 1.0),
         (:y,2) => ((-5.0, 3.0, 2.0), 3.0, 1.0), 
         (:y,3) => ((-6.0, 6.0, 2.0), 3.0, 1.0))
```

    - the keys define two(2) discrete parents, with values `[:x,:y]` and `[1,2,3]`
    - in the values, the first tuple defines three(3) continuous parents for each
      underlying `LinearGausian`, with values `-1.0:-6.0`, `1.0:6.0`, and `2.0`
    - the values `3.0` and `1.0` are mean/stddev of the underlying `LinearGaussian`
    
See also: [`Table`](@ref), [`LinearGaussian`](@ref)
"""
function CLG(paramdict::Dict) 
    numdiscreteinputs = length(collect(keys(paramdict))[1])
    numcontinuousinputs = length(collect(values(paramdict))[1][1])
    sfmaker(weights) = LinearGaussian(weights[1], weights[2], weights[3])
    return Table(NTuple{numcontinuousinputs, Float64}, Float64, numdiscreteinputs, paramdict, sfmaker)
end

#=

This representation is deprecated, but there may be useful implementation details.

# TODO (MRH): Convert CLG constructor into a call to a method that builds a Tables as in make_CLG above

struct CLG{NumDiscreteInputs, NumContinuousInputs, O} <:
    SFunc{Tuple{NTuple{NumDiscreteInputs, O},
                NTuple{NumContinuousInputs, Float64}},
          Float64,
          Tuple{}} # TODO (MRH): Params type
    # each segment corresponds to one setting of the discrete parents
    # and includes a linear weight for each continuous parent and a bias term
    segments :: Dict{<: Array{O, N} where N, <: Tuple{Array{Float64, N} where N, Float64}}
    variance :: Float64

    function CLG(num_discrete_inputs, num_continous_inputs, seg, var)
        T = typeof(seg).parameters[1].parameters[1]
        new{num_discrete_inputs, num_continous_inputs, T}(seg, var)
    end
end

#############################
#                           #
# Helper functions for CLGs #
#                           #
#############################

# Return all 2-tuples of discrete parent combinations and continuous
# parent combinations
function get_parent_combos(:: CLG{M,N}, parent_ranges) where {M,N}
    discrete_ranges = Array{Array{Any, 1}, 1}(parent_ranges[1:M])
    continuous_ranges = Array{Array{Any, 1}, 1}(parent_ranges[M+1:M+N])
    if isempty(discrete_ranges)
        discrete_combos = [[]]
    else
        discrete_combos = cartesian_product(discrete_ranges)
    end
    if isempty(continuous_ranges)
        continuous_combos = [[]]
    else
        continuous_combos = cartesian_product(continuous_ranges)
    end
    l = Array{Array{Any,1},1}([discrete_combos, continuous_combos])
    return map(a -> Tuple(a), cartesian_product(l))
end

function overlaps(int1, int2) :: Bool
    (l1, u1) = int1
    (l2, u2) = int2
    return l1 <= l2 && u1 >= l2 || l1 <= u2 && u1 >= u2
end

# Return the minimum and maximum probabilities of a normal distribution
# over an interval, when the mean is bounded and the variance is given.
function minmaxprob(lower, upper, lower_mean, upper_mean, variance)
    if lower == -Inf || upper == Inf return (0,1) end
    # The point can be anywhere between lower and upper
    # The mean can be anywhere between lower_mean and upper_mean
    # We need to find the minimum and maximum possible density
    # To find this, we compute the minimum and maximum possible distances
    # from the point to the mean.
    # If the point interval overlaps the mean interval, the minimum possible
    # distance is zero.
    # Otherwise the minimum possible distance is the min distance between
    # mean endpoints and interval endpoints.
    # The maximum possible distance is always the max distance between mean
    # endpoints and interval endpoints.
    d1 = abs(lower_mean - lower)
    d2 = abs(upper_mean - lower)
    d3 = abs(lower_mean - upper)
    d4 = abs(upper_mean - upper)
    dmin = overlaps((lower, upper), (lower_mean, upper_mean)) ? 0 :
             max(min(d1, d2, d3, d4), 0)
    dmax = max(d1, d2, d3, d4)
    densmin = normal_density(dmax, 0, variance)
    densmax = normal_density(dmin, 0, variance)
    diff = upper - lower
    pmin = min(max(densmin * diff, 0), 1)
    pmax = min(max(densmax * diff, 0), 1)
    return (pmin, pmax)
end

# Compute numerical lower and upper bounds on the density by partitioning
# the interval into num_partitions and computing bounds in each partition
function numerical_bounds(lower, upper, lmean, umean,
                          num_partitions, variance)
    if lower == -Inf || upper == Inf
        return (0,1)
    end
    start = lower
    step = (upper - lower) / num_partitions
    l = 0.0
    u = 0.0
    for i = 1:num_partitions
        (x,y) = minmaxprob(start, start + step, lmean, umean, variance)
        l += x
        u += y
        start += step
    end
    return (l,u)
end

#= ============================================================
   Operators
=============================================================== =#

function make_range(sf :: CLG, parranges, size :: Int)
    sd = sqrt(sf.variance)
    parent_combos = get_parent_combos(sf, parranges)

    # Prepare for accumulating the list of candidates by determining
    # whether the mean should be included for each segment and the number
    # of candidates around the mean. We choose the number of values per
    # segment so that the total is approximately the number of values desired.
    values_per_segment = max(div(size, length(parent_combos)), 1)
    if mod(values_per_segment, 2) == 1
        use_mean = true
        values_per_segment -= 1
    else
        use_mean = false
    end
    pairs_per_segment = values_per_segment / 2

    # Get candidates for including in the range
    # For each combinatiuon of parent values, we get the segment according
    # to the discrete values and the mean according to the continuous values.
    # We then spread candidates around the mean according to the standard
    # deviation.
    candidates = []
    for (discrete_combo, continuous_combo) in parent_combos
        (weights, bias) = sf.segments[discrete_combo]
        mean = linear_value(weights, bias, continuous_combo)
        if use_mean
            push!(candidates, mean)
            for i = 1:pairs_per_segment
                push!(candidates, mean - i * sd)
                push!(candidates, mean + i * sd)
            end
        else
            for i = 1:pairs_per_segment
                push!(candidates, mean - (i - 0.5) * sd)
                push!(candidates, mean + (i - 0.5) * sd)
            end
        end
    end
    sort!(candidates)

    # Select size of the candidates evenly spread
    skip = div(length(candidates), size)
    position = div(mod(length(candidates), size), 2) + 1
    result = []
    for i = 1:size
        push!(result, candidates[position])
        position += skip
    end

    return result
end

@op_impl begin
    struct CLGSupport{NumDiscreteInputs, NumContinuousInputs, O} <: Support{CLG{NumDiscreteInputs, NumContinuousInputs, O}} end
    function support((parranges, size, curr)) 
        if isempty(parranges)
            return nothing
        end
        
        old_size = length(curr)
        if old_size >= size
            return curr
        end
        result = make_range(sf, parranges, size)
        # We must make sure that any value in the current range is kept
        # so we replace the closest element in the new range with an element
        # from the current range
        i = 1
        j = 1
        is_current = fill(false, size)
        while i <= old_size && j <= size
            if curr[i] < result[j]
                if j == 1 || is_current[j-1] ||
                        result[j] - curr[i] < curr[i] - result[j-1]
                    result[j] = curr[i]
                    is_current[j] = true
                    i += 1
                    j += 1
                else
                    result[j-1] = curr[i]
                    is_current[j-1] = true
                    i += 1
                end
            else
                j += 1
            end
        end
        if i <= old_size
            tail = old_size - i
            result[size-tail:new_size] = curr[old_size-tail:old_size]
        end
        return result
    end
end


# We bound the integral under the density by dividing each interval of the
# range into num_partitions partitions and computing the lower and upper
# bound in each partition. However, this is complicated by the fact that
# we don't know the density function because we only have ranges for the
# parents.
@op_impl begin
    mutable struct CLGBoundedProbs <: BoundedProbs{CLG} 
        numpartitions::Int64 = 100
    end

    function bounded_probs((range, parrange))
        vec = typeof(sf).parameters
        M = vec[1]
        N = vec[2]

        intervals = make_intervals(range)
        prs = Array{Array{Union{Symbol, Tuple{Float64, Float64}}, 1}, 1}(undef, M+N)
        for i = 1:M
            prs[i] = parrange[i]
        end
        for i = M+1:M+N
            prs[i] = make_intervals(parrange[i])
        end

        parent_combos = get_parent_combos(sf, prs)
        # Normally, the length of the range and the number of intervals is the same
        # However, for an empty range, there is still one unbounded interval
        if isempty(range)
            lower = Array{Float64}(undef, length(parent_combos))
            upper = Array{Float64}(undef, length(parent_combos))
        else
            lower = Array{Float64}(undef, length(parent_combos) * length(range))
            upper = Array{Float64}(undef, length(parent_combos) * length(range))
        end
        pos = 1
        for (discrete_combo, continuous_combo) in parent_combos
            (weights, bias) = sf.segments[discrete_combo]
            (lmean, umean) = bounded_linear_value(weights, bias, continuous_combo)
            ls = []
            us = []
            for interval in intervals
                (il, iu) = interval
                (l,u) = numerical_bounds(il, iu, lmean, umean,
                                op_impl.numpartitions, sf.variance)
                push!(ls, l)
                push!(us, u)
            end
            # We get better bounds by considering the bounds on other intervals.
            # This is especially important for intervals with -Inf or Inf as an
            # endpoint.
            for i = 1:length(intervals)
                otherls = 0.0
                otherus = 0.0
                for j = 1:i-1
                    otherls += ls[j]
                    otherus += us[j]
                end
                for j = i+1:length(intervals)
                    otherls += ls[j]
                    otherus += us[j]
                end
                l = max(ls[i], 1 - otherus)
                u = min(us[i], 1 - otherls)
                lower[pos] = l
                upper[pos] = u
                pos += 1
            end
        end
        return (lower, upper)
    end
end
=#