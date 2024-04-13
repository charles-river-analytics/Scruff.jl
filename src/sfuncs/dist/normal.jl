export Normal

import Distributions

"""
    mutable struct Normal <: Dist{Float64}

`Normal` defines an *sfunc* representing unconditional Gaussian distributions.

# Additional supported operators
- `support`
- `support_quality`
- `sample`
- `logcpdf`
- `bounded_probs`
- `compute_pi`
"""

const Normal{T} = DistributionsSF{Distributions.Normal{T}, T}
Normal(mean, sd) = Normal{typeof(mean)}(mean, sd)

@impl begin
    struct NormalSupport end
    function support(
            sf::Normal, 
            parranges::NTuple{N,Vector}, 
            size::Integer, 
            curr::Vector{Float64}) where N
        
        mu = expectation(sf, ())
        sd = variance(sf, ())^0.5

        if isempty(curr)
            oldl = mu
            oldu = mu
        else
            oldl = minimum(filter(x -> x > -Inf, curr))
            oldu = maximum(filter(x -> x < Inf, curr))
        end

        if oldl == oldu
            if size <= 1
                result = [oldl]
                return result
            else
                l = oldl - sd
                u = oldl + sd
                result = Vector{Float64}()
                gap = (u-l) / (size-1)
                push!(result, l)
                for i = 1:(size-1)
                    push!(result, l + i*gap)
                end
                return result
            end
        else
            len = length(curr)
            if size <= len
                return curr
            else # we need to extend and thicken the current range while keeping existing points
                olddiff = oldu - oldl
                oldgap = olddiff / (len-1)
                newl = oldl - olddiff
                newu = oldu + olddiff
                result = copy(curr)
                i = newl
                while i < oldl
                    push!(result, i)
                    i += oldgap
                end
                i = newu
                while i > oldu
                    push!(result, i)
                    i -= oldgap
                end
                # Now we have 3 * len points covering twice the range
                # If size is greater, we will add points in between
                numperinterval = ceil(size / (3*len))
                if numperinterval > 1
                    newgap = oldgap / numperinterval
                    for x in copy(result)
                        for j = 1:(numperinterval-1)
                            push!(result, x + newgap * j)
                        end
                    end
                end
                sort!(result)
                return result
            end
        end
    end
end

@impl begin
    struct NormalSupportQuality end
    function support_quality(::Normal, parranges)
        :IncrementalSupport
    end
end

@impl begin
    struct NormalBoundedProbsBoundedProbs
        numpartitions::Int64 = 10
    end

    function bounded_probs(
            sf::Normal, 
            range::VectorOption{Float64}, 
            parranges::NTuple{N,Vector})::Tuple{Vector{<:AbstractFloat}, Vector{<:AbstractFloat}} where {N}
        
        intervals = make_intervals(range)
        lower = Array{Float64}(undef, length(range))
        upper = Array{Float64}(undef, length(range))
        d = sf.dist
        ls = []
        us = []
        for interval in intervals
            (l, u) = interval
            if l == -Inf || u == Inf
                push!(ls, 0.0)
                push!(us, 1.0)
            else
                diff = (u - l) / numpartitions
                mn = 0.0
                mx = 0.0
                for i = 1:numpartitions
                    il = l + (i-1) * diff
                    iu = l + i * diff
                    p1 = Distributions.pdf(d, il) * diff
                    p2 = Distributions.pdf(d, iu) * diff
                    mn += min(p1,p2)
                    mx += max(p1,p2)
                end
                push!(ls, mn)
                push!(us, mx)
            end
        end
        # We get better bounds by considering the bounds on other intervals.
        # For any point, the lower bound can be normalized using the upper bounds of every other point, and vice versa.
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
            lower[i] = l
            upper[i] = u
        end
        return(lower, upper)
    end
end

# TODO: Replace this with a lazy implicit representation that doesn't require enumerating until the last minute
@impl begin
    struct NormalComputePi end

    function compute_pi(sf::Normal, range::VectorOption{Float64}, parranges::NTuple{N,Vector}, 
            incoming_pis::Tuple)::Dist{Float64} where {N}
        Cat(range, collect(map(x -> Distributions.pdf(sf.dist, x), range)))
    end

end
#=
@impl begin
    struct NormalExpectedStats end

    function expected_stats(sf::Normal, range::VectorOption{Float64}, parranges::NTuple{N,Vector},
            pis::NTuple{M,Dist},
            child_lambda::Score{Float64}) where {N,M}

        pis = [Distributions.pdf(sf.dist, x) for x in range]
        ls = [get_score(child_lambda, r) for r in range]
        prob = pis .* ls
        let totalX = 0.0, totalX2 = 0.0, count = 0.0
            for (i, x) in enumerate(range)
                totalX += x * prob[i]
                totalX2 += x^2 * prob[i]
                count += prob[i]
            end
            return (count,totalX,totalX2)
        end
    end
end

@impl begin
    struct NormalAccumulateStats end
    
    function accumulate_stats(sf::Normal, existing_stats, new_stats)
        existing_stats .+ new_stats
    end
end

@impl begin
    struct NormalInitialStats end

    function initial_stats(sf::Normal)
        return (0.0,0.0,0.0)
    end
end

@impl begin
    struct NormalMaximizeStats end

    function maximize_stats(sf::Normal, stats)
        (count, totalX, totalX2) = stats
        mean = totalX/count
        std = sqrt(totalX2/count - mean^2)
        return (mean, std)
    end
end
=#

Base.hash(n::Normal, h::UInt) = hash(n.dist.σ, hash(n.dist.μ, hash(:Normal, h)))
Base.isequal(a::Normal, b::Normal) = Base.isequal(hash(a), hash(b))
Base.:(==)(a::Normal, b::Normal) = Base.isequal(hash(a), hash(b))
Base.isless(a::Normal, b::Normal) = a.dist.μ < b.dist.μ
Base.:<(a::Normal, b::Normal) = Base.isless(a, b)
