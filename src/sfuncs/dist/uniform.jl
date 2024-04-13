# Continuous uniform sfunc

export Uniform

import Distributions

const Uniform{T} = DistributionsSF{Distributions.Uniform{T}, T}
Uniform(lb, ub) = Uniform{typeof(lb)}(lb, ub)

@impl begin
    struct UniformSupport end
    function support(
        sf::Uniform,
        ::NTuple,
        size::Integer,
        curr::Vector{Float64}
    )
        newsize = size - length(curr)
        result = curr
        sfmin = support_minimum(sf, ())
        sfmax = support_maximum(sf, ())
        if newsize > 0
            x = sfmin
            push!(result, x)
            numsteps = newsize - 1
            step = (sfmax - sfmin) / numsteps
            for i in 1:numsteps
                x += step
                push!(result, x)
            end
        end
        unique(result)
    end
end

@impl begin
    struct UniformSupportQuality end
    function support_quality(::Uniform, parranges)
        :IncrementalSupport
    end
end

@impl begin
    struct UniformBoundedProbs end

    # assumes range is sorted
    function bounded_probs(
        sf::Uniform,
        range::Vector{Float64},
        ::NTuple
    )
        l = support_minimum(sf, ())
        u = support_maximum(sf, ())
        d = u - l
        n = length(range)

        # Each element in the range is associated with the interval between the midpoint
        # of it and the point below and the midpoint between it and the point above,
        # except for the end intervals which go to negative or positive infinity.
        points = [-Inf64]
        for i in 2:n
            push!(points, (range[i-1] + range[i]) / 2)
        end
        push!(points, Inf64)

        # Each interval might be completely, partially, or not contained in the bounds
        # of the uniform distribution. The following code determines the length of each
        # interval that is in the bounds.
        lengthsinbound = Float64[]
        for i in 1:n
            a = max(points[i], l)
            b = min(points[i+1], u)
            push!(lengthsinbound, max(b-a, 0.0))
        end

        ps = [lengthsinbound[i] / d for i in 1:n]
        return (ps, ps)
    end

end

@impl begin
    struct UniformComputePi end

    function compute_pi(sf::Uniform,
                    range::Vector{Float64}, 
                    ::NTuple, 
                    ::Tuple)::Cat{Float64}

        ps = bounded_probs(sf, range, ())[1]
        Cat(range, ps)
    end
end
