export 
    support_quality_rank,
    support_quality_from_rank

@impl begin
    struct SFuncExpectation end

    function expectation(sf::SFunc{I,O}, i::I) where {I,O}
        return f_expectation(sf, i, x -> x)
    end
end

"""
    support_quality_rank(sq::Symbol)

Convert the support quality symbol into an integer for comparison.
"""
function support_quality_rank(sq::Symbol)
    if sq == :CompleteSupport return 3
    elseif sq == :IncrementalSupport return 2
    else return 1 end
end

"""
    support_quality_from_rank(rank::Int)

Convert the rank back into the support quality.
"""
function support_quality_from_rank(rank::Int)
    if rank == 3 return :CompleteSupport
    elseif rank == 2 return :IncrementalSupport
    else return :BestEffortSupport() end
end

@impl begin
    struct SFuncSupportQuality end

    function support_quality(s::SFunc, parranges)
            :BestEffortSupport
    end
end

@impl begin
    struct DefaultWeightedValues
        num_samples::Int
    end
    function weighted_values(s::Dist)
        samples = [sample(s, ()) for _ in 1:num_samples]
        return (samples, ones(num_samples))
    end
end

@impl begin
    struct DefaultFitMLE end
    function fit_mle(s::Type{D}, ref::D) where {D <: Dist}
        return ref
    end
end

@impl begin
    struct SampledFitMLEJoint end
    function fit_mle_joint(t::Type{D}, dat::Dist{Tuple{Tuple{}, O}})::D where {O, D <: Dist{O}}
        samples, weights = weighted_values(dat)

        # Just get the output component (rest should be empty tuple)
        samples = [s[2] for s in samples]

        cat = Discrete(samples, weights / sum(weights))

        return fit_mle(t, cat)
    end
end
