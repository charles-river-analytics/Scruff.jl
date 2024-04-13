export 
    support_quality_rank,
    support_quality_from_rank,

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

