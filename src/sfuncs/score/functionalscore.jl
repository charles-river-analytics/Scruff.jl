export FunctionalScore

"""
    struct FunctionalScore{I} <: Score{I}

A score defined by a function.
"""
struct FunctionalScore{I} <: Score{I}
    fn :: Function # Function I => Double
end

@impl begin
    struct FunctionalScoreGetScore end
    function get_score(sf::FunctionalScore{I}, i::I)::AbstractFloat where {I}
        return sf.fn(i)
    end
end

