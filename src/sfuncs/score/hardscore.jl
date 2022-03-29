export HardScore

"""
    HardScore{I} <: Score{I}

A fixed score.
"""
struct HardScore{I} <: Score{I}
    value :: I
end

@impl begin
    struct HardScoreGetScore end
    function get_score(sf::HardScore{I}, i::I)::AbstractFloat where {I}
        i == sf.value ? 1.0 : 0.0
    end
end
  
