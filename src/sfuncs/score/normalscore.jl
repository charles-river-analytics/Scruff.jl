export NormalScore

"""
    struct NormalScore <: Score{Float64}

A score defined by a normal density given the mean and sd of the score.
"""
struct NormalScore <: Score{Float64}
    mean :: Float64
    sd :: Float64
end

@impl begin
    struct NormalScoreGetScore end
    function get_score(sf::NormalScore, i::Float64)::AbstractFloat
        return normal_density(i, sf.mean, sf.sd)
    end
end
