export Parzen

"""
    struct Parzen <: Score{Float64}

A parzen score.
"""
struct Parzen <: Score{Float64}
    means :: Vector{Float64}
    sd :: Float64
end

@impl begin
    struct ParzenGetScore end
    function get_score(sf::Parzen, i::Float64)::AbstractFloat
        t = 0.0
        for m in sf.means
            t += normal_density(i, m, sf.sd)
        end
        return t / length(sf.means)
    end
end
