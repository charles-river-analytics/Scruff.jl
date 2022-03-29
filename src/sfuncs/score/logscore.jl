export LogScore

"""
    struct LogScore{I} <: Score{I}

A Log score.
"""
struct LogScore{I} <: Score{I}
    logscores :: Dict{I, Float64}
    function LogScore(ls::Dict{I, Float64}) where I
        new{I}(ls)
    end
    function LogScore(vs::Vector{I}, ss::Vector{Float64}) where I
        # must handle non-unique values correctly
        d = Dict{I, Float64}()
        for (v,s) in zip(vs,ss)
            d[v] = logsumexp([get(d, v, -Inf), s])
        end
        new{I}(d)
    end
end

@impl begin
    struct LogScoreGetScore end
    function get_score(sf::LogScore{I}, i::I)::AbstractFloat where {I}
        return exp(get_log_score(sf, i))
    end
end

@impl begin
    struct LogScoreGetLogScore end
    function get_log_score(sf::LogScore{I}, x::I)::AbstractFloat where {I}
        return x in keys(sf.logscores) ? sf.logscores[x] : -Inf
    end
end
