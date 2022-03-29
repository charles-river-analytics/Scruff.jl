export SoftScore

"""
    SoftScore(vs::Vector{I}, ss::Vector{Float64})

Return a `LogScore` of the log values in `ss` vector for 
the associated keys in `vs`.
"""
function SoftScore(vs::Vector{I}, ss::Vector{Float64}) where I
    return LogScore(vs, [log(s) for s in ss])
end

"""
    SoftScore(scores::Dict{I,Float64})

Return a `LogScore` of the keys and log values in `score`.
"""
function SoftScore(scores::Dict{I,Float64}) where I
    d = Dict([(k,log(v)) for (k,v) in scores])
    return LogScore(d)
end
