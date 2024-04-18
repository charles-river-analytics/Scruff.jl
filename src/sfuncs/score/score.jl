#default implementation
@impl begin
    struct ScoreGetLogScore end
    function get_log_score(sf::Score{I}, i::I)::AbstractFloat where {I}
        return log(get_score(sf, i))
    end
end

include("hardscore.jl")
include("softscore.jl")
include("multiplescore.jl")
include("logscore.jl")
include("functionalscore.jl")
include("normalscore.jl")
include("parzen.jl")
