#default implementation
@impl begin
    struct ScoreGetLogScore end
    function get_log_score(sf::Score{I}, i::I)::AbstractFloat where {I}
        return log(get_score(sf, i))
    end
end

