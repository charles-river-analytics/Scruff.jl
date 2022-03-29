# Default implementations of basic operators

# cpdf and logcpdf have default operators in terms of the other. Sfuncs should implement one or the other.

@impl begin
    struct SFuncCpdf end
    function cpdf(sf::SFunc{I,O}, i::I, o::O)::AbstractFloat where {I,O}
        exp(logcpdf(sf, i, o))
    end
end

@impl begin
    struct SFuncLogcpdf end
    function logcpdf(sf::SFunc{I,O}, i::I, o::O)::AbstractFloat where {I,O}
        log(cpdf(sf, i, o))
    end
end
