#= 
# Default implementations of operators. These are defined using other operators.
# They will always be called if a more specific implementation is not provided.
# If the operators they rely on are not implemented, they will produce a runtime error.
# Writers of default implementations should avoid infinite loops.
=#

# if forward is defined, we get a default implementation of sample
@impl begin
    struct DefaultSample end
    # This should not produce an infinite loop, because a dist should not implement forward,
    # since forwards maps a parent to a dist, but here the parent is empty.
    function sample(sf::SFunc{I,O}, i::I)::O where {I,O}
        d = forward(sf, i)
        return sample(d, tuple())
    end
end

@impl begin
    function cpdf(sf::SFunc{I,O}, i::I, o::O)::AbstractFloat where {I,O}
        exp(logcpdf(sf, i, o))
    end
end

@impl begin
    function logcpdf(sf::SFunc{I,O}, i::I, o::O)::AbstractFloat where {I,O}
        log(cpdf(sf, i, o))
    end
end

# TODO: Create default implementations of compute_pi and send_lambda
# TODO: Create weighted_sample operator with default implementation using inverse
