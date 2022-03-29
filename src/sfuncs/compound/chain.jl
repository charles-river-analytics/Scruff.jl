export Chain

"""
    struct Chain{I, J, K, O} <: Conditional{I, J, K, O, Nothing, Q, SFunc{J, O, Nothing}}

A `Conditional` that chains its input `I` through a given function that returns an 
`SFunc{J,O}`.
"""
struct Chain{I, J, K, O} <: Conditional{I, J, K, O, SFunc{J, O}}
    fn::Function
    """
    function Chain(I, J, O, fn)

    Chain an input through `fn`.

    The chain is an `SFunc{K,O}`, where `K` is the concatenation of tuples `I` and `J`.
    `fn` is a function that takes an argument of type `I` and returns an `SFunc{J,O}`.
    The `Chain` defines a generative conditional distribution as follows:
    - Given inputs `i` and `j`
    - Let `s::SFunc{J,O} = fn(i)`
    - Use `j` to generate a value from `s`
    
    For the common case, `Chain` has a special constructors where J is empty.
    """
    function Chain(I, J, O, fn)
        K = extend_tuple_type(I,J)
        new{I, J, K, O}(fn)
    end
    Chain(I, O, fn) = Chain(I, Tuple{}, O, fn)
end

function gensf(ch::Chain{I, J, K, O}, i::I) where {I, J, K, O}
    ch.fn(i) 
end