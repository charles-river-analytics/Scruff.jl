export Flip

"""
    Flip(p)

Constructs a very simple *sfunc* corresponding to a Bernoulli distribution, 
represented by a `Cat`.  The output is `true` with probability `p`, and `false`
with probability `1-p`.

See also: [`Cat`](@ref)
"""
Flip(p) = Cat([false, true], [1-p, p])
