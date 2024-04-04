export
    Algorithm,
    marginal,
    joint,
    probability,
    mean,
    variance

"""
    Algorithm

The supertype of all algorithms.

A standard set of queries is defined for algorithms. Any given subtype of `Algorithm`
will implement a subset of these queries.
"""
abstract type Algorithm end

