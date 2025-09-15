export SimpleModel

"""
    struct SimpleModel{I,O} <: TimelessInstantModel{I,O}

A model that always produces the same SFunc.
This is an TimelessInstantModel, so must always be called when the parents are the same time point.
The constructor takes the sfunc as argument, which is stored.
There is a convenience method to create a SimpleModel for any sfunc by applying the sfunc to zero arguments.
"""
struct SimpleModel{I, O} <: TimelessInstantModel{I, O}
    sf::SFunc{I, O}
end

make_initial(m::SimpleModel) = m.sf

# Convenience for making constant models, a very common case
(sf::SFunc{I, O})() where {I, O} = SimpleModel{I, O}(sf)
