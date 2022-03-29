

export TimelessInstantModel

"""
    abstract type TimelessInstantModel{I,O} <: InstantModel{I,O}

An InstantModel in which the sfunc made does not depend on time.
Must implement a version of `make_initial` that does not take the current time as argument.
Note that `make_initial` can be defined to take keyword arguments, so the sfunc created
need not be exactly the same every time.
"""

abstract type TimelessInstantModel{I,O} <: InstantModel{I,O} end

make_initial(m::TimelessInstantModel,t;keys...) = make_initial(m;keys...)