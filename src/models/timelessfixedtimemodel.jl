export TimelessFixedTimeModel

"""
    abstract type TimelessFixedTimeModel{I,J,O} <: FixedTimeModel{I,J,O}

A FixedTimeModel in which the initial and transition models do not depend on the current time.
In addition to `get_dt`, must implement a version of `get_initial` and `get_transition` that
do not take the current time as an argument.
"""
abstract type TimelessFixedTimeModel{I,J,O} <: FixedTimeModel{I,J,O} end

get_initial(m::TimelessFixedTimeModel, t) = get_initial(m)

get_transition(m::TimelessFixedTimeModel, t) = get_transition(m)
