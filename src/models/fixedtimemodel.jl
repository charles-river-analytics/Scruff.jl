export 
    FixedTimeModel

"""
    abstract type FixedTimeModel{I,O} <: Model{I,O}

A dynamic model described only for fixed time delta.  Must implement 
`get_initial`, `get_transition`, and `get_dt`.
These can depend on the current time.
"""
abstract type FixedTimeModel{I,J,O} <: Model{I,J,O} end

# get_initial(::FixedTimeModel) = error("Not implemented")
# get_transition(::FixedTimeModel) = error("Not implemented")
# get_dt(::FixedTimeModel) = error("Not implemented")

make_initial(m::FixedTimeModel, t=0) = get_initial(m, t)

function make_transition(m::FixedTimeModel, parenttimes, time)
    dt = get_dt(m)
    # Allow edges from the same time step, or intertemporal edges from the previous time step at distance dt
    if all(t -> time - t == dt || time == t, parenttimes)
        return get_transition(m, time)
    else
        error("make_transition called on FixedTimeModel with incorrect dt")
    end
end

# is_fixed(::FixedTimeModel) = true

