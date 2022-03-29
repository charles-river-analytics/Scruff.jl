export HomogeneousModel

"""
    HomogeneousModel{I,O} <: FixedTimeModel{I,O}

A dynamic model with a fixed time step and same transition model at every time point..
The constructor is called with the initial sfunc and transition sfuncs.
The constructor is also called with an optional dt (defaults to 1).
"""
struct HomogeneousModel{I,J,O} <: TimelessFixedTimeModel{I,J,O}
    initial :: SFunc{I,O}
    transition :: SFunc{J,O}
    dt :: Number
end
function HomogeneousModel(init::SFunc{I,O}, trans::SFunc{J,O}, dt = 1) where {I,J,O} 
    HomogeneousModel{I,J,O}(init, trans, dt)
end

get_initial(m::HomogeneousModel) = m.initial

get_transition(m::HomogeneousModel) = m.transition

get_dt(m::HomogeneousModel) = m.dt
