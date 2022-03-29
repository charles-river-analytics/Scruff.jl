export 
    InstantModel

"""
    abstract type InstantModel{I,O} <: Model{I,Nothing,O} 

A model for a variable with no time dependencies.
Since this model has no transitions, it can only called with 
`make_initial` and not `make_transition` 
(i.e. `make_transition` = `make_initial`)
"""
abstract type InstantModel{I,O} <: Model{I,Nothing,O} end

# Instant models can be freely used dynamically, with no time dependencies
make_transition(m::InstantModel, parenttimes, time) = make_initial(m)
