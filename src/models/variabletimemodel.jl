export VariableTimeModel

"""
    abstract type VariableTimeModel{I,J,O} <: Model{I,J,O} 

A model that creates sfuncs based on the time delta between the parents and the current instance.
In general, the deltas can be different for different parents.
This type does not introduce any new functionality over Model.
Its purpose is to make explicit the fact that for this type of model, separate time deltas are possible.
Must implement 'make_initial', which takes the current time, and 'make_transition', which takes the current time and parent times. 
"""
abstract type VariableTimeModel{I,J,O} <: Model{I,J,O} end