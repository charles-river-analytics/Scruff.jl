export StaticModel

"""
    struct StaticModel{I,O} <: VariableTimeModel{I,O,O} end

A static model represents a variable that never changes its value.
The value is setup through an sfunc created by make_initial.
At any time point, it simply copies its previous value.
Because it can be used flexibly, we make it a subtype of VariableTimeModel.
"""
struct StaticModel{I,O} <: VariableTimeModel{I,Tuple{O},O}
    sf::SFunc{I,O}
end

function make_initial(m::StaticModel, time) 
    return m.sf
end

function make_transition(::StaticModel{I,O}, parenttimes, time) where {I,O} 
    return Det(Tuple{O}, O, x -> x[1])
end