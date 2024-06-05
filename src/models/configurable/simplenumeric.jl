export
    SimpleNumeric

mutable struct SimpleNumeric{I,O,S} <: Parameterized{I,O,S,S}
    base :: SimpleModel{I, O}
    config_spec :: S
    epsilon :: Float64
    SimpleNumeric{I,O,S}(sf) where {I,O,S} = new(SimpleModel(sf), get_params(sf), 0.01)
    SimpleNumeric{I,O,S}(sf, eps) where {I,O,S} = new(SimpleModel(sf), get_params(sf), eps)
end

get_sf(m :: SimpleNumeric) = m.base.sf

base_model(m :: SimpleNumeric) = m.base

get_config_spec(m :: SimpleNumeric) = m.config_spec

function set_config_spec!(m :: SimpleNumeric{I, O, S}, cs :: S) where {I,O,S}
    m.config_spec = cs
    m
end

converged(m :: SimpleNumeric, old_spec, new_spec) = converged_numeric(old_spec, new_spec, m.epsilon)



