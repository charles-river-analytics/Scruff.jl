export
    SimpleNumeric

struct SimpleNumeric{I,O,C,S} <: Parameterized{I,O,C,S}
    base :: SimpleModel{I, O}
    config_spec :: C
    epsilon :: Float64
    SimpleNumeric{I,O,C,S}(sf) where {I,O,C,S} = new(SimpleModel(sf), get_params(sf), 0.01)
    SimpleNumeric{I,O,C,S}(sf, eps) where {I,O,C,S} = new(SimpleModel(sf), get_params(sf), eps)
end

base_model(m :: SimpleNumeric) = m.base

config_spec(m :: SimpleNumeric) = m.config_spec

converged(m :: SimpleNumeric, old_spec, new_spec) = converged_numeric(old_spec, new_spec, m.epsilon)




