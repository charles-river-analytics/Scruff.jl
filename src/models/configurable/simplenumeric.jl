export
    SimpleNumeric

struct SimpleNumeric{I,O,C,S} <: SimpleConfigurable{I,O,C,S}
    base :: SimpleModel{I, O}
    epsilon :: Float64
    converge_by_max :: Bool
    SimpleNumeric{I,O,C,S}(sf) where {I,O,C,S} = new(SimpleModel(sf), 0.01, false)
    SimpleNumeric{I,O,C,S}(sf, eps) where {I,O,C,S} = new(SimpleModel(sf), eps, false)
    SimpleNumeric{I,O,C,S}(sf, eps, cbm) where {I,O,C,S} = new(SimpleModel(sf), eps, cbm)
end

base_model(m :: SimpleNumeric) = m.base

converged(m :: SimpleNumeric, old_spec, new_spec) = converged_numeric(old_spec, new_spec, m.epsilon, m.converge_by_max)




