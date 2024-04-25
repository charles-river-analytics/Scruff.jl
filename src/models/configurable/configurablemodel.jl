export 
    ConfigurableModel,
    set_config_spec!,
    get_config_spec,
    converged


#=
A ConfigurableModel contains an underlying model, and allows the SFuncs returned by the underlying model
to be configured using a configuration specification. Those SFuncs should support the following operators:
    configure(sf, config_spec), which returns an sfunc (could be fresh or a mutation of sf)
    converged(sf, old_config_spec, new_config_spec) :: Boolean

The type parameters of ConfigurableModel are as follows:
    I : Parents of initial sfunc
    J : Parents of transition sfunc
    O : Outputs of both initial and transition sfuncs
    C : Configuration specification
    S : Stats

An instance of ConfigurableModel must be provided with the following methods:
    base_Model(m)
    configSpec(m)
    initial_stats(m) :: S
    accumulate_stats(m, current_stats :: s, new_stats) :: S
        which uses the current stats to produce updated stats
    maximize_stats!(m, stats :: S) :: C
    
Typical usage will start by first setting current_stats to initial_stats, and then repeatedly 
- computing update_info
- calling add_stats using current_stats and update_info
- calling maximize_stats! after all the stats have been added to set config_spec   
=#

abstract type ConfigurableModel{I,J,O,C,S} <: Model{I,J,O} end

make_initial(m :: ConfigurableModel, t) = configure(make_initial(m.base_model, t), m.config_spec)

make_transition(m :: ConfigurableModel, parent_times, time) = configure(make_transition(m.base_model, parent_times, time), m.config_spec)

function set_config_spec!(m :: ConfigurableModel, spec :: C) where C
    m.config_spec = spec
end

get_config_spec(m :: ConfigurableModel) = m.config_spec

# Eventually we want to create a dynamic version, but for now it's just static
# function converged(m :: ConfigurableModel{I, J, O, C, S}, old_spec :: C, new_spec :: C,
#                     parent_times, time) :: Boolean where {I, J, O, C, S}:: Boolean
#     init = make_initial(m.base_model, time)
#     trans = make_transition(m.base_model, parent_times, time)
#     converged(init, old_spec, new_spec) && converged(trans, old_spec, new_spec)
# end
function converged(m :: ConfigurableModel{I, J, O, C, S}, old_spec :: C, new_spec :: C) :: Boolean where {I, J, C, O, S} 
    init = make_initial(m.base_model, time)
    converged(init, old_spec, new_spec)
end

