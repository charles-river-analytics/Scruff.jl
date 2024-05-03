export 
    Parameterized

import Scruff: make_initial, make_transition
import Scruff.Operators: initial_stats, accumulate_stats, maximize_stats

# Parameterized is a ConfigurableModel in which the base model is a SimpleModel over an sfunc, and the config_spec has the same datatype as explicitly defined 
# parameters of the sfunc. The sfunc must have get_params, set_params!, initial_stats, accumulate_stats,
# and maximize_stats methods are defined.
# This very common case is made easy with this code.
# A Parameterized must have an method defined
#
# base_model(m) :: SimpleModel{I, O}
#
# All the other methods of a ConfigurableModel will then be defined automatically.

abstract type Parameterized{I,O,C,S} <: ConfigurableModel{I,I,O,C,S} end

function make_initial(m :: Parameterized, t) 
    mod = base_model(m)
    cs = get_config_spec(m)
    sf = make_initial(mod, t)
    set_params!(sf, cs)
    set_params!(make_initial(base_model(m), t), get_config_spec(m))
end

make_transition(m :: Parameterized, parent_times, time) = set_params!(make_transition(base_model(m), parent_times, time), get_config_spec(m))

initial_stats(m :: Parameterized) = initial_stats(get_sf(m))

accumulate_stats(m :: Parameterized, s1, s2) = accumulate_stats(get_sf(m), s1, s2)

function maximize_stats(m :: Parameterized{I, O, S}, stats) where {I,O,S}
    conf_sp = maximize_stats(get_sf(m), stats)
    set_config_spec!(m, conf_sp)
end

