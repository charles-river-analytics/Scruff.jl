export 
    Parameterized


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

make_initial(m :: Parameterized, t) = set_params!(make_initial(base_model(m), t), config_spec(m))

make_transition(m :: Parameterized, parent_times, time) = set_params!(make_transition(base_model(m), parent_times, time), config_spec(m))

initial_stats(m :: Parameterized) = initial_stats(sf(m))

accumulate_stats(m :: Parameterized, s1, s2) = accumulate_stats(sf(m), s1, s2)

maximize_stats(m :: ConfigurableModel, s) = maximize_stats(sf(m), s)

