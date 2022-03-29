export
    LSFI

"""
    function LSFI(query_vars; 
        increment = 10, start_size = increment, max_iterations = 100, start_depth = 1) 

A lazy inference algorithm that uses variable elimination at every step.

# Arguments
- `query_vars`: Variables that can be queried after each `refine` step
- `increment`: The increment to range size on every iteration
- `start_size`: The starting range size
- `max_iterations`: The maximum number of refinement steps
- `start_depth`: The depth of recursive expansion in the first iteration
"""
function LSFI(query_vars; 
        increment = 10, start_size = increment, max_iterations = 100, start_depth = 1) 
    # query_vars = [get_node(net, :out)]
    maker(size, depth) = VE(query_vars; depth = depth, bounds = true, range_size =size)
    return LazyInference(maker; increment = increment, start_size = start_size, 
        max_iterations = max_iterations, start_depth = start_depth)
end
