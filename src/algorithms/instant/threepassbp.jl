export
    ThreePassBP,
    # The export of three_pass_BP is kept for backward compatibility,
    # but new implementations should use ThreePassBP.
    three_pass_BP 

"""
    ThreePassBP

    An instant algorithm that runs three passes of belief propagation.
"""
struct ThreePassBP <: BP 
    default_range_size::Int
    ThreePassBP(drs = 10) = new(drs)
end

function debugparams(sf, range, prob, varname, fname, name)
    Dict(:type=>(output_type(sf) <: Real ? :cont : :discrete),
         :numBins=>min(10, length(range)),
         :range=>range,
         :prob=>prob,
         :varname=>varname,
         :fname=>fname,
         :name=>name)
end

function three_pass_BP(runtime::InstantRuntime)
    for node in get_nodes(get_network(runtime))
        remove_messages!(runtime, node, :pi_message)
        remove_messages!(runtime, node, :lambda_message)
    end
    run_bp(ThreePassBP(), runtime)
end

function run_bp(::ThreePassBP, runtime::InstantRuntime)
    network = get_network(runtime)
    variables = [v for v in topsort(get_initial_graph(network)) if v isa Variable]
    ranges = Dict{Symbol, Array{Any, 1}}()
    for node in get_nodes(network)
        inst = current_instance(runtime, node)
        ranges[node.name] = get_range(runtime, inst)
    end

    for var in variables
        for par in get_parents(network, var)
            if par isa Variable
                rng = ranges[par.name]
                initial_message = Cat(rng,  ones(length(rng)))
                set_message!(runtime, par, var, :pi_message, initial_message)
            end
        end
    end

    for var in variables
        _forwardstep(runtime, var, ranges, true)
    end

    for var in reverse(variables)
        _backstep(runtime, var, ranges)
    end

    for var in variables
        _forwardstep(runtime, var, ranges, false)
    end
end
