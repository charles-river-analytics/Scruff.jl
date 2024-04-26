using Base.Threads

export
    em,
    bp_info_provider

using Scruff
using Scruff.Models

#=========================================
The EM algorithm is written generically.
It takes a runtime and a source of data.
To make the algorithm concrete, it requires an inference algorithm,
as well as a data_batcher to convert the source of data into training examples
for each iteration.

The format of the data is not specified.
A data_batcher is used to produce the required data to use at each iteration.
The result of the data_batcher should be a one-dimensional array of examples,
where each example is a Dict mappimg variable names to evidence.
The evidence format is specified in evidence_ops.
It can either be hard evidence, consisting of a value of the variable,
or soft evidence, consisting of a distribution over values.
The default data_batcher assumes the data already has the right format and
simply returns the data as a batch.
This implementation allows for both batch and online versions of EM.
For online EM, use a non-zero discount_factor (see below).
Furthermore, the data_batcher takes the runtime as an argument.
This enables it, for example, to change the network as new objects are
encountered, enabling EM with flexible structures.
This also supports dynamic and asynchronous operation.

The algorithm can be any algorithm that computes beliefs over its variables
that can be used to compute statistics. The algorithm takes only the runtime
as argument and uses the evidence associated with instances in the runtime.
Parameters associated with the variables are generalized to any configuration specification
that can change an sfunc.
The configuration specifications used by the algorithm are found
in the runtime, associated with instances using the :config_spec key.
The configuration specifications are used by ConfigurableModels.
In EM, ConfigurableModels and other kinds of models are treated differently.
Only ConfigurableModels are learned and have statistics computed.
The models defining the variables in the network must support the operations:
    initial_stats, which produces initial statistics as the basis for
        accumulation
    expected_stats, which produces the expected sufficient statistics for a
        single training example, using information computed by the algorithm
    accumulate_stats, which adds statistics across examples
    maximize_stats, which produces new parameter values from the statistics
The default algorithm is three pass BP.

Along with the algorithm, an initializer is required that does whatever work
is necessary for the algorithm before the first iteration.
At the minimum, the initializer should instantiate all variables in the
initial network and extract initial spec from the configurable variables into
the runtime under the :config_spec key.
At the minimum, get_config should be called on the configurable model to initialize 
the value of :config_spec.
A default initializer is provided in initializer.jl, which does only this.
The EM algorithm will take care of maintaining the specs in the runtime
through the iterations.

In order to accommodate different algorithms, which may use different data
structures to compute statistics, em uses an info_provider, which takes
a runtime and an instance and returns the information to pass to the
expected_stats operation. Therefore, for all models, the expected_stats
operation takes a single argument, the information produced by the
info_provider.

em returns a flag indicating whether or not the parameters have
converged at termination.

In addition to the previous arguments, em takes the following configuration
parameters:
    max_iterations The maximum number of iterations to run before terminating.
        If this number is negative (the default), there is no maximum.
    epsilon The maximum difference allowed between parameter values for
        convergence to be considered achieved
    # discount_factor The degree to which old statistics should be discounted
    #     when considering new examples. For batch mode, this will be zero.
    #     For online mode, this will be close to 1. discount_factor is actually
    #     a function that takes the runtime as an argument. In many cases it will
    #     be constant, but this allows it to change with runtime factors.
===================#

function bp_info_provider(runtime, inst)
    net = get_network(runtime)
    var = get_node(inst)

    pars = get_parents(net, var)
    parentpis :: Array{SFunc} =
        collect_messages(runtime, pars, var, :pi_message)
    childlam = get_value(runtime, inst, :lambda)
    # cpd = get_params(inst.sf)
    # cpd = get_value(runtime, inst, :params)
    pars = get_parents(net, var)
    parranges = [get_range(runtime, current_instance(runtime, p)) for p in pars]
    parranges = Tuple(parranges)
    range = get_range(runtime, inst)
    # sts = operate(runtime, inst, expected_stats, parranges, parentpis, childlam)
    sts = expected_stats(inst.sf, range, parranges, Tuple(parentpis), childlam)
    sts = normalize(sts) # We normalize here so we can apply normalization uniformly,
                        # whatever the sfunc of the variable. 
                        # This is a very important point in the design! 
                        # We don't have to normalize the individual sfuncs' expected_stats.
    return sts
    # return (parent_πs, λ, cpd)
end


function init_config_spec(network)
    result = Dict{Symbol, Any}()
    for var in get_variables(network)
        if var.model isa ConfigurableModel
            result[var.name] = get_config_spec(var.model)
        end
    end
    return result
end

function em(network, data ;
            data_batcher = (n,x) -> x, 
            algorithm = three_pass_BP,
            initializer = default_initializer,
            info_provider = bp_info_provider, 
            showprogress = false,
            max_iterations = -1, 
            min_iterations = 1)
    iteration = 0
    new_config_specs = init_config_spec(network)
    conv = false
    stats = nothing
    while (max_iterations < 0 || iteration < max_iterations) && !(conv && iteration >= min_iterations)
        # We have to deepcopy the config specs since their values are complex
        # and are updated, in many cases, in place.
        if showprogress
            println("Iteration ", iteration)
        end
        old_config_specs = deepcopy(new_config_specs)
        batch = data_batcher(network, data)
        println("POINT 0")
        iteration_result = em_iteration(network, batch, algorithm, initializer,
                         info_provider,
                         old_config_specs, showprogress)
        println("POINT 1.5")
        println("iteration_result = ", iteration_result)
        (stats, new_config_specs) = iteration_result
        println("POINT 2")
        newscore = score(network, new_config_specs, validationset, algorithm, initializer) 
        println("POINT 3")
        conv = newscore <= validationscore
        if conv
            new_config_specs = old_config_specs # roll back
        end
        iteration += 1
    end
    # At the end of the algorithm, the runtime will store the most recent
    # parameter values.
    # We return a flag indicating whether the algorithm converged
    # within the given iterations
    if showprogress
        if conv
            println("EM converged after ", iteration, " iterations")
        else
            println("EM did not converge; terminating after ", iteration, " iterations")
        end
    end
    return ((conv, iteration), new_config_specs)
end

function em_iteration(network, batch, algorithm, initializer,
                      info_provider, old_config_specs, showprogress)
    vars = get_variables(network)
    config_vars = filter(v -> v.model isa ConfigurableModel, vars)

    # 1: Initialize the statistics
    if showprogress
        println("Initializing statistics")
    end
    new_stats = Dict{Symbol, Any}()
    for var in config_vars
        new_stats[var.name] = initial_stats(make_initial(var.model, 0))
    end
    new_config_specs = Dict{Symbol, Any}()
    newruntime = Runtime(network)

    # 2: For each example, accumulate statistics
    alock = SpinLock()
    Threads.@threads for i = 1:length(batch)
    for i = 1:length(batch)
        runtime = deepcopy(newruntime)
        initializer(runtime)
        # Need to get the variables again because we did a deep copy
        runvars = get_variables(get_network(runtime))
        config_vars = filter(v -> v.model isa ConfigurableModel, runvars)
    
    
        example = batch[i]
        if showprogress
            println("Accumulating statistics for example ", i)
        end
        # 2.1: Prepare the evidence
        for var in runvars
            inst = current_instance(runtime, var)
            delete_evidence!(runtime, inst)
            if var.name in keys(example)
                post_belief!(runtime, inst, example[var.name])
            end
        end

        # 2.2: Run the algorithm
        if showprogress
            println("Running the algorithm for example ", i)
        end
        algorithm(runtime)

        # 2.3: Use the algorithm results to compute statistics for this example
        if showprogress
            println("Computing statistics for example ", i)
        end
        for var in config_vars
            mod = var.model
            if mod isa ConfigurableModel
                inst = current_instance(runtime, var)
                sf = get_sfunc(inst)
                info = info_provider(runtime, inst)
                lock(alock)
                sts = accumulate_stats(sf, new_stats[var.name], info)
                println("sts = ", sts)
                new_stats[var.name] = sts
                unlock(alock)

                #=
                (parentpis, childlam, _) = info_provider(runtime, inst)
                pars = get_parents(get_network(runtime), var)
                parranges = [get_range(runtime, current_instance(runtime, p)) for p in pars]
                parranges = Tuple(parranges)
                range = get_range(runtime, inst)
                # sts = operate(runtime, inst, expected_stats, parranges, parentpis, childlam)
                sts = expected_stats(sf, range, parranges, Tuple(parentpis), childlam)
                sts = normalize(sts) # We normalize here so we can apply normalization uniformly,
                                    # whatever the sfunc of the variable. 
                                    # This is a very important point in the design! 
                                    # We don't have to normalize the individual sfuncs' expected_stats.
                                    
                    newstats[]
                if var.name in keys(newstats)
                    newstats[var.name] = accumulate_stats(sf, newstats[var.name], sts)
                else
                    newstats[var.name] = sts
                end
                unlock(alock)
                =#
            end
        end
    end

    #= This doesn't make sense in the generalized algorithm
    # 3 blend in the old statistics
    if showprogress
        println("Blending in old statistics")
    end
    if !isnothing(oldstats) # will be nothing on first iteration
        d = discount_factor(newruntime)
        for var in vars
            if var.name in keys(oldstats)
                if !isnothing(oldstats[var.name]) 
                    ss = mult_through(oldstats[var.name], d)
                    tt = add_through(newstats[var.name], ss)
                    newstats[var.name] = tt
                end
            end
        end
    end
    =#

    # 4 choose the maximizing parameter values and store them in the runtime
    # To implement parameter sharing, we invoke the maximize_stats once per model
    # for all the variables defined by the model.
    # Currently, parameter sharing doesn't work
    if showprogress
        println("Choosing maximizing parameters")
    end
    println("new_config_specs = ", new_config_specs)
    for var in config_vars
        maximize_stats(make_initial(var.model, 0), new_stats[var.name])
        new_config_specs[var.name] = get_config_spec(var.model)
    end
    println("POINT 1")
    #=
    modelvars = Dict{Model, Array{Variable, 1}}()
    for var in vars
        m = var.model
        if false && isa(m, FixedModel)
            vs = get(modelvars, m, [])
            push!(vs, var)
            modelvars[m] = vs
        else
            # no parameter sharing
            sf = make_sfunc(var, newruntime) 
            newparams[var.name] = maximize_stats(sf, newstats[var.name])
        end
    end
    for (m, vs) in modelvars
        stats = initial_stats(m.sf)
        for v in vs
            stats = accumulate_stats(m.sf, stats, newstats[v.name])
        end
        modelparams = maximize_stats(m.sf, stats)
        for v in vs
            newparams[v.name] = modelparams
        end
    end
    =#
    println("RETURNING")
    println("new_stats = ", new_stats)
    println("new_config_specs = ", new_config_specs)
    
    (new_stats, new_config_specs)
end

# function close(x::Float64, y::Float64, eps::Float64)
#     return abs(x-y) < eps
# end

# function close(xs::Array, ys::Array, eps::Float64)
#     if length(xs) != length(ys) return false end
#     for i = 1:length(xs)
#         if !close(xs[i], ys[i], eps) return false end
#     end
#     return true
# end

# function close(xs::Tuple, ys::Tuple, eps::Float64)
#     if length(xs) != length(ys) return false end
#     for i = 1:length(xs)
#         if !close(xs[i], ys[i], eps) return false end
#     end
#     return true
# end

# function close(xs::Dict, ys::Dict, eps::Float64)
#     println("close: xs = ", xs, ", ys = ", ys)
#     ks = keys(xs)
#     ls = keys(ys)
#     if length(ks) != length(ls) 
#         println("Lengths unequal: returning false")
#         return false 
#     end
#     for k in ks
#         if !k in ls 
#             println("Value not present: returning false")
#             return false 
#         end
#         if !close(xs[k], ys[k], eps) 
#             println("Values not close: returning false")
#             return false 
#         end
#     end
#     println("Returning true")
#     return true
# end

#=
# Score the new parameters on the given validation set
function score(network, params, validationset, algorithm, initializer)
    result = 0.0
    alock = SpinLock()
    Threads.@threads for example in validationset
        runtime = Runtime(network)
        tvars = get_variables(runtime)
        for var in tvars
            set_params!(make_sfunc(var.model), params[var.name])
        end
        # we have to call the initializer after we set the params, since we create instances
        # with the underlying parameter values
        initializer(runtime)
        # Don't set the evidence. Run the algorithm and check the marginal probability of the evidence.
        algorithm(runtime)
        localscore = 0
        for var in tvars
            inst = current_instance(runtime, var)
            bel = get_belief(runtime, inst)
            if var.name in keys(example)
                prob = bel[example[var.name]]
                localscore += log(prob)
            end
        end
        lock(alock)
        result += localscore
        unlock(alock)
    end
    return result
end
=#

end