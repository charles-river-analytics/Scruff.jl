using Base.Threads

export
    em,
    accumulate_stats,
    bp_info_provider,
    converged,
    expected_stats,
    initial_stats,
    maximize_stats

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
Parameters associated with the variables used by the algorithm are found
in the runtime, associated with instances using the :params key.
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
initial network and extract initial parameters from the variables into
the runtime under the :params key.
Models should provide a get_params operation which gets the initial
parameters.
A default initializer is provided in initializer.jl, which does only this.
The EM algorithm will take care of maintaining the parameters in the runtime
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
    discount_factor The degree to which old statistics should be discounted
        when considering new examples. For batch mode, this will be zero.
        For online mode, this will be close to 1. discount_factor is actually
        a function that takes the runtime as an argument. In many cases it will
        be constant, but this allows it to change with runtime factors.
===================#

function em(network, data ;
            data_batcher = (n,x) -> x, 
            algorithm = three_pass_BP,
            initializer = default_initializer,
            info_provider = bp_info_provider, 
            showprogress = false,
            max_iterations = -1, 
            epsilon = 0.0001, 
            discount_factor = r -> 0.0,
            min_iterations = 1,
            validationset = nothing, # used instead of convergence test
            convergebymax = false)
    iteration = 0
    newparams = init_params(network)
    conv = false
    stats = nothing
    if !isnothing(validationset)
        validationscore = score(network, newparams, validationset, algorithm, initializer)
    end
    while (max_iterations < 0 || iteration < max_iterations) && !(conv && iteration >= min_iterations)
        # we have to deepcopy the params since its values are a pointer
        # into the model parameters, which are updated, in many cases, in place
        if showprogress
            println("Iteration ", iteration)
        end
        oldparams = deepcopy(newparams)
        batch = data_batcher(network, data)
        (stats, newparams) =
            em_iteration(network, batch, algorithm, initializer,
                         info_provider, discount_factor, stats,
                         oldparams, showprogress)
        if isnothing(validationset)
            conv = converged(oldparams, newparams, epsilon, convergebymax)
        else
            newscore = score(network, newparams, validationset, algorithm, initializer) 
            conv = newscore <= validationscore
            if conv
                newparams = oldparams # roll back
            end
            validationscore = newscore
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
    return ((conv, iteration), newparams)
end

function em_iteration(network, batch, algorithm, initializer,
                      info_provider, discount_factor, oldstats, oldparams, showprogress)
    vars = get_variables(network)

    # 1: Initialize the statistics
    if showprogress
        println("Initializing statistics")
    end
    newstats = Dict{Symbol, Any}()
    newparams = Dict{Symbol, Any}()
    newruntime = Runtime(network)

    # 2: For each example, accumulate statistics
    # varstats = Array{Any}(undef, (length(batch), length(vars)))
    alock = SpinLock()
    Threads.@threads for i = 1:length(batch)
    # for i = 1:length(batch)
        runtime = deepcopy(newruntime)
        tvars = get_variables(runtime)
        for var in tvars
            set_params!(make_sfunc(var.model), oldparams[var.name])
        end
        # we have to call the initializer after we set the params, since we create instances
        # with the underlying parameter values
        initializer(runtime)

        example = batch[i]
        if showprogress
            println("Accumulating statistics for example ", i)
        end
        # 2.1: Prepare the evidence
        for var in tvars
            inst = current_instance(runtime, var)
            delete_evidence!(runtime, inst)
            if var.name in keys(example)
                post_evidence!(runtime, inst, example[var.name])
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
        for j in 1:length(tvars)
            var = tvars[j]
            inst = current_instance(runtime, var)
            sf = get_sfunc(inst)
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
            lock(alock)
            if var.name in keys(newstats)
                newstats[var.name] = accumulate_stats(sf, newstats[var.name], sts)
            else
                newstats[var.name] = sts
            end
            unlock(alock)
        end
    end

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

    # 4 choose the maximizing parameter values and store them in the runtime
    # To implement parameter sharing, we invoke the maximize_stats once per model
    # for all the variables defined by the model.
    # Currently, parameter sharing doesn't work
    if showprogress
        println("Choosing maximizing parameters")
    end
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
    
    return (newstats, newparams)
end

function init_params(network)
    result = Dict{Symbol, Any}()
    for var in get_variables(network)
        result[var.name] = get_params(make_sfunc(var.model, 0, get_dt(var.model)))
    end
    return result
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

diff(x :: Number, y :: Number) = abs(x-y)

function diff(xs :: Dict, ys :: Dict)
    total = 0.0
    for (k,v) in keys(xs)
        total += diff(xs[k], ys[k])
    end
    return total
end

function diff(xs, ys)
    total = 0.0
    for i = 1:length(xs)
        total += diff(xs[i], ys[i])
    end
    return total
end

numparams(x :: Number) = 1

numparams(xs) = sum(map(numparams, xs))

function converged(oldp, newp, eps::Float64, convergebymax::Bool = false)
    # This is written so that if new variables are added to the network
    # by the inference algorithm, we don't have convergence.
    # However, if keys are deleted, we can, because all existing keys
    # might have converged.
    totaldiff = 0.0
    num = 0
    for k in keys(newp)
        if !(k in keys(oldp))
            return false
        end
        if isnothing(newp[k])
            break
        end
        if !(length(newp[k]) == length(oldp[k]))
            return false
        end
        d = diff(newp[k], oldp[k])
        n = numparams(newp[k])
        if convergebymax && d / n >= eps
            return false
        end
        totaldiff += d
        num += n
    end
    return totaldiff / num < eps
end

function bp_info_provider(runtime, inst)
    net = get_network(runtime)
    var = get_node(inst)
    pars = get_parents(net, var)
    parent_πs :: Array{Array{Any, 1}} =
        collect_messages(runtime, pars, var, :pi_message)
    λ = get_value(runtime, inst, :lambda)
    cpd = get_params(inst.sf)
    # cpd = get_value(runtime, inst, :params)
    return (parent_πs, λ, cpd)
end

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