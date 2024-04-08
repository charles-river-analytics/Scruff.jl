export
    _complete,
    depth,
    expander_complete,
    expansions,
    expanded,
    expansion,
    subruntime,
    clear_analysis!,
    ensure_expansion_state!,
    expand!,
    expander_probs

function expander_complete(runtime :: Runtime, v :: Variable, parent_ranges)
    for p in parent_ranges[1]
        if !expanded(runtime, v, p) return false end
        subnet = expansion(runtime, v, p)
        subrun = subruntime(runtime, v, subnet)
        if !_complete(subrun) return false end
    end
    return true
end

# This method _only_ works if the model being operated upon contains an
# operated called 'support_quality'
function _complete(runtime :: Runtime)
    net = runtime.network
    order = topsort(get_initial_graph(net))
    for var in order
        if !has_instance(runtime, var)
            return false
        end
        inst = current_instance(runtime, var)
        sf = get_sfunc(inst)

        if !has_range(runtime, inst)
            return false
        end
        parranges = parent_ranges(runtime, var)
        imp = get_imp(MultiInterface.get_policy(), Support, sf, parranges, 0, output_type(sf)[])
        # if is_fixed(var.model) && isa(sf, Expander)
        if isa(sf, Expander)
            if !expander_complete(runtime, var, Tuple(parranges))
                return false
            end
        elseif !(support_quality(imp, sf, parranges) == :CompleteSupport)
            return false
        end
    end
    return true
end

function expansions(runtime::InstantRuntime, var::Variable) :: Dict
    exp :: Expander = make_initial(var.model)
    mod = var.model
    has_state(runtime, :subnets) || return Dict()
    exps = get_state(runtime, :subnets)
    mod in keys(exps) || return Dict() 
    return exps[mod]
end

function expanded(runtime::Runtime, var::Variable, arg)
    return arg in keys(expansions(runtime, var))
end

function expansion(runtime :: Runtime, var :: Variable, arg) :: Network
    return expansions(runtime, var)[arg]
end

function subruntime(runtime::Runtime, var::Variable, net)
    subruns = get_state(runtime, :subruntimes)
    return subruns[net]
end

function depth(runtime::Runtime) :: Int
    if has_state(runtime, :depth)
        get_state(runtime, :depth)
    else
        0
    end
end

function ensure_expansion_state!(runtime::Runtime)
    has_state(runtime, :subnets) || set_state!(runtime, :subnets, Dict())
    has_state(runtime, :subruntimes) || set_state!(runtime, :subruntimes, Dict())
end

# Returns the expanded subnet and a flag indicating whether it was newly expanded
function expand!(runtime::InstantRuntime, var::Variable, arg)
    ensure_expansion_state!(runtime)
    exp :: Expander = make_initial(var.model)
    mod = var.model
    exps = get_state(runtime, :subnets)
    subruns = get_state(runtime, :subruntimes)
    if !(mod in keys(exps))
        exps[mod] = Dict()
    elseif arg in keys(exps[mod])
        return (exps[mod][arg], false)
    end

    subnet = apply(exp, arg)
    subrun = Runtime(subnet)
    # push all parent variables to subruntime
    set_state!(subrun, :parent_env, get_env(runtime))
    # we associate a subrun with the network, in case the same network is
    # produced for multiple arguments, so we avoid repeating computation for all
    #  the arguments that produce the same network
    subruns[subnet] = subrun
    ensure_all!(subrun)
    exps[mod][arg] = subnet
    return (subnet, true)
end

# analysis is 'per network', so we have to make sure we track per-network
network_analysis = Dict{Network, Dict{Symbol, Any}}()

clear_analysis!() = empty!(network_analysis)

function has_analysis(net,sym)
    haskey(network_analysis, net) && haskey(network_analysis[net], sym)
end

get_analysis(net,sym) = network_analysis[net][sym]

function add_analysis!(net,sym,val) 
    get!(network_analysis, net, Dict{Symbol, Any}())[sym] = val
end

function expander_probs(runtime::Runtime, fn::Function, v::Variable, depth::Int) 
    net = get_network(runtime)
    par = get_parents(net, v)[1]
    parinst = current_instance(runtime, par)
    inst = current_instance(runtime, v)
    if has_range(runtime, parinst)
        parrange = get_range(runtime, parinst, depth)
    else
        parrange = []
    end
    if has_range(runtime, inst)
        range = get_range(runtime, inst, depth)
    else
        range = []
    end
    lowers = Float64[]
    uppers = Float64[]

    function make_unexpanded_row()
        for i in 1:length(range)
            push!(lowers, 0)
            push!(uppers, 1)
        end
    end

    for p in parrange
        if depth < 2 || !expanded(runtime, v, p)
            make_unexpanded_row()
        else
            subnet = expansion(runtime, v, p)
            subrun = subruntime(runtime, v, subnet)
            output = get_node(subnet, :out)
            need_to_compute = true
            if has_analysis(subnet, :depthsolution)
                (saved_depth, saved_range, solution) =
                    get_analysis(subnet, :depthsolution)
                # To be able to reuse the solution to define this node's CPD,
                # it must have the required depth and range
                if saved_depth >= depth && saved_range == range
                    need_to_compute = false
                    (lfact, ufact) = solution
                end
            end
            if need_to_compute
                order = topsort(get_initial_graph(subnet))
                ((lfact, ufact), _) = fn(subrun, order, [output], depth - 1)
                add_analysis!(subnet, :depthsolution,
                              (depth, range, (lfact, ufact)))
            end
            lsum = sum(lfact.entries)
            output_range = get_range(subrun, output, depth - 1)
            inds = indexin(range, output_range)
            for i in 1:length(range)
                if isempty(inds) || isnothing(inds[i])
                    # If this value is not in the output range, then it
                    # might not be in the range of the output, so has
                    # probability 0, or it might be in the range but not
                    # expanded yet. But it's probability cannot be more than
                    # 1 - lsum
                    push!(lowers, 0)
                    push!(uppers, 1 - lsum)
                else
                    if isempty(lfact.entries)
                        push!(lowers, 0)
                    else
                        push!(lowers, lfact.entries[inds[i]])
                    end
                    if isempty(ufact.entries)
                        push!(uppers, 0)
                    else
                        push!(uppers, ufact.entries[inds[i]])
                    end
                end
            end
        end
    end
    return (lowers, uppers)
end
