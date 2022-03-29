using Logging

import Cassette

export 
    trace_algorithm, 
    time_algorithm, 
    trace_runtime

# these are the methods in this file
function toskip(nm)
    startswith(string(nm), "#module_functions") ||
    :trace_algorithm == nm ||
    :time_algorithm == nm ||
    :trace_runtime == nm
end

function module_functions(modname; private=true)
    list = Tuple{Module, Symbol}[]
    for nm in names(modname,all=private)
        toskip(nm) || typeof(eval(:($modname.$nm))) <: Function && push!(list,(modname,nm))
    end
    return list
end

function getparams(sig)
    while isa(sig, UnionAll)
        sig = sig.body
    end
    sig.parameters
end

function runtime_functions(modname; private=true)
    set = Set{Tuple{Module, Symbol}}()
    for nm in names(modname,all=private)
        toskip(nm) && continue
        evald = eval(:($modname.$nm))
        if typeof(evald) <: Function
            for m in methods(evald)
                for tp in getparams(m.sig)
                    tp = isa(tp, UnionAll) ? tp.body : isa(tp, TypeVar) ? tp.ub : tp
                    isa(tp, DataType) && tp <: Runtime && push!(set, (modname, nm))
                end
            end
        end
    end
    return set
end


filteronmethod(filter, m) = filter(string(typeof(m).name.mt.name))
filteronmethod(::Nothing, m) = true

filteronargs(filter, args...) = filter(args...)
filteronargs(::Nothing, args...) = true

function dofilter(ctx, f, args...)
    ctx.metadata === nothing || (filteronmethod(ctx.metadata.methodfilter, f) 
        && filteronargs(ctx.metadata.argsfilter, args...))
end

struct Meta 
    methodfilter
    argsfilter
end

function meta(fnamefilter, argfilter) 
    fnamefilter === nothing && argfilter === nothing ? nothing : Meta(fnamefilter, argfilter)
end

# construct Scruff function list before we start, since all the methods
# in this file will also be in the Scruff module
fns = Tuple{Module, Symbol}[]
fns = append!(fns, module_functions(Scruff))

rtfns = Tuple{Module, Symbol}[]
rtfns = append!(rtfns, runtime_functions(Scruff))


Cassette.@context DoTracing
Cassette.@context DoTiming
Cassette.@context DoTraceRuntime

rtfns = append!(rtfns, runtime_functions(Scruff.Operators))
rtfns = append!(rtfns, runtime_functions(Scruff.Algorithms))
rtfns = append!(rtfns, runtime_functions(Scruff.SFuncs))
rtfns = append!(rtfns, runtime_functions(Scruff.Utils))
rtfns = append!(rtfns, runtime_functions(Scruff.RTUtils))
rtfns = unique(p->p[2], rtfns) # unique function name

# create Cassette prehooks for Tracing Runtime

for (m,op) in rtfns
    @eval begin
        function Cassette.prehook(ctx::DoTraceRuntime, f::typeof($m.$op), args...)
            if dofilter(ctx, f, args...)
                @info "$(f)" called_by=stacktrace()[3] args=args
            end
        end 
    end
end

fns = append!(fns, module_functions(Scruff.Operators))
fns = append!(fns, module_functions(Scruff.Algorithms))
fns = append!(fns, module_functions(Scruff.SFuncs))
fns = append!(fns, module_functions(Scruff.RTUtils))
fns = unique(p->p[2], fns) # unique function name

# create Cassette prehooks for Tracing
for (m,op) in fns
    @eval begin
        function Cassette.prehook(ctx::DoTracing, f::typeof($m.$op), args...)
            if dofilter(ctx, f, args...)
                @info "$(f)" called_by=stacktrace()[3] args=args
            end
        end 
    end
end

# create Cassette overdub for Timing
for (m,op) in fns
    @eval begin
        function Cassette.overdub(ctx::DoTiming, f::typeof($m.$op), args...) 
            val, t, bytes, gctime, memallocs = @timed Cassette.recurse(ctx, f, args...)
            if dofilter(ctx, f, args...)
                @info "$f" time=string(t*1000," ms") bytes=bytes gctime=gctime
            end 
            return val
        end
    end
end

"""
    trace_runtime(alg::Function, args...; fnamefilter=nothing, argfilter=nothing)

Wrap `Runtime` methods (called with its `args`) in a trace.  All Scruff functions
that have `Runtime` as a parameter and their calling parameter values will 
be logged using the `@info` macro using Julia's logging system.  

Filtering can be done on either the name of the method or the args list; 
both parameters are functions that take a `x::String` or a `x...`,
respectively, as input and return `true` if the method is to be logged.

The output looks like a series of
```julia
┌ Info: get_variable
│   called_by = get_definition at runtime.jl:93 [inlined]
└   args = (Instance{Int64}(Variable{Int64,DiscreteCPD{1}}(:x5, DiscreteCPD{1}([[0.35, 0.65], [0.45, 0.55]])), 0),)
```
"""
function trace_runtime(alg::Function, args...; fnamefilter=nothing, argfilter=nothing, kwargs...)
    m = meta(fnamefilter, argfilter)
    Cassette.overdub(Scruff.DoTraceRuntime(metadata=m), ()->alg(args...; kwargs...))
end

"""
    trace_algorithm(alg::Function, args...; fnamefilter=nothing, argfilter=nothing)

Wrap an algorithm (called with its `args`) in a trace.  All Scruff functions and 
their calling parameter values will be logged using the `@info` macro using 
Julia's logging system.  

Filtering can be done on either the name of the method or the args list; 
both parameters are functions that take a `x::String` or a `x...`,
respectively, as input and return `true` if the method is to be logged.
    
The output looks like a series of
```julia
┌ Info: outgoing_pis
│   called_by = operate at runtime.jl:387 [inlined]
└   args = (DiscreteCPD{1}([[0.35, 0.65], [0.45, 0.55]]), [0.38, 0.6200000000000001], Any[]) 
```
"""
function trace_algorithm(alg::Function, args...; fnamefilter=nothing, argfilter=nothing, kwargs...)
    m = meta(fnamefilter, argfilter)
    Cassette.overdub(Scruff.DoTracing(metadata=m), ()->alg(args...; kwargs...))
end

"""
    time_algorithm(alg::Function, args...; fnamefilter=nothing, argfilter=nothing)

Wrap an algorithm (callect with its `args`) such that every Scruff function
is called with the `@timed` macro.  The output is sent to the `@info` macro
using Julia's logging system.  

Filtering can be done on either the name of the method or the args list; 
both parameters are functions that take a `x::String` or a `x...`, 
respectively, as input and return `true` if the method is to be logged.
    
The output looks like a series of
```julia
┌ Info: outgoing_pis
│   time = "0.0068000000000000005 ms"
│   bytes = 144
└   gctime = 0.0
```
"""
function time_algorithm(alg::Function, args...; fnamefilter=nothing, argfilter=nothing, kwargs...) 
    ctx = Cassette.disablehooks(Scruff.DoTiming(metadata=meta(fnamefilter, argfilter)))
    val, t, bytes, gctime, memallocs =
        @timed Cassette.overdub(ctx, alg, args...; kwargs...)
    @info "$alg" time=string(t*1000," ms") bytes=bytes gctime=gctime 
    return val
end
