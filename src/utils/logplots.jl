"""
logplots.jl contains code to help gain insight into belief propagation.
It creates a Logger called `BPLogger` that plots the results, for
a single variable, of the following methods in the three pass BP algorithm:

* compute_pi
* compute_lambda
* compute_bel
* incoming_pi
* outgoing_pis

To use `BPLogger`, first install `Plots` and use `pyplot()`.  Then,

```
julia> include("src/utils/logplots.jl")
histogram_plot_discrete (generic function with 1 method)

julia> logger = BPLogger(:x1) # :x1 is the name of the variable being plotted
BPLogger(:x1)

julia> with_logger(logger) do
         include("mybpcall.jl") # mybpcall.jl contains a call to threepassbp()
       end
```

"""

export BPLogger

import Logging: handle_message, shouldlog, min_enabled_level

using Plots
using Logging

struct BPLogger <: AbstractLogger
    varname::Vector{Symbol}
end
BPLogger(varname::Symbol...) = BPLogger(collect(varname))

function handle_message(
        logger::BPLogger,
        level,
        message,
        _module,
        group,
        id,
        filepath,
        line; kwargs...)

    if isempty(kwargs) || !in(:type, keys(kwargs))
        with_logger(global_logger()) do
            @logmsg level message kwargs...
        end
        return
    end

    d = Dict(kwargs...)
    type = get(d, :type, :cont)
    if (type != :cont && type != :discrete)
        write_missing("Value of :type must be either :cont or :discrete, not $(type)",
            level, _module, filepath, line; kwargs...)
        return
    end

    if (type == :cont && !in(:numBins, keys(kwargs)))
        write_missing("Continuous missing a required parameter [:numBins]",
            level, _module, filepath, line; kwargs...)
        return
    end

    if !issubset([:range,:prob,:varname,:fname,:name], keys(kwargs))
        write_missing("Missing a required parameter [:range,:prob,:varname,:fname,:name]",
            level, _module, filepath, line; kwargs...)
        return
    end

    if (d[:varname] in logger.varname)
        if (type == :cont)
            histogram_plot(
                d[:range],
                d[:prob],
                d[:numBins],
                d[:varname],
                d[:fname],
                d[:name])
        else
            histogram_plot_discrete(
                d[:range],
                d[:prob],
                d[:varname],
                d[:fname],
                d[:name])
        end
    end
end

function write_missing(msg, level, _module, filepath, line; kwargs...)
    buf = IOBuffer()
    iob = IOContext(buf, stderr)
    levelstr = level == Logging.Warn ? "Warning" : string(level)
    msglines = split(chomp(string(msg)::String), '\n')
    println(iob, "┌ ", levelstr, ": ", msglines[1])
    for i in 2:length(msglines)
        println(iob, "│ ", msglines[i])
    end
    for (key, val) in kwargs
        println(iob, "│   ", key, " = ", val)
    end
    println(iob, "└ @ ", _module, " ", filepath, ":", line)
    write(stderr, take!(buf))
    nothing
end

function shouldlog(logger::BPLogger, level, _module, group, id)
    true
    # group == :threepassbp
end

function min_enabled_level(logger::BPLogger)
    Logging.Debug
end


"""
    histogram_plot(range, prob, numBins::Int64, name::String)

A utility function that, if the Plots module is loaded, returns a histogram
bar graph; otherwise it returns a string with the values that would have
been a histogram

"""
function histogram_plot(range, prob, numBins::Int64, varname::Symbol, fname, name::String)
    @debug("histogram_plot", range=range,
                prob=prob,numBins=numBins,varname=varname,
                fname=fname,name=name)
    if isempty(prob)
        prob = zeros(length(range))
    end
    range_min = minimum(range)
    range_max = maximum(range)
    if(range_min==range_max)
        range_max = range_min+1
    end
    numBins = max(min(length(range), numBins),1)
    stepsize = (range_max - range_min)/ numBins
    samplespace = range_min : stepsize : range_max
    bins = zeros(numBins)
    for (i,lb) in enumerate(samplespace)
        curr_bin_min = samplespace[i]
        curr_bin_max = samplespace[i] + stepsize
        idx = findall(x -> curr_bin_min <= range[x] < curr_bin_max, 1:length(range))
        if(!isempty(idx))
            if(i==length(samplespace)) # last elements
                bins[i-1] += sum(prob[idx])
            else
                bins[i] = sum(prob[idx])
            end
        end
    end
    x_min = samplespace[1]
    x_max = samplespace[end] + stepsize

    @debug "bin_centers=$(collect(samplespace)[1:numBins] .+ (stepsize/2)), bins=$bins"
    gui(bar(samplespace[1:numBins].+(stepsize/2), bins; reuse=false,title="$(varname).$(fname)", label=name, xlims=(x_min-2, x_max+2), hover = samplespace[1:numBins].+(stepsize/2), legend=true))
end


function histogram_plot_discrete(range, prob:: Array{Float64,1}, varname, fname, name::String)
    gui(bar(string.(range),prob; reuse=false,title="$(varname).$(fname)",label=name, hover = string.(range), legend=true))
end
