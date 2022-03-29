export 
    Filter,
    init_filter,
    filter_step

"""
    abstract type Filter <: Algorithm end

General type of filtering algorithms.

Must implement init_filter and filter_step methods.
"""
abstract type Filter <: Algorithm end

"""
    init_filter!(::Filter, ::DynamicRuntime)

An interface for intializing the filter for a dynamic runtime.
"""
function init_filter!(::Filter, ::DynamicRuntime) end

"""
    filter_step(filter::Filter, runtime::Runtime, variables::Vector{Variable}, time::T, evidence::Dict{Symbol, Score})

Run one step of the filter by instantiating the given variables at the given time and passing in the given evidence.
"""
function filter_step(::Filter, ::DynamicRuntime{T}, ::Vector{Variable}, ::T, ::Dict{Symbol, Score}) where T end

