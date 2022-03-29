export
    SyncLoopy,
    AsyncLoopy,
    CoherentLoopy

"""
    SyncLoopy(range_size = 10)    

A window filter that uses a synchronous window with LoopyBP with the given range size.
"""
SyncLoopy(range_size = 10) = WindowFilter(SyncWindow(), LoopyBP(range_size))

"""
    AsyncLoopy(range_size = 10, T = Float64)    

A window filter that uses an asynchronous window with LoopyBP with the given range size.
`T` represents the time type and must be the same as used in creation of the runtime.
"""
AsyncLoopy(range_size = 10, T = Float64)  = WindowFilter(AsyncWindow{T}(), LoopyBP(range_size))

"""
    CoherentLoopy(range_size = 10, T = Float64)    

A window filter that uses a coherent window with LoopyBP with the given range size.
`T` represents the time type and must be the same as used in creation of the runtime.
"""
CoherentLoopy(range_size = 10, T = Float64)  = WindowFilter(CoherentWindow{T}(), LoopyBP(range_size))

