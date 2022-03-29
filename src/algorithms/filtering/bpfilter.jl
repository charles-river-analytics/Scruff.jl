export
    SyncBP,
    AsyncBP,
    CoherentBP

"""
    SyncBP(range_size = 10)    

A window filter that uses a synchronous window with ThreePassBP with the given range size.
"""
SyncBP(range_size = 10) = WindowFilter(SyncWindow(), ThreePassBP(range_size))

"""
    AsyncBP(range_size = 10, T = Float64)    

A window filter that uses an asynchronous window with ThreePassBP with the given range size.
`T` represents the time type and must be the same as used in creation of the runtime.
"""
AsyncBP(range_size = 10, T = Float64) = WindowFilter(AsyncWindow{T}(), ThreePassBP(range_size))

"""
    CoherentBP(range_size = 10, T = Float64)    

A window filter that uses a coherent window with ThreePassBP with the given range size.
`T` represents the time type and must be the same as used in creation of the runtime.
"""
CoherentBP(range_size = 10, T = Float64) = WindowFilter(CoherentWindow{T}(), ThreePassBP(range_size))

