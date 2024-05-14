using BenchmarkTools
using Scruff.SFuncs
using Scruff.RTUtils
using Scruff.Algorithms

include("run_instant.jl")
include("instant_bp.jl")
include("instant_importance.jl")
include("dynamic_bp.jl")
include("dynamic_pf.jl")

println("UNOPTIMIZED")

println("Instant network using BP")
@btime instant_bp()

println("Instant network using importance")
@btime instant_importance()

println("Dynamic network using BP")
@btime dynamic_bp()

println("Dynamic network using importance")
@btime dynamic_pf()

