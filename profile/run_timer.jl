using BenchmarkTools
using Scruff.SFuncs
using Scruff.Models
using Scruff.Utils
using Scruff.RTUtils
using Scruff.Algorithms

include("make_instant.jl")
include("make_filter.jl")
include("run_instant.jl")
include("run_filter.jl")
include("instant_bp.jl")
include("instant_importance.jl")
include("dynamic_bp.jl")
include("dynamic_pf.jl")

println("TYPE OPTIMIZED\n")

println("Instant network using BP")
# Arguments are number of values in range of each node
println("- Instant network (2, 3, 2, 2, 2)")
@btime instant_bp(2, 3, 2, 2, 2)
println("- Instant network (6, 9, 6, 6, 6)")
@btime instant_bp(6, 9, 6, 6, 6)
println("- Instant network (18, 27, 18, 18, 18)")
@btime instant_bp(18, 27, 18, 18, 18)

println("Instant network using Importance")
println("- Instant network (2,3,2,2,2) with 100 samples")
@btime instant_importance(2, 3, 2, 2, 2, 100)
println("- Instant network (2,3,2,2,2) with 1000 samples")
@btime instant_importance(2, 3, 2, 2, 2, 1000)
println("- Instant network (2,3,2,2,2) with 10000 samples")
@btime instant_importance(2, 3, 2, 2, 2, 10000)
println("- Instant network (6,9,6,6,6) with 1000 samples")
@btime instant_importance(6, 9, 6, 6, 6, 1000)
println("- Instant network (18,27,18,18,18) with 1000 samples")
@btime instant_importance(18, 27, 18, 18, 18, 1000)

println("Dynamic network using BP")
println("- Dynamic network (2,2) over 10 time steps")
@btime dynamic_bp(2, 2, 10)
println("- Dynamic network (2,2) over 100 time steps")
@btime dynamic_bp(2, 2, 100)
println("- Dynamic network (4,4) over 10 time steps")
@btime dynamic_bp(4, 4, 10)
println("- Dynamic network (8,8) over 10 time steps")
@btime dynamic_bp(8, 8, 10)

println("Dynamic network using Particle Filter")
println("- Dynamic network (2,2) over 10 time steps with 100 particles")
@btime dynamic_pf(2, 2, 100, 10)
println("- Dynamic network (2,2) over 10 time steps with 1000 particles")
@btime dynamic_pf(2, 2, 1000, 10)
println("- Dynamic network (2,2) over 10 time steps with 10000 particles")
@btime dynamic_pf(2, 2, 10000, 10)
println("- Dynamic network (2,2) over 100 time steps with 1000 particles")
@btime dynamic_pf(2, 2, 1000, 100)
println("- Dynamic network (4,4) over 10 time steps with 1000 particles")
@btime dynamic_pf(4, 4, 1000, 10)
println("- Dynamic network (8,8) over 10 time steps with 1000 particles")
@btime dynamic_pf(8, 8, 1000, 10)

