module Algorithms

import Distributions

using ..Scruff
using ..Utils
using ..RTUtils
using ..Models
using ..SFuncs
using ..Operators

include("algorithms/algorithm.jl")
include("algorithms/query.jl")
include("algorithms/sample_utils.jl")
include("algorithms/instant/instantalgorithm.jl")
include("algorithms/instant/ve.jl")
include("algorithms/instant/bp.jl")
include("algorithms/instant/threepassbp.jl")
include("algorithms/instant/loopybp.jl")
include("algorithms/instant/importance.jl")
include("algorithms/iterative/iterativealgorithm.jl")
include("algorithms/iterative/lazyinference.jl")
include("algorithms/iterative/lsfi.jl")
include("algorithms/iterative/iterativesampler.jl")
include("algorithms/filtering/filter.jl")
include("algorithms/filtering/windowutils.jl")
include("algorithms/filtering/windowcreator.jl")
include("algorithms/filtering/windowfilter.jl")
include("algorithms/filtering/bpfilter.jl")
include("algorithms/filtering/loopyfilter.jl")
include("algorithms/filtering/particlefilter.jl")
end
