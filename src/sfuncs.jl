module SFuncs

using Base: reinit_stdio
using ..MultiInterface

using ..Scruff
using ..Utils
using ..Operators
import ..Operators
import ..Operators: VectorOption, Support, SupportQuality
import ..Operators: InitialStats, AccumulateStats, ExpectedStats, MaximizeStats

macro impl(expr)
    return esc(MultiInterface.impl(__module__, __source__, expr, Operators))
end

include("sfuncs/dist/dist.jl")

include("sfuncs/score/score.jl")

include("sfuncs/util/extend.jl")

include("sfuncs/conddist/conddist.jl")

include("sfuncs/compound/compound.jl")

include("sfuncs/op_impls/bp_ops.jl")
include("sfuncs/op_impls/basic_ops.jl")

end
