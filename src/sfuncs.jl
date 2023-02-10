module SFuncs

using Base: reinit_stdio
using ..MultiInterface

using ...Scruff
using ..Utils
import ..Operators
Operators.@import_operators()
import ..Operators:  __OptVec, Support, SupportQuality

include("sfuncs/dist/dist.jl")
include("sfuncs/dist/cat.jl")
include("sfuncs/dist/constant.jl")
include("sfuncs/dist/flip.jl")
include("sfuncs/dist/normal.jl")

include("sfuncs/score/score.jl")
include("sfuncs/score/hardscore.jl")
include("sfuncs/score/softscore.jl")
include("sfuncs/score/logscore.jl")
include("sfuncs/score/functionalscore.jl")
include("sfuncs/score/normalscore.jl")
include("sfuncs/score/parzen.jl")

include("sfuncs/util/extend.jl")

include("sfuncs/conddist/conditional.jl")
include("sfuncs/conddist/det.jl")
include("sfuncs/conddist/invertible.jl")
include("sfuncs/conddist/table.jl")
include("sfuncs/conddist/discretecpt.jl")
include("sfuncs/conddist/lineargaussian.jl")
include("sfuncs/conddist/CLG.jl")
include("sfuncs/conddist/separable.jl")
include("sfuncs/conddist/switch.jl")

include("sfuncs/compound/generate.jl")
include("sfuncs/compound/apply.jl")
include("sfuncs/compound/chain.jl")
include("sfuncs/compound/mixture.jl")
include("sfuncs/compound/network.jl")
include("sfuncs/compound/serial.jl")
include("sfuncs/compound/expander.jl")

include("sfuncs/op_impls/bp_ops.jl")
include("sfuncs/op_impls/basic_ops.jl")

end
