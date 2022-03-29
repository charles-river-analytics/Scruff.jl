module Models

using ..Scruff
using ..Scruff.Operators
using ..Scruff.SFuncs

import ..Scruff: make_initial, make_transition

include("models/instantmodel.jl")
include("models/timelessinstantmodel.jl")
include("models/simplemodel.jl")
include("models/fixedtimemodel.jl")
include("models/timelessfixedtimemodel.jl")
include("models/homogeneousmodel.jl")
include("models/variabletimemodel.jl")
include("models/staticmodel.jl")

end
