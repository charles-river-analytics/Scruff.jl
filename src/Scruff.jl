module Scruff
include("../MultiInterface.jl/src/MultiInterface.jl")  ## Scruff.MultiInterface module

include("utils.jl")            ## Scruff.Utils module
include("core.jl")             ## Scruff module
include("operators.jl")        ## Scruff.Operators module
include("sfuncs.jl")           ## Scruff.SFuncs module
include("models.jl")           ## Scruff.Models module 
include("runtime.jl")          ## Scruff module
include("runtime_utils.jl")    ## Scruff.RTUtils module
include("algorithms.jl")       ## Scruff.Algorithms module
include("logger.jl")           ## Scruff module

end
