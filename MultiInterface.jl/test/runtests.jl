using MultiInterface
using Test

@testset "MultiInterface" begin
    include("basic.jl")
    # include("validation.jl")
    include("policy.jl")
    # include("wip.jl")
end
