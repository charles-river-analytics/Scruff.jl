# File test/runtests.jl
module ScruffTest

using Test

@testset "ScruffTests" begin
    include("test_core.jl")
    include("test_sfuncs.jl")
    include("test_score.jl")
    include("test_utils.jl")
    include("test_ve.jl")
    include("test_lsfi.jl")
    include("test_bp.jl")
    include("test_importance.jl")
    include("test_net.jl")
    include("test_filter.jl")
    # These don't test anything, but we want to make sure any changes haven't broken the examples
    redirect_stdout(devnull) do
        include("../docs/examples/novelty_example.jl")
        include("../docs/examples/novelty_lazy.jl")
        include("../docs/examples/novelty_filtering.jl")
        include("../docs/examples/rembrandt_example.jl")
        include("../docs/examples/soccer_example.jl")
    end
end

end
