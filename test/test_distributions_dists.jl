using Test

using Scruff
using Scruff.SFuncs
using Scruff.Operators

import Distributions

@testset "Discrete Distributions.jl" begin
    d = Distributions.Categorical([0.4, 0.3, 0.3])
    sf = DistributionsSF(d)
    samples = [sample(sf, ()) for _ in 1:10]
    sf_mean = expectation(sf, ())
end
