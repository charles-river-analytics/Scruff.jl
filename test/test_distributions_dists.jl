using Test

using Scruff
using Scruff.SFuncs
using Scruff.Operators

import Distributions

@testset "Discrete Distributions.jl" begin
    d = Distributions.Categorical([0.4, 0.3, 0.3])
    sf = DistributionsSF(d)
    N = 100
    samples = [sample(sf, ()) for _ in 1:N]
    sf_mean = expectation(sf, ())
    @test isapprox(sf_mean, sum(samples) / N; atol=0.1)
end

@testset "Continuous Distributions.jl" begin
    d = Distributions.Normal()
    sf = DistributionsSF(d)
    samples = [sample(sf, ()) for _ in 1:10]
    @test isapprox(expectation(sf, ()), 0.0)
    @test isapprox(variance(sf, ()), 1.0)
    sf2 = sumsfs((sf, sf))
    @test isapprox(variance(sf2, ()), 2.0)
end
