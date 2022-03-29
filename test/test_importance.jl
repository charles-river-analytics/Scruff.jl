import Base.timedwait
import Base.isapprox
import PrettyPrint

using Test
using Scruff
using Scruff.Utils
using Scruff.RTUtils
using Scruff.SFuncs
using Scruff.Algorithms
import Scruff.Operators: cpdf

@testset "Importance" begin
    
    @testset "Sampling utilities" begin
        @testset "Probability" begin
            sample1 = Dict(:a => 1, :b => 2)
            sample2 = Dict(:a => 1, :b => 3)
            sample3 = Dict(:a => 2, :b => 4)
            lws = [-0.1, -0.2, -0.3]
            parts = Particles([sample1, sample2, sample3], lws)
            tot = sum([exp(x) for x in lws])
            p1 = exp(-0.1) + exp(-0.2)
            @test isapprox(probability(parts, s -> s[:a] == 1), p1 / tot)
        end

        @testset "Marginal" begin
            sample1 = Dict(:a => 1, :b => 2)
            sample2 = Dict(:a => 1, :b => 3)
            sample3 = Dict(:a => 2, :b => 4)
            lws = [-0.1, -0.2, -0.3]
            parts = Particles([sample1, sample2, sample3], lws)
            p1 = exp(-0.1) + exp(-0.2)
            p2 = exp(-0.3)
            tot = p1 + p2
            marg = marginal(parts, :a)
            @test isapprox(cpdf(marg, (), 1), p1 / tot)
            @test isapprox(cpdf(marg, (), 2), p2 / tot)
        end

        @testset "Effective sample size" begin
            @test isapprox(effective_sample_size([log(0.4)]), 1.0)
            @test isapprox(effective_sample_size([log(0.4), log(0.2)]), (0.6 * 0.6 / (0.4 * 0.4 + 0.2 * 0.2)))
        end

        @testset "Normalizing weights" begin
            @test isapprox(normalize_weights([log(0.1), log(0.3)]), [log(0.25), log(0.75)])
        end

        @testset "Probability of evidence" begin
            @test isapprox(log_prob_evidence([log(0.1), log(0.3)]), log((0.1+0.3)/2))
        end

        @testset "Resampling" begin
            samples = fill(Dict(:a => false), 1000)
            append!(samples, fill(Dict(:a => true), 1000))
            weights = fill(log(0.1), 1000)
            append!(weights, fill(log(0.9), 1000))
            ps = resample(Particles(samples, weights))
            @test all(x -> x == 0.0, ps.log_weights)
            @test isapprox(count(x -> x[:a], ps.samples) / 2000, 0.9, atol = 0.05)
        end
    end
    
    @testset "Rejection" begin
        
        @testset "Basic" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            alg = Rejection(1000)
            infer(alg, runtime)
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            @test isapprox(probability(alg, runtime, i1, x -> x == :a), 0.1; atol = 0.05)
            @test isapprox(probability(alg, runtime, i1, x -> x == :b), 0.9; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, x -> x == 1), 
                0.1 * 0.2 + 0.9 * 0.3; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, x -> x == 2), 
                0.1 * 0.8 + 0.9 * 0.7; atol = 0.05)
            m1 = marginal(alg, runtime, i1)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m1, (), :a), 0.1; atol = 0.05)
            @test isapprox(cpdf(m1, (), :b), 0.9; atol = 0.05)
            @test isapprox(cpdf(m2, (), :1), 0.1 * 0.2 + 0.9 * 0.3; atol = 0.05)
            @test isapprox(cpdf(m2, (), :2), 0.1 * 0.8 + 0.9 * 0.7; atol = 0.05)
        end

        @testset "With placeholder" begin
            p1 = Placeholder{Symbol}(:p1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v2], VariableGraph(v2 => [p1]), Placeholder[p1])
            runtime = Runtime(net)
            default_initializer(runtime, 10, Dict(p1.name => Cat([:a,:b], [0.1, 0.9])))
            alg = Rejection(1000)
            infer(alg, runtime)
            i2 = current_instance(runtime, v2)
            @test isapprox(probability(alg, runtime, i2, 1), 
                0.1 * 0.2 + 0.9 * 0.3; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 2), 
                0.1 * 0.8 + 0.9 * 0.7; atol = 0.05)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m2, (), 1), 0.1 * 0.2 + 0.9 * 0.3; atol = 0.05)
            @test isapprox(cpdf(m2, (), 2), 0.1 * 0.8 + 0.9 * 0.7; atol = 0.05)
        end

        @testset "With hard evidence" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            alg = Rejection(1000)
            infer(alg, runtime, Dict{Symbol, Score}(:v2 => HardScore(2)))
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            p1 = 0.1 * 0.8
            p2 = 0.9 * 0.7
            z = p1 + p2
            @test isapprox(probability(alg, runtime, i1, :a), p1 / z; atol = 0.05)
            @test isapprox(probability(alg, runtime, i1, :b), p2 / z; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 1), 0.0; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 2), 1.0; atol = 0.05)
            m1 = marginal(alg, runtime, i1)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m1, (), :a), p1 / z; atol = 0.05)
            @test isapprox(cpdf(m1, (), :b), p2 / z; atol = 0.05)
            @test isapprox(cpdf(m2, (), 1), 0.0; atol = 0.05)
            @test isapprox(cpdf(m2, (), 2), 1.0; atol = 0.05)
        end

        @testset "With soft evidence" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            alg = Rejection(1000)
            infer(alg, runtime, Dict{Symbol, Score}(:v2 => SoftScore(Dict(1 => 0.6, 2 => 0.4))))
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            pa = 0.1 * (0.2 * 0.6 + 0.8 * 0.4)
            pb = 0.9 * (0.3 * 0.6 + 0.7 * 0.4)
            z1 = pa + pb
            # soft evidence is treated like a lambda message on a variable,
            # so the prior also comes into play, unlike an intervention
            p1 = (0.1 * 0.2 + 0.9 * 0.3) * 0.6
            p2 = (0.1 * 0.8 + 0.9 * 0.7) * 0.4
            z2 = p1 + p2
            @assert isapprox(z1, z2)
            @test isapprox(probability(alg, runtime, i1, :a), pa / z1; atol = 0.05)
            @test isapprox(probability(alg, runtime, i1, :b), pb / z1; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 1), p1 / z2; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 2), p2 / z2; atol = 0.05)
            m1 = marginal(alg, runtime, i1)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m1, (), :a), pa / z1; atol = 0.05)
            @test isapprox(cpdf(m1, (), :b), pb / z1; atol = 0.05)
            @test isapprox(cpdf(m2, (), :1), p1 / z2; atol = 0.05)
            @test isapprox(cpdf(m2, (), :2), p2 / z2; atol = 0.05)
        end

        @testset "With intervention" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            alg = Rejection(1000)
            infer(alg, runtime, Dict{Symbol, Score}(), Dict{Symbol, Dist}(:v2 => Constant(2)))
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            @test isapprox(probability(alg, runtime, i1, :a), 0.1; atol = 0.05)
            @test isapprox(probability(alg, runtime, i1, :b), 0.9; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 1), 0.0; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 2), 1.0; atol = 0.05)
            m1 = marginal(alg, runtime, i1)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m1, (), :a), 0.1; atol = 0.05)
            @test isapprox(cpdf(m1, (), :b), 0.9; atol = 0.05)
            @test isapprox(cpdf(m2, (), :1), 0.0; atol = 0.05)
            @test isapprox(cpdf(m2, (), :2), 1.0; atol = 0.05)
        end
       
    end

    @testset "LW" begin
        @testset "Basic" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            alg = LW(1000)
            infer(alg, runtime)
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            @test isapprox(probability(alg, runtime, i1, x -> x == :a), 0.1; atol = 0.05)
            @test isapprox(probability(alg, runtime, i1, x -> x == :b), 0.9; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, x -> x == 1), 
                0.1 * 0.2 + 0.9 * 0.3; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, x -> x == 2), 
                0.1 * 0.8 + 0.9 * 0.7; atol = 0.05)
            m1 = marginal(alg, runtime, i1)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m1, (), :a), 0.1; atol = 0.05)
            @test isapprox(cpdf(m1, (), :b), 0.9; atol = 0.05)
            @test isapprox(cpdf(m2, (), :1), 0.1 * 0.2 + 0.9 * 0.3; atol = 0.05)
            @test isapprox(cpdf(m2, (), :2), 0.1 * 0.8 + 0.9 * 0.7; atol = 0.05)
        end

        @testset "With placeholder" begin
            p1 = Placeholder{Symbol}(:p1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v2], VariableGraph(v2 => [p1]), Placeholder[p1])
            runtime = Runtime(net)
            default_initializer(runtime, 10, Dict(p1.name => Cat([:a,:b], [0.1, 0.9])))
            alg = LW(1000)
            infer(alg, runtime)
            i2 = current_instance(runtime, v2)
            @test isapprox(probability(alg, runtime, i2, 1), 
                0.1 * 0.2 + 0.9 * 0.3; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 2), 
                0.1 * 0.8 + 0.9 * 0.7; atol = 0.05)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m2, (), 1), 0.1 * 0.2 + 0.9 * 0.3; atol = 0.05)
            @test isapprox(cpdf(m2, (), 2), 0.1 * 0.8 + 0.9 * 0.7; atol = 0.05)
        end

        @testset "With hard evidence" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            alg = LW(1000)
            infer(alg, runtime, Dict{Symbol, Score}(:v2 => HardScore(2)))
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            p1 = 0.1 * 0.8
            p2 = 0.9 * 0.7
            z = p1 + p2
            @test isapprox(probability(alg, runtime, i1, :a), p1 / z; atol = 0.05)
            @test isapprox(probability(alg, runtime, i1, :b), p2 / z; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 1), 0.0; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 2), 1.0; atol = 0.05)
            m1 = marginal(alg, runtime, i1)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m1, (), :a), p1 / z; atol = 0.05)
            @test isapprox(cpdf(m1, (), :b), p2 / z; atol = 0.05)
            @test isapprox(cpdf(m2, (), :1), 0.0; atol = 0.05)
            @test isapprox(cpdf(m2, (), :2), 1.0; atol = 0.05)
        end

        @testset "With soft evidence" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            alg = LW(1000)
            infer(alg, runtime, Dict{Symbol, Score}(:v2 => SoftScore(Dict(1 => 0.6, 2 => 0.4))))
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            pa = 0.1 * (0.2 * 0.6 + 0.8 * 0.4)
            pb = 0.9 * (0.3 * 0.6 + 0.7 * 0.4)
            z1 = pa + pb
            # soft evidence is treated like a lambda message on a variable,
            # so the prior also comes into play, unlike an intervention
            p1 = (0.1 * 0.2 + 0.9 * 0.3) * 0.6
            p2 = (0.1 * 0.8 + 0.9 * 0.7) * 0.4
            z2 = p1 + p2
            @assert isapprox(z1, z2)
            @test isapprox(probability(alg, runtime, i1, :a), pa / z1; atol = 0.05)
            @test isapprox(probability(alg, runtime, i1, :b), pb / z1; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 1), p1 / z2; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 2), p2 / z2; atol = 0.05)
            m1 = marginal(alg, runtime, i1)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m1, (), :a), pa / z1; atol = 0.05)
            @test isapprox(cpdf(m1, (), :b), pb / z1; atol = 0.05)
            @test isapprox(cpdf(m2, (), :1), p1 / z2; atol = 0.05)
            @test isapprox(cpdf(m2, (), :2), p2 / z2; atol = 0.05)
        end

        @testset "With intervention" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            alg = LW(1000)
            infer(alg, runtime, Dict{Symbol, Score}(), Dict{Symbol, Dist}(:v2 => Constant(2)))
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            @test isapprox(probability(alg, runtime, i1, :a), 0.1; atol = 0.05)
            @test isapprox(probability(alg, runtime, i1, :b), 0.9; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 1), 0.0; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 2), 1.0; atol = 0.05)
            m1 = marginal(alg, runtime, i1)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m1, (), :a), 0.1; atol = 0.05)
            @test isapprox(cpdf(m1, (), :b), 0.9; atol = 0.05)
            @test isapprox(cpdf(m2, (), :1), 0.0; atol = 0.05)
            @test isapprox(cpdf(m2, (), :2), 1.0; atol = 0.05)
        end
    end

    @testset "Custom proposal" begin
        prop::Dict{Symbol,SFunc} = Dict(:v2 => 
                    DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7])))
        alg = Importance(make_custom_proposal(prop), 1000)

        @testset "Basic" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            infer(alg, runtime)
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            @test isapprox(probability(alg, runtime, i1, x -> x == :a), 0.1; atol = 0.05)
            @test isapprox(probability(alg, runtime, i1, x -> x == :b), 0.9; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, x -> x == 1), 
                0.1 * 0.2 + 0.9 * 0.3; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, x -> x == 2), 
                0.1 * 0.8 + 0.9 * 0.7; atol = 0.05)
            m1 = marginal(alg, runtime, i1)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m1, (), :a), 0.1; atol = 0.05)
            @test isapprox(cpdf(m1, (), :b), 0.9; atol = 0.05)
            @test isapprox(cpdf(m2, (), :1), 0.1 * 0.2 + 0.9 * 0.3; atol = 0.05)
            @test isapprox(cpdf(m2, (), :2), 0.1 * 0.8 + 0.9 * 0.7; atol = 0.05)
        end

        @testset "With placeholder" begin
            p1 = Placeholder{Symbol}(:p1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v2], VariableGraph(v2 => [p1]), Placeholder[p1])
            runtime = Runtime(net)
            default_initializer(runtime, 10, Dict(p1.name => Cat([:a,:b], [0.1, 0.9])))
            infer(alg, runtime)
            i2 = current_instance(runtime, v2)
            @test isapprox(probability(alg, runtime, i2, 1), 
                0.1 * 0.2 + 0.9 * 0.3; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 2), 
                0.1 * 0.8 + 0.9 * 0.7; atol = 0.05)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m2, (), 1), 0.1 * 0.2 + 0.9 * 0.3; atol = 0.05)
            @test isapprox(cpdf(m2, (), 2), 0.1 * 0.8 + 0.9 * 0.7; atol = 0.05)
        end

        @testset "With hard evidence" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            infer(alg, runtime, Dict{Symbol, Score}(:v2 => HardScore(2)))
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            p1 = 0.1 * 0.8
            p2 = 0.9 * 0.7
            z = p1 + p2
            @test isapprox(probability(alg, runtime, i1, :a), p1 / z; atol = 0.05)
            @test isapprox(probability(alg, runtime, i1, :b), p2 / z; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 1), 0.0; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 2), 1.0; atol = 0.05)
            m1 = marginal(alg, runtime, i1)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m1, (), :a), p1 / z; atol = 0.05)
            @test isapprox(cpdf(m1, (), :b), p2 / z; atol = 0.05)
            @test isapprox(cpdf(m2, (), :1), 0.0; atol = 0.05)
            @test isapprox(cpdf(m2, (), :2), 1.0; atol = 0.05)
        end

        @testset "With soft evidence" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            infer(alg, runtime, Dict{Symbol, Score}(:v2 => SoftScore(Dict(1 => 0.6, 2 => 0.4))))
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            pa = 0.1 * (0.2 * 0.6 + 0.8 * 0.4)
            pb = 0.9 * (0.3 * 0.6 + 0.7 * 0.4)
            z1 = pa + pb
            # soft evidence is treated like a lambda message on a variable,
            # so the prior also comes into play, unlike an intervention
            p1 = (0.1 * 0.2 + 0.9 * 0.3) * 0.6
            p2 = (0.1 * 0.8 + 0.9 * 0.7) * 0.4
            z2 = p1 + p2
            @assert isapprox(z1, z2)
            @test isapprox(probability(alg, runtime, i1, :a), pa / z1; atol = 0.05)
            @test isapprox(probability(alg, runtime, i1, :b), pb / z1; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 1), p1 / z2; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 2), p2 / z2; atol = 0.05)
            m1 = marginal(alg, runtime, i1)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m1, (), :a), pa / z1; atol = 0.05)
            @test isapprox(cpdf(m1, (), :b), pb / z1; atol = 0.05)
            @test isapprox(cpdf(m2, (), :1), p1 / z2; atol = 0.05)
            @test isapprox(cpdf(m2, (), :2), p2 / z2; atol = 0.05)
        end

        @testset "With intervention" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            infer(alg, runtime, Dict{Symbol, Score}(), Dict{Symbol, Dist}(:v2 => Constant(2)))
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            @test isapprox(probability(alg, runtime, i1, :a), 0.1; atol = 0.05)
            @test isapprox(probability(alg, runtime, i1, :b), 0.9; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 1), 0.0; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 2), 1.0; atol = 0.05)
            m1 = marginal(alg, runtime, i1)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m1, (), :a), 0.1; atol = 0.05)
            @test isapprox(cpdf(m1, (), :b), 0.9; atol = 0.05)
            @test isapprox(cpdf(m2, (), :1), 0.0; atol = 0.05)
            @test isapprox(cpdf(m2, (), :2), 1.0; atol = 0.05)
        end
    end

    @testset "Iterative sampling" begin
        
        @testset "Basic" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            alg = IterativeSampler(LW(1000))
            prepare(alg, runtime)
            particles = get_state(runtime, :particles)
            @test length(particles.samples) == 0
            refine(alg, runtime)
            particles = get_state(runtime, :particles)
            @test length(particles.samples) == 1000
            refine(alg, runtime)
            particles = get_state(runtime, :particles)
            @test length(particles.samples) == 2000
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            @test isapprox(probability(alg, runtime, i1, x -> x == :a), 0.1; atol = 0.05)
            @test isapprox(probability(alg, runtime, i1, x -> x == :b), 0.9; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, x -> x == 1), 
                0.1 * 0.2 + 0.9 * 0.3; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, x -> x == 2), 
                0.1 * 0.8 + 0.9 * 0.7; atol = 0.05)
            m1 = marginal(alg, runtime, i1)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m1, (), :a), 0.1; atol = 0.05)
            @test isapprox(cpdf(m1, (), :b), 0.9; atol = 0.05)
            @test isapprox(cpdf(m2, (), :1), 0.1 * 0.2 + 0.9 * 0.3; atol = 0.05)
            @test isapprox(cpdf(m2, (), :2), 0.1 * 0.8 + 0.9 * 0.7; atol = 0.05)
        end

        @testset "With placeholder" begin
            p1 = Placeholder{Symbol}(:p1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v2], VariableGraph(v2 => [p1]), Placeholder[p1])
            runtime = Runtime(net)
            default_initializer(runtime, 10, Dict(p1.name => Cat([:a,:b], [0.1, 0.9])))
            alg = IterativeSampler(LW(1000))
            prepare(alg, runtime)
            particles = get_state(runtime, :particles)
            @test length(particles.samples) == 0
            refine(alg, runtime)
            particles = get_state(runtime, :particles)
            @test length(particles.samples) == 1000
            refine(alg, runtime)
            particles = get_state(runtime, :particles)
            @test length(particles.samples) == 2000
            i2 = current_instance(runtime, v2)
            @test isapprox(probability(alg, runtime, i2, 1), 
                0.1 * 0.2 + 0.9 * 0.3; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 2), 
                0.1 * 0.8 + 0.9 * 0.7; atol = 0.05)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m2, (), 1), 0.1 * 0.2 + 0.9 * 0.3; atol = 0.05)
            @test isapprox(cpdf(m2, (), 2), 0.1 * 0.8 + 0.9 * 0.7; atol = 0.05)
        end

        @testset "With hard evidence" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            alg = IterativeSampler(LW(1000))
            prepare(alg, runtime, Dict(:v2 => HardScore(2)))
            particles = get_state(runtime, :particles)
            @test length(particles.samples) == 0
            refine(alg, runtime)
            particles = get_state(runtime, :particles)
            @test length(particles.samples) == 1000
            refine(alg, runtime)
            particles = get_state(runtime, :particles)
            @test length(particles.samples) == 2000
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            p1 = 0.1 * 0.8
            p2 = 0.9 * 0.7
            z = p1 + p2
            @test isapprox(probability(alg, runtime, i1, :a), p1 / z; atol = 0.05)
            @test isapprox(probability(alg, runtime, i1, :b), p2 / z; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 1), 0.0; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 2), 1.0; atol = 0.05)
            m1 = marginal(alg, runtime, i1)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m1, (), :a), p1 / z; atol = 0.05)
            @test isapprox(cpdf(m1, (), :b), p2 / z; atol = 0.05)
            @test isapprox(cpdf(m2, (), :1), 0.0; atol = 0.05)
            @test isapprox(cpdf(m2, (), :2), 1.0; atol = 0.05)
        end

        @testset "With soft evidence" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            alg = IterativeSampler(LW(1000))
            prepare(alg, runtime, Dict(:v2 => SoftScore(Dict(1 => 0.6, 2 => 0.4))))
            particles = get_state(runtime, :particles)
            @test length(particles.samples) == 0
            refine(alg, runtime)
            particles = get_state(runtime, :particles)
            @test length(particles.samples) == 1000
            refine(alg, runtime)
            particles = get_state(runtime, :particles)
            @test length(particles.samples) == 2000
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            pa = 0.1 * (0.2 * 0.6 + 0.8 * 0.4)
            pb = 0.9 * (0.3 * 0.6 + 0.7 * 0.4)
            z1 = pa + pb
            # soft evidence is treated like a lambda message on a variable,
            # so the prior also comes into play, unlike an intervention
            p1 = (0.1 * 0.2 + 0.9 * 0.3) * 0.6
            p2 = (0.1 * 0.8 + 0.9 * 0.7) * 0.4
            z2 = p1 + p2
            @assert isapprox(z1, z2)
            @test isapprox(probability(alg, runtime, i1, :a), pa / z1; atol = 0.05)
            @test isapprox(probability(alg, runtime, i1, :b), pb / z1; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 1), p1 / z2; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 2), p2 / z2; atol = 0.05)
            m1 = marginal(alg, runtime, i1)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m1, (), :a), pa / z1; atol = 0.05)
            @test isapprox(cpdf(m1, (), :b), pb / z1; atol = 0.05)
            @test isapprox(cpdf(m2, (), :1), p1 / z2; atol = 0.05)
            @test isapprox(cpdf(m2, (), :2), p2 / z2; atol = 0.05)
        end

        @testset "With intervention" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            alg = IterativeSampler(LW(1000))
            prepare(alg, runtime, Dict{Symbol, Score}(), Dict(:v2 => Constant(2)))
            particles = get_state(runtime, :particles)
            @test length(particles.samples) == 0
            refine(alg, runtime)
            particles = get_state(runtime, :particles)
            @test length(particles.samples) == 1000
            refine(alg, runtime)
            particles = get_state(runtime, :particles)
            @test length(particles.samples) == 2000
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            @test isapprox(probability(alg, runtime, i1, :a), 0.1; atol = 0.05)
            @test isapprox(probability(alg, runtime, i1, :b), 0.9; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 1), 0.0; atol = 0.05)
            @test isapprox(probability(alg, runtime, i2, 2), 1.0; atol = 0.05)
            m1 = marginal(alg, runtime, i1)
            m2 = marginal(alg, runtime, i2)
            @test isapprox(cpdf(m1, (), :a), 0.1; atol = 0.05)
            @test isapprox(cpdf(m1, (), :b), 0.9; atol = 0.05)
            @test isapprox(cpdf(m2, (), :1), 0.0; atol = 0.05)
            @test isapprox(cpdf(m2, (), :2), 1.0; atol = 0.05)
        end
        
    end
    
end




