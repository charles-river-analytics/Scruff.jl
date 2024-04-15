module CoreTest

using Test
using Plots

import Distributions

using Scruff
using Scruff.Operators
using Scruff.MultiInterface
using Scruff.Models
using Scruff.SFuncs

import Scruff: make_initial, make_transition
import Scruff.Models: get_dt
import Scruff.Operators: Sample, sample, Logcpdf, logcpdf, Marginalize, marginalize, Expectation, expectation, Variance, variance

struct MyModel <: Model{Tuple{}, Tuple{Int}, Tuple{Int, Int}} end

function make_transition(m::MyModel, parenttimes, time)
    return Constant((parenttimes[1],time))
end

function make_initial(m::MyModel, time)
    return Cat([(1,0)], [1.0])
end

struct MyNormal <: SFunc{Tuple{}, Float32}
    mean::Float32
    var::Float32
end

@impl begin
    struct MyNormalExpectation end
    function expectation(sf::MyNormal, i::Tuple{})::Float32
        # Have access the sfunc as `sf` as well as `op_imp`
        return sf.mean
    end
end

@impl begin 
    struct MyNormalVariance end
    function variance(sf::MyNormal, i::Tuple{})::Float32
        return sf.var
    end
end

@impl begin
    struct MyNormalSample end
    function sample(sf::MyNormal, i::Tuple{})::Float32
        return rand(Distributions.Normal(sf.mean, sqrt(sf.var)))
    end
end

@impl begin
    struct MyNormalLogcpdf end
    function logcpdf(sf::MyNormal, i::Tuple{}, o::Float32)::AbstractFloat
        return Distributions.logpdf(Distributions.Normal(sf.mean, sqrt(sf.var)), o)
    end
end
struct MyCondMuNormal <: SFunc{Tuple{Float32}, Float32}
    # A conditional distribution for a Normal conditioned on mu with fixed var.
    # CondMuNormal_var(mu) = N(mu,var), essentially.
    var::Float32
end

@impl begin
    struct MyCondMuNormalSample end
    function sample(sf::MyCondMuNormal, x::Tuple{Float32})::Float32
        return rand(Distributions.Normal(x[1], sqrt(sf.var)))
    end
end

@impl begin
    struct MyCondMuNormalLogcpdf end
    function logcpdf(sf::MyCondMuNormal, i::Tuple{Float32}, o::Float32)::AbstractFloat
        return Distributions.logpdf(Distributions.Normal(i[1], sqrt(sf.var)), o)
    end
end

@impl begin
    struct MyCondMuNormalMarginalize end
    function marginalize(x::MyNormal, sf::MyCondMuNormal)::MyNormal
        mu = expectation(x, tuple())
        var = variance(x, tuple()) + sf.var
        return MyNormal(mu, var)
    end
end

@impl begin
    struct MyCondMuNormalExpectation end
    function expectation(sf::MyCondMuNormal, x::Tuple{Float32, Tuple{}})::Float32
        return x[1]
    end
end

@impl begin
    struct MyCondMuNormalVariance end
    function variance(sf::MyCondMuNormal, i::Tuple{Float32, Tuple{}})::Float32 
        return sf.var
    end
end

randomwalk = HomogeneousModel(MyNormal(0.0, 1.0), MyCondMuNormal(1.0), 2.0)

struct WienerProcess <: VariableTimeModel{Tuple{}, Tuple{Float32}, Float64}
    # A continuous limit of Gaussian random walk
    k::Float32 # "Rate" of random walk. Units of var/time
end
function make_transition(m::WienerProcess,parenttimes,time)
    var = m.k*abs(time-parenttimes[1])
    return MyCondMuNormal(var)
end
function make_initial(m::WienerProcess)
    return MyNormal(0,1)
end
wienerprocess = WienerProcess(0.1)

@testset "Core" begin
    @testset "instantiate!" begin
        @testset "instant network" begin
            @testset "correctly instantiates variables and placeholders" begin
                v = MyModel()(:v)
                p = Placeholder{Tuple{Int,Int}}(:p)
                net = InstantNetwork(Variable[v], VariableGraph(v => [p]), Placeholder[p])
                run = Runtime(net)
                ensure_all!(run)
                inst1 = current_instance(run, p)
                inst2 = current_instance(run, v)
                @test inst1 isa PlaceholderInstance
                @test inst2 isa VariableInstance
                @test get_sfunc(inst2) isa Cat
                # ensure has_timeoffset returns false
                @test !has_timeoffset(net, v, v)
            end
        end

        @testset "dynamic network" begin
            @testset "correctly calls make_initial or make_transition" begin
                v = MyModel()(:v)
                u = Constant(1)()(:u)
                net = DynamicNetwork(Variable[u,v], VariableGraph(), VariableGraph(v => [u]))
                run = Runtime(net)
                instantiate!(run, u, 1)
                instantiate!(run, v, 2)
                instantiate!(run, v, 4)
                inst2 = get_instance(run, v, 2)
                inst4 = get_instance(run, v, 4)
                @test get_sfunc(inst2) isa Cat
                @test get_sfunc(inst4) isa Constant
            end

            @testset "correctly instantiates variables and placeholders" begin
                v = MyModel()(:v)
                p = Placeholder{Int}(:p)
                net = DynamicNetwork(Variable[v], VariableGraph(), 
                    VariableGraph(v => [p]), VariableParentTimeOffset(), Placeholder[p])
                run = Runtime(net)
                inst1 = instantiate!(run, p, 1)
                inst2 = instantiate!(run, v, 2)
                @test isa(inst1, PlaceholderInstance)
                @test isa(inst2, VariableInstance)
            end

            @testset "passes the correct times to the model" begin
                v = MyModel()(:v)
                net = DynamicNetwork(Variable[], VariableGraph(), VariableGraph(v => [v]))
                run = Runtime(net)
                inst2 = instantiate!(run, v, 2)
                inst4 = instantiate!(run, v, 4)
                inst3 = instantiate!(run, v, 3)
                inst5 = instantiate!(run, v, 5)
                # Each instance should have a previous time of the previously existing latest instance, if any
                @test get_sfunc(inst4).x == (2,4)
                @test get_sfunc(inst3).x == (2,3)
                @test get_sfunc(inst5).x == (4,5)
            end

            @testset "passes the correct offset times to the model" begin
                u = MyModel()(:u)
                v = MyModel()(:v)
                net = DynamicNetwork(Variable[u,v], VariableGraph(), VariableGraph(v => [u], u => [u]), VariableParentTimeOffset([Pair(v, u)]))
                run = Runtime(net)
                uinst1 = instantiate!(run, u, 0)
                inst1 = instantiate!(run, v, 1)
                inst3 = instantiate!(run, v, 3)
                uinst4 = instantiate!(run, u, 4)
                uinst5 = instantiate!(run, u, 5)
                inst5 = instantiate!(run, v, 5)
                # Each instance should have a previous time of the previously existing latest instance, if any
                @test get_sfunc(inst3).x == (0,3)
                @test get_sfunc(inst5).x == (4,5)
            end

        end
    end

    @testset "models" begin
        @testset "HomogeneousModel" begin
            @testset "Initial model" begin
                sf = make_initial(randomwalk)
                @test sf isa MyNormal
                # Test some ops
                _sample = sample(sf, ())
                _logcpdf = logcpdf(sf, (), _sample)
            end

            @testset "Transition model" begin
                sf = make_transition(randomwalk,0.,2.)
                @test sf isa MyCondMuNormal
                # Test some ops
                _sample = sample(sf, (0.0f0,))
                _logcpdf = logcpdf(sf, (0.0f0,), _sample)
            end
        end

        @testset "VariableTimeModel" begin
            @testset "Initial model" begin
                sf = make_initial(wienerprocess)
                @test sf isa MyNormal
                # Test some ops
                _sample = sample(sf, ())
                _logcpdf = logcpdf(sf, (), _sample)
            end

            @testset "Transition model" begin
                sf = make_transition(wienerprocess,(0.,),2.)
                @test sf isa MyCondMuNormal
                # Test some ops
                _sample = sample(sf, (0.0f0,))
                _logcpdf = logcpdf(sf, (0.0f0,), _sample)
            end   
        end
    end
end

end
