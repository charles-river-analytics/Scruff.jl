import Base.timedwait
import Base.isapprox
import PrettyPrint

using Test
using ..Scruff
using ..Scruff.Utils
using ..Scruff.RTUtils
using ..Scruff.Models
using ..Scruff.SFuncs
using ..Scruff.Operators
import ..Scruff.Algorithms: LSFI, probability_bounds, prepare, refine, ve

using Scruff.MultiInterface

import Scruff.Operators: Support, bounded_probs, support_quality, support
import Scruff.SFuncs: Dist

struct MyFlip <: Dist{Bool} end

solve_count = 0

@impl begin
    struct MyFlipBoundedProbs end
    function bounded_probs(sf::MyFlip, 
            range::__OptVec{Bool}, 
            parranges::NTuple{N,Vector})::Tuple{Vector{<:AbstractFloat}, Vector{<:AbstractFloat}} where {N}

        global solve_count += 1
        return ([0.1, 0.9], [0.1, 0.9])
    end
end

@impl begin
    struct MyFlipSupport end

    function support(sf::MyFlip, 
                    parranges::NTuple{N,Vector}, 
                    size::Integer, 
                    curr::Vector{Bool}) where {N}
        [false, true]
    end
end    

@impl begin
    struct MyFlipSupportQuality end
    function support_quality(::MyFlip, parranges)
        :CompleteSupport
    end
end

@testset "lsfi" begin

    @testset "expander" begin

        @testset "Construction" begin
            x1 = Cat([1,2], [0.1, 0.9])()
            net1 = InstantNetwork(Variable[x1(:out)], VariableGraph())
            
            x2 = Cat([1,2], [0.2, 0.8])()
            net2 = InstantNetwork(Variable[x2(:out)], VariableGraph())
            
            f(b) = b ? net1 : net2
            b3v = Flip(0.5)()(:b3)
            e3v = Expander(f, Tuple{Bool}, Int)()(:out)
            net3 = InstantNetwork(Variable[e3v,b3v], VariableGraph(e3v=>[b3v]))
            
            run3 = Runtime(net3)
            ensure_expansion_state!(run3)

            @test isempty(get_state(run3, :subnets))
            @test isempty(get_state(run3, :subruntimes))
        end

        @testset "Expand" begin
            x1 = Cat([1,2], [0.1, 0.9])()
            net1 = InstantNetwork(Variable[x1(:out)], VariableGraph())
            
            x2 = Cat([1,2], [0.2, 0.8])()
            net2 = InstantNetwork(Variable[x2(:out)], VariableGraph())
            
            f(b) = b ? net1 : net2
            vb3 = Flip(0.5)()(:b3)
            ve3 = Expander(f, Tuple{Bool}, Int)()(:out)
            net3 = InstantNetwork(Variable[ve3,vb3], VariableGraph(ve3=>[vb3]))
            
            run3 = Runtime(net3)
            @testset "The first time" begin
                expand!(run3, ve3, true)
                @test expanded(run3, ve3, true)
                @test expansion(run3, ve3, true) == net1
                run1 = subruntime(run3, ve3, net1)
                @test run1.network == net1
                @test !expanded(run3, ve3, false)
            end
            @testset "The second time" begin
                expand!(run3, ve3, true)
                @test expanded(run3, ve3, true)
                @test !expanded(run3, ve3, false)
            end
        end

        @testset "Getting range" begin
            x1 = Cat([1,2], [0.1, 0.9])()
            net1 = InstantNetwork(Variable[x1(:out)], VariableGraph())
            
            x2 = Cat([1,2], [0.2, 0.8])()
            net2 = InstantNetwork(Variable[x2(:out)], VariableGraph())
            
            f₀(b) = (b) = b ? net1 : net2
            vb3 = Flip(0.5)()(:b3)
            ve3 = Expander(f₀ , Tuple{Bool}, Int)()(:out)
            net3 = InstantNetwork(Variable[ve3,vb3], VariableGraph(ve3=>[vb3]))

            @testset "With size requiring all expansions" begin
                run3 = Runtime(net3)
                ensure_all!(run3)
                instb3 = current_instance(run3, vb3)
                set_range!(run3, instb3, [false, true], 2)
                (rng, _) = expander_range(run3, ve3, 4, 2)
                @test expanded(run3, ve3, false)
                @test expanded(run3, ve3, true)
                @test rng == [1, 2]
            end

            @testset "With size not requiring all expansions" begin
                run3 = Runtime(net3)
                ensure_all!(run3)
                instb3 = current_instance(run3, vb3)
                set_range!(run3, instb3, [false, true], 2)
                (rng, _) = expander_range(run3, ve3, 1, 2)
                @test expanded(run3, ve3, false)
                @test !expanded(run3, ve3, true)
                @test rng == [1, 2]
            end

            @testset "With recursion" begin
                b1 = Cat([1,2], [0.1, 0.9])()
                net1 = InstantNetwork(Variable[b1(:out)], VariableGraph())
                
                b2 = Cat([1,2], [0.2, 0.8])()
                net2 = InstantNetwork(Variable[b2(:out)], VariableGraph())
                

                f₁(b) = b ? net1 : net2
                vb3 = Flip(0.5)()(:b3)
                ve3 = Expander(f₁, Tuple{Bool}, Int)()(:out)
                net3 = InstantNetwork(Variable[ve3,vb3], VariableGraph(ve3=>[vb3]))
        

                f₂(b) = b ? net1 : net2
                vb3 = Flip(0.7)()(:b3)
                ve3 = Expander(f₂, Tuple{Bool}, Int)()(:out)
                net3 = InstantNetwork(Variable[vb3,ve3], VariableGraph(ve3=>[vb3]))
                
                g(b) = net3
                vb4 = Flip(0.6)()(:b4)
                ve4 = Expander(g, Tuple{Bool}, Int)()(:out)
                net4 = InstantNetwork(Variable[vb4,ve4], VariableGraph(ve4=>[vb4]))
                run4 = Runtime(net4)
                
                ensure_all!(run4)
                expand!(run4, ve4, false)
                expand!(run4, ve4, true)

                run3 = subruntime(run4, ve4, net3)
                expand!(run3, ve3, false)
                expand!(run3, ve3, true)
                
                instb4 = current_instance(run4, vb4)
                set_range!(run4, instb4, [false, true], 2)
                
                (rng, _) = expander_range(run4, ve4, 4, 2)
                @test rng == [] # recursion depth too shallow
                
                (rng, _) = expander_range(run4, ve4, 4, 3)
                @test Set(rng) == Set([1, 2])
            end
            
        end

        @testset "Expanding range" begin
            vx1 = Normal(-0.1, 1.0)()(:out)
            net1 = InstantNetwork(Variable[vx1], VariableGraph())

            vx2 = Normal(0.4, 1.0)()(:x2)
            vy2 = LinearGaussian((0.7,), 0.0, 1.0)()(:out)
            net2 = InstantNetwork(Variable[vx2,vy2], VariableGraph(vy2=>[vx2]))

            g(b) = b ? net1 : net2
            vb3 = Flip(0.5)()(:b3)
            ve3 = Expander(g, Tuple{Bool}, Float64)()(:out)
            net3 = InstantNetwork(Variable[vb3,ve3], VariableGraph(ve3=>[vb3]))

            run3 = Runtime(net3)
            
            instb3 = instantiate!(run3, vb3, 0)
            set_range!(run3, instb3, [false, true], 2)
            (r1,_) = expander_range(run3, ve3, 5, 2)
            (r2,_) = expander_range(run3, ve3, 10, 2)
            
            @test length(r2) >= 10
            @test issorted(r2)
            @test length(Set(r2)) == length(r2)
            @test issubset(Set(r1), Set(r2))
        end

        @testset "Computing probability bounds" begin
            vx1 = Cat([1,2], [0.1, 0.9])()(:out)
            net1 = InstantNetwork(Variable[vx1], VariableGraph())
            
            vx2 = Cat([1,2], [0.2, 0.8])()(:out)
            net2 = InstantNetwork(Variable[vx2], VariableGraph())
            
            f(b) = b ? net1 : net2
            vb3 = Flip(0.5)()(:b3)
            ve3 = Expander(f, Tuple{Bool}, Int)()(:out)
            net3 = InstantNetwork(Variable[vb3,ve3], VariableGraph(ve3=>[vb3]))
            
            @testset "With a fully expanded model" begin
                run3 = Runtime(net3)
                ensure_all!(run3)
                instb3 = current_instance(run3, vb3)
                set_range!(run3, instb3, [false, true])
                ord = topsort(get_initial_graph(net3))
                set_ranges!(run3, Dict{Symbol, Score}(), 4, 2, ord)
                f₃(runtime, order, query_vars, depth) = ve(runtime, order, query_vars; depth = depth, bounds = true)
                (l, u) = expander_probs(run3, f₃, ve3, 2)

                @test l == [0.2, 0.8, 0.1, 0.9]
                @test u == [0.2, 0.8, 0.1, 0.9]
            end

            @testset "With a partially expanded model" begin
                run3 = Runtime(net3)
                ensure_all!(run3)
                instb3 = current_instance(run3, vb3)
                set_range!(run3, instb3, [false, true])
                ord = topsort(get_initial_graph(net3))
                set_ranges!(run3, Dict{Symbol, Score}(), 1, 2, ord)
                f₄(runtime, order, query_vars, depth) = ve(runtime, order, query_vars; depth = depth, bounds = true)
                (l, u) = expander_probs(run3, f₄, ve3, 2)

                @test l == [0.2, 0.8, 0, 0]
                @test u == [0.2, 0.8, 1, 1]
            end
        end
       
    end

    @testset "inside lsfi" begin

        @testset "Completeness" begin
            vx1 = Cat([1,2], [0.1, 0.9])()(:out)
            net1 = InstantNetwork(Variable[vx1], VariableGraph())
            
            vx2 = Cat([1,2], [0.2, 0.8])()(:out)
            net2 = InstantNetwork(Variable[vx2], VariableGraph())

            vx3 = Normal(-0.1, 1.0)()(:out)
            net3 = InstantNetwork(Variable[vx3], VariableGraph())
            
            vx4 = Normal(0.4, 1.0)()(:x4)
            vy4 = LinearGaussian((0.7,), 0.0, 1.0)()(:out)
            net4 = InstantNetwork(Variable[vx4,vy4], VariableGraph(vy4=>[vx4]))
            
            f5(b) = b ? net1 : net2
            vb5 = Flip(0.5)()(:b5)
            ve5 = Expander(f5, Tuple{Bool}, Int)()(:out)
            net5 = InstantNetwork(Variable[vb5,ve5], VariableGraph(ve5=>[vb5]))

            f6(b) = b ? net3 : net4
            vb6 = Flip(0.6)()(:b6)
            ve6 = Expander(f6, Tuple{Bool}, Float64)()(:out)
            net6 = InstantNetwork(Variable[vb6,ve6], VariableGraph(ve6=>[vb6]))

            run5 = Runtime(net5)
            ord5 = topsort(get_initial_graph(net5))
            run6 = Runtime(net6)
            ord6 = topsort(get_initial_graph(net6))
            ensure_all!(run5, 0)
            ensure_all!(run6, 0)
            expand!(run5, ve5, false)
            expand!(run5, ve5, true)
            expand!(run6, ve6, false)
            expand!(run6, ve6, true)

            @test !RTUtils._complete(run5)

            set_ranges!(run5, Dict{Symbol, Score}(), 1, 3, ord5) # range is too small to be complete
            @test !RTUtils._complete(run5)              # because subnetwork not expanded

            set_ranges!(run5, Dict{Symbol, Score}(), 4, 3, ord5) # range is large enough
            @test RTUtils._complete(run5)

            set_ranges!(run6, Dict{Symbol, Score}(), 10, 3, ord6)
            @test !RTUtils._complete(run6)

            set_ranges!(run6, Dict{Symbol, Score}(), 20, 3, ord6) # continuous range not complete
            @test !RTUtils._complete(run6)
        end

        @testset "Completeness with recursion" begin
            f(b) = net1
            vb1 = Flip(0.1)()(:b1)
            ve1 = Expander(f, Tuple{Bool}, Bool)()(:out)
            net1 = InstantNetwork(Variable[vb1,ve1], VariableGraph(ve1=>[vb1]))

            run1 = Runtime(net1)
            expand!(run1, ve1, false)
            expand!(run1, ve1, true)
            
            @test !RTUtils._complete(run1)
        end

        @testset "Run to completion" begin
            v1 = Cat([1,2], [0.1, 0.9])()(:out)
            net1 = InstantNetwork(Variable[v1], VariableGraph())

            x2 = Cat([1,2], [0.2, 0.8])()(:out)
            net2 = InstantNetwork(Variable[x2], VariableGraph())

            vb5 = Flip(0.4)()(:b5)
            f5(b) = b ? net2 : net1
            ve5 = Expander(f5, Tuple{Bool}, Int)()(:out)
            net5 = InstantNetwork(Variable[vb5,ve5], VariableGraph(ve5=>[vb5]))

            run5 = Runtime(net5)
            ord5 = topsort(get_initial_graph(net5))
            ensure_all!(run5, 0)
            alg = LSFI([ve5]; start_depth = 2)
            prepare(alg, run5)
            refine(alg, run5)
            inst = current_instance(run5, ve5)
            range = get_range(run5, inst)
            (ls,us) = probability_bounds(alg, run5, inst, range)
            pa = 0.4 * 0.2 + 0.6 * 0.1
            pb = 0.4 * 0.8 + 0.6 * 0.9
            @test isapprox(ls, [pa, pb])
            @test isapprox(us, [pa, pb])
        end

        @testset "With dynamic programming" begin
            expand_count = 0

            global function PrettyPrint.pp_impl(io, frame::StackTraces.StackFrame, indent::Int)
                line = frame.inlined ? "[inlined]" : "$(frame.line)"
                print(io, "$(frame.func) at $(frame.file):$(line)")
                indent
            end

            b1 = MyFlip()()(:out)
            net1 = InstantNetwork(Variable[b1], VariableGraph())

            function f1(x)
                if x expand_count += 1 end
                net1
            end

            vb2 = Flip(0.2)()(:b2)
            ve2 = Expander(f1, Tuple{Bool}, Bool)()(:out)
            net2 = InstantNetwork(Variable[vb2,ve2], VariableGraph(ve2=>[vb2]))

            vb3 = Flip(0.3)()(:b3)
            ve3 = Expander(x -> net2, Tuple{Bool}, Bool)()(:out)
            net3 = InstantNetwork(Variable[vb3,ve3], VariableGraph(ve3=>[vb3]))

            vb4 = Flip(0.4)()(:b4)
            ve4 = Expander(x -> net2, Tuple{Bool}, Bool)()(:out)
            net4 = InstantNetwork(Variable[vb4,ve4], VariableGraph(ve4=>[vb4]))

            vb5 = Flip(0.5)()(:b5)
            ve5 = Expander(b -> b ? net3 : net4, Tuple{Bool}, Bool)()(:out)
            net5 = InstantNetwork(Variable[vb5,ve5], VariableGraph(ve5=>[vb5]))

            run5 = Runtime(net5)
            ensure_all!(run5, 0)
            order = topsort(get_initial_graph(net5))
            alg = LSFI([ve5]; start_depth = 10)
            prepare(alg, run5)
            refine(alg, run5)
            inst = current_instance(run5, ve5)
            range = get_range(run5, inst)
            (lower, upper) = probability_bounds(alg, run5, inst, range)
            @test isapprox(lower, [0.1, 0.9], atol = 0.0000001)
            @test isapprox(upper, lower)
        end

        @testset "With recursion" begin
            bx1 = Flip(1.0)()(:out)
            netx1 = InstantNetwork(Variable[bx1], VariableGraph())

            vx2 = Flip(0.9)()(:b2)
            fx(b) = b ? netx2 : netx1
            vex2 = Expander(fx, Tuple{Bool}, Bool)()(:out)
            netx2 = InstantNetwork(Variable[vx2,vex2], VariableGraph(vex2=>[vx2]))

            runtime = Runtime(netx2)
            ensure_all!(runtime, 0)
            alg = LSFI([vex2]; start_depth = 11)
            prepare(alg, runtime)
            refine(alg, runtime)
            @test true # This test succeeds if the above call terminates
        end
       
    end

    @testset "Using the LazyInference interface" begin
        @testset "LSFI" begin
            m1 = Cat([1,2], [0.1, 0.9])()
            v1 = m1(:out)
            net1 = InstantNetwork(Variable[v1], VariableGraph())

            m2 = Cat([1,2], [0.2, 0.8])()
            v2 = m2(:out)
            net2 = InstantNetwork(Variable[v2], VariableGraph())

            f(b) = b ? net1 : net2
            m3 = Flip(0.9)()
            m4 = Expander(f, Tuple{Bool}, Int)()
            m5 = DiscreteCPT([1,2], Dict((1,) => [0.3, 0.7], (2,) => [0.4, 0.6]))()
            v3 = m3(:v3)
            v4 = m4(:v4)
            v5 = m5(:out)
            net3 = InstantNetwork(Variable[v3,v4,v5], VariableGraph(v4 => [v3], v5 => [v4]))

            alg = LSFI([get_node(net3, :out)]; increment = 3, max_iterations = 20, start_size=3)
                runtime = Runtime(net3)
            ensure_all!(runtime, 0)
            prepare(alg, runtime)
            refine(alg, runtime)
            @test has_instance(runtime, v5)
            i5 = current_instance(runtime, v5)
            state = alg.state
            @test state.next_size == 6
            @test state.next_iteration == 2
            @test state.next_depth == 2
            @test !state.is_complete
            # At first expansion, the bounds for i5 are (0,1)
            (ls1, us1) = probability_bounds(alg, runtime, i5, [1,2])
            @test isapprox(ls1[1], 0.0)
            @test isapprox(us1[1], 1.0)

            q2 = 0.9 * (0.1 * 0.3 + 0.9 * 0.4) + 0.1 * (0.2 * 0.3 + 0.8 * 0.4)
            refine(alg, runtime)
            @test has_instance(runtime, v5)
            i5 = current_instance(runtime, v5)
            state = alg.state
            @test state.next_size == 9
            @test state.next_iteration == 3
            @test state.next_depth == 3
            @test state.is_complete
            # At second expansion, lower bounds = upper bounds = q2
            (ls2, us2) = probability_bounds(alg, runtime, i5, [1,2])
            @test isapprox(ls2[1], q2)
            @test isapprox(us2[1], q2)
        end
    end   
 
end

