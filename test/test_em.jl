using Test

using Scruff
using Scruff.Utils
using Scruff.RTUtils
using Scruff.SFuncs
using Scruff.Operators
using Scruff.Algorithms
import Scruff.Algorithms: em, bp_info_provider
using Scruff.Models
using Base.Filesystem
using Random

@testset "EM" begin
    
    @testset "Cat operations" begin
        range = [1,2,3]
        x = Cat(range, [0.2, 0.3, 0.5])

        @testset "initial_stats" begin
            @test initial_stats(x) == [0, 0, 0]
        end

        @testset "expected_stats" begin
            @test expected_stats(x, range, (), (), FunctionalScore{Int}(i->[0,1,0][i])) == [0.0, 0.3, 0.0]
            @test expected_stats(x, range, (), (), FunctionalScore{Int}(i->[0.5, 0.6, 0][i])) == [0.1, 0.18, 0.0]
        end

        @testset "accumulate_stats" begin
            @test accumulate_stats(x, [0.2, 0.3, 0.5], [0.5, 0.5, 0.0]) == [0.7, 0.8, 0.5]
        end

        @testset "maximize_stats" begin
            @test maximize_stats(x, [0.7, 0.8, 0.5]) == [0.35, 0.4, 0.25]
        end

        @testset "full loop" begin
            s = initial_stats(x)
            es1 = expected_stats(x, range, (), (), FunctionalScore{Int}(i->[0,1,0][i]))
            es2 = expected_stats(x, range, (), (), FunctionalScore{Int}(i->[0.5, 0.6, 0][i]))
            s = accumulate_stats(x, s, normalize(es1))
            s = accumulate_stats(x, s, normalize(es2)) 
            @test isapprox(maximize_stats(x, s), [0.1 / 0.28 / 2, (1 + 0.18 / 0.28) / 2, 0])
        end
    end

    @testset "DiscreteCPT operations" begin
        cpd = Dict((1,1) => [0.3, 0.7], (1,2) => [0.6, 0.4], (2,1) => [0.4, 0.6], (2,2) => [0.7, 0.3], (3,1) => [0.5, 0.5], (3,2) => [0.8, 0.2])
        x = DiscreteCPT([1,2], cpd)

        @testset "initial_stats" begin
            @test initial_stats(x) == Dict()
        end

        @testset "expected_stats " begin
            parpi1 = [0.2, 0.3, 0.5]
            parpi2 = [0.9, 0.1]
            lam = FunctionalScore{Int}(i->[0.1, 0.4][i])
            px11 = [0.3 * 0.1, 0.7 * 0.4] * 0.9 * 0.2
            px12 = [0.6 * 0.1, 0.4 * 0.4] * 0.1 * 0.2
            px21 = [0.4 * 0.1, 0.6 * 0.4] * 0.9 * 0.3
            px22 = [0.7 * 0.1, 0.3 * 0.4] * 0.1 * 0.3
            px31 = [0.5 * 0.1, 0.5 * 0.4] * 0.9 * 0.5
            px32 = [0.8 * 0.1, 0.2 * 0.4] * 0.1 * 0.5
            d = Dict((1,1) => px11, (1,2) => px12, (2,1) => px21, (2,2) => px22, (3,1) => px31, (3,2) => px32)
            stats = expected_stats(x, [1,2], ([1,2,3], [1,2]), (Cat([1,2,3], parpi1), Cat([1,2], parpi2)), lam)
            @test length(stats) == length(d)
            ans = [isapprox(stats[k], d[k]) for k in keys(stats)]
            @test all(ans)
            # for k in keys(stats)
            #     @test isapprox(stats[k], d[k])
            # end
        end

        @testset "accumulate_stats" begin
            d1 = Dict((1,1) => [0.4, 0.2], (1,2) => [0.5, 0.6])
            d2 = Dict((1,1) => [0.7, 0.8], (2,1) => [1.0, 0.0])
            d3 = Dict((1,1) => [1.1, 1.0], (1,2) => [0.5, 0.6], (2,1) => [1.0, 0.0])
            stats = accumulate_stats(x, d1, d2)
            # This test is written assuming the stats are not normalized while accumulating
            @test length(stats) == length(d3)
            for k in keys(stats)
                @test isapprox(stats[k], d3[k])
            end
        end

        @testset "maximize_stats" begin
            stats = Dict((1,1) => [0.55, 0.95], (1,2) => [0.7, 1.3], (2,1) => [0.6, 1.4],
                        (2,2) => [1.0, 1.0], (3,1) => [0.9, 1.1], (3,2) => [1.3, 0.7])
            m = maximize_stats(x, stats)
            for k in keys(stats)
                @test isapprox(m[k], normalize(stats[k]))
            end
        end
    end
#=
    @testset "Separable operations" begin
        alphas = [0.2, 0.3, 0.5]
        cpt1 = Dict((1,) => [0.1, 0.9], (2,) => [0.2, 0.8])
        cpt2 = Dict((1,) => [0.3, 0.7], (2,) => [0.4, 0.6], (3,) => [0.5, 0.5])
        cpt3 = Dict((1,) => [0.6, 0.4], (2,) => [0.7, 0.3])
        cpts::SepCPTs = [cpt1, cpt2, cpt3]
        x = Separable([1,2], alphas, cpts)
        insts = [Dict(), Dict(), Dict()]
        range = [1,2]
        parranges = ([1,2], [1,2,3], [1,2])
        parent_πs = ([0.8, 0.2], [0.5, 0.3, 0.2], [0.9, 0.1])
        child_λ = FunctionalScore{Int}(i->[0.1, 0.3][i]) 

        (p111, p112) = [0.1 * 0.1, 0.9 * 0.3] * 0.8
        (p121, p122) = [0.2 * 0.1, 0.8 * 0.3] * 0.2
        (p211, p212) = [0.3 * 0.1, 0.7 * 0.3] * 0.5
        (p221, p222) = [0.4 * 0.1, 0.6 * 0.3] * 0.3
        (p231, p232) = [0.5 * 0.1, 0.5 * 0.3] * 0.2
        (p311, p312) = [0.6 * 0.1, 0.4 * 0.3] * 0.9
        (p321, p322) = [0.7 * 0.1, 0.3 * 0.3] * 0.1
        es1 = Dict((1,) => [p111, p112] * alphas[1], (2,) => [p121, p122] * alphas[1])
        es2 = Dict((1,) => [p211, p212] * alphas[2], (2,) => [p221, p222] * alphas[2], (3,) => [p231, p232] * alphas[2])
        es3 = Dict((1,) => [p311, p312] * alphas[3], (2,) => [p321, p322] * alphas[3])
        exps = [es1, es2, es3]

        @testset "initial_stats" begin
            stats = initial_stats(x)
            @test stats == insts
        end

        # This test checks whether the code correctly computes according to the theory in the comment
        # prior to the function. It does not test whether this theory is correct.
        @testset "expected_stats" begin
            compparams = expected_stats(x, range, parranges, (Cat(parranges[1], parent_πs[1]), Cat(parranges[2], parent_πs[2]), Cat(parranges[3], parent_πs[3])), child_λ)
            for i in 1:3
                @test length(keys(compparams[i])) == length(keys(exps[i]))
                for j in keys(compparams[i])
                    @test isapprox(compparams[i][j], exps[i][j])
                end
            end
        end

        @testset "accumulate_stats" begin
            @test accumulate_stats(x, insts, exps) == exps
        end

        @testset "maximize_stats" begin
            ms = maximize_stats(x, exps)
            alphas = ms[1]
            ps = tuple(ms[2:end]...)
            a1 = sum(exps[1][(1,)]) + sum(exps[1][(2,)])
            a2 = sum(exps[2][(1,)]) + sum(exps[2][(2,)]) + sum(exps[2][(3,)])
            a3 = sum(exps[3][(1,)]) + sum(exps[3][(2,)])
            as = normalize([a1, a2, a3])
            @test length(alphas) == 3
            @test isapprox(alphas[1], as[1])
            @test isapprox(alphas[2], as[2])
            @test isapprox(alphas[3], as[3])
            @test length(cpts) == 3
            c11 = normalize(exps[1][(1,)])
            c12 = normalize(exps[1][(2,)])
            c21 = normalize(exps[2][(1,)])
            c22 = normalize(exps[2][(2,)])
            c23 = normalize(exps[2][(3,)])
            c31 = normalize(exps[3][(1,)])
            c32 = normalize(exps[3][(2,)])
            k1 = [x.components[1].given.inversemaps[1][i] for i in 1:2]
            k2 = [x.components[2].given.inversemaps[1][i] for i in 1:3]
            k3 = [x.components[3].given.inversemaps[1][i] for i in 1:2]
            @test isapprox(ps[1][k1[1][1]], c11)
            @test isapprox(ps[1][k1[2][1]], c12)
            @test isapprox(ps[2][k2[1][1]], c21)
            @test isapprox(ps[2][k2[2][1]], c22)
            @test isapprox(ps[2][k2[3][1]], c23)
            @test isapprox(ps[3][k3[1][1]], c31)
            @test isapprox(ps[3][k3[2][1]], c32)
        end
    end
=#
    ConfigurableCatModel(sf) = SimpleNumeric{Tuple{}, Int, Vector{Float64}}(sf)
    ConfigurableDiscreteCPTModel(I, sf) = SimpleNumeric{I, Int, Dict{I, Vector{Float64}}}(sf)

    sf1 = Cat([1,2], [0.1, 0.9])
    mod1 = ConfigurableCatModel(sf1)
    x1 = Variable(:x1, mod1)
    sf2 = Cat([1,2,3], [0.2,0.3,0.5])
    mod2 = ConfigurableCatModel(sf2)
    x2 = Variable(:x2, mod2)
    sf3 = DiscreteCPT([1,2], Dict((1,1) => [0.3, 0.7], (1,2) => [0.6, 0.4], (2,1) => [0.4, 0.6], 
                        (2,2) => [0.7, 0.3], (3,1) => [0.5, 0.5], (3,2) => [0.8, 0.2]))
    mod3 = ConfigurableDiscreteCPTModel(Tuple{Int, Int}, sf3)
    x3 = Variable(:x3, mod3)
    sf4 = DiscreteCPT([1,2], Dict((1,) => [0.15, 0.85], (2,) => [0.25, 0.75]))
    mod4 = ConfigurableDiscreteCPTModel(Tuple{Int}, sf4)
    x4 = Variable(:x4, mod4)
    sf5 = DiscreteCPT([1,2], Dict((1,) => [0.35, 0.65], (2,) => [0.45, 0.55]))
    mod5 = ConfigurableDiscreteCPTModel(Tuple{Int}, sf5)
    x5 = Variable(:x5, mod5)

    fivecpdnet = InstantNetwork([x1,x2,x3,x4,x5], VariableGraph(x3=>[x2,x1], x4=>[x3], x5=>[x3]))

    @testset "support functions" begin
        @testset "bp_info_provider" begin
            run = Runtime(fivecpdnet)
            default_initializer(run)
            three_pass_BP(run)
            # v_x3 = get_node(run, :x3)
            # inst = current_instance(run, v_x3)
            inst = current_instance(run, x3)
            sf = get_sfunc(inst)
            # m1 = get_message(run, get_node(run, :x1), v_x3, :pi_message)
            # m2 = get_message(run, get_node(run, :x2), v_x3, :pi_message)
            m1 = get_message(run, x1, x3, :pi_message)
            m2 = get_message(run, x2, x3, :pi_message)
            l = get_value(run, inst, :lambda)
            info = bp_info_provider(run, inst)
            expected_stats(sf, [1,2], ([1, 2, 3], [1, 2]), (m2, m1), l)
        end

        @testset "convergence" begin
            p1 = Dict(:a => 0.1, :b => [0.2, 0.3])
            p2 = Dict(:a => 0.1, :b => [0.2, 0.3])
            p3 = Dict(:a => 0.10001, :b => [0.20001, 0.30001])
            p4 = Dict(:a => 0.11, :b => [0.20001, 0.30001])
            p5 = Dict(:a => 0.1, :b => [0.2, 0.3], :c => 0.4)
            @test converged_numeric(p1, p2, 0.001)
            @test converged_numeric(p1, p3, 0.001)
            @test !converged_numeric(p1, p4, 0.001)
            @test !converged_numeric(p1, p5, 0.001)
            @test !converged_numeric(p5, p1, 0.001)
        end

    end

    @testset "EM" begin
  
        @testset "termination" begin
      
            @testset "should terminate immediately with false flag with 0 max_iterations" begin
                function err(runtime)
                    error()
                end
                @test em(fivecpdnet, nothing ; algorithm = err, max_iterations = 0)[1] == (false, 0)
            end

            @testset "should converge right away if parameters don't change" begin
                # if there's no evidence, the parameters shouldn't change
                data = [Dict()]
                @test em(fivecpdnet, data ; max_iterations = 2)[1] == (true, 2)
            end

            @testset "should converge in few iterations with fully observed data" begin
                # Ordinarily, this would be 2 iterations.
                # However, with noise added to the evidence, it may take a few more.
                data = [Dict(:x1 => HardScore(1), :x2 => HardScore(1), :x3 => HardScore(1), :x4 => HardScore(1), :x5 => HardScore(2))]
                @test !em(fivecpdnet, data ; max_iterations = 1)[1][1]
                # @test em(fivecpdnet, data ; max_iterations = 5)[1][1]
            end

            @testset "does not use less than min_iterations" begin
                data = [Dict(:x1 => HardScore(1), :x2 => HardScore(1), :x3 => HardScore(1), :x4 => HardScore(1), :x5 => HardScore(2))]
                @test em(fivecpdnet, data ; min_iterations = 7)[1] == (true, 7)
            end
            
        end

        @testset "learning with Cat" begin
            @testset "on single node" begin
                data = [
                        Dict(:x1 => HardScore(1)), 
                        Dict(:x1 => HardScore(2)), 
                        Dict(:x1 => HardScore(1)), 
                        Dict(:x1 => HardScore(1))
                        ]
                onenet = InstantNetwork([x1], VariableGraph())
                newparams = em(onenet, data)[2]
                @test isapprox(newparams[:x1],  normalize([3, 1]))
            end

            @testset "with two independent nodes, each observed sometimes" begin
                data = [
                    Dict(:x1 => HardScore(1)), 
                    Dict(:x2 => HardScore(1)), 
                    Dict(:x1 => HardScore(2), :x2 => HardScore(2))]
                twonet = InstantNetwork([x1, x2], VariableGraph())
                # We should converge to ignoring the unobserved cases since the variables are independent
                # and there is no other evidence affecting them
                newparams = em(twonet, data)[2]
                # When :x1 is not observed, its stats are 0.1, 0.9
                # When :x2 is not observed, its stats are 0.2, 0.3, 0.5
                @test isapprox(newparams[:x1], normalize([1 + 0.1, 1 + 0.9]))
                @test isapprox(newparams[:x2], normalize([1 + 0.2, 1 + 0.3, 0.5]))
            end

            @testset "with soft score, should consider prior" begin
                data = [Dict(:x1 => SoftScore(Dict(1 => 0.8, 2 => 0.2)))]
                onenet = InstantNetwork([x1], VariableGraph())
                newparams = em(onenet, data)[2]
                @test isapprox(newparams[:x1],  normalize([0.1 * 0.8, 0.9 * 0.2]))
            end

        end

        @testset "learning with DiscreteCPT" begin
        
            @testset "fully observed" begin
                data = [
                    Dict(:x1 => HardScore(1), :x2 => HardScore(1), :x3 => HardScore(1)), 
                    Dict(:x1 => HardScore(1), :x2 => HardScore(2), :x3 => HardScore(2)), 
                    Dict(:x1 => HardScore(2), :x2 => HardScore(3), :x3 => HardScore(2))]
                newparams = em(fivecpdnet, data)[2]
                # p3 = [[[1.0, 0.0], [0.5, 0.5]] [[0.0, 1.0], [0.5, 0.5]] [[0.5, 0.5], [0.0, 1.0]]]
                @test isapprox(newparams[:x1], [2.0 / 3, 1.0 / 3], atol = 0.0001)
                @test isapprox(newparams[:x2], [1.0 / 3, 1.0 / 3, 1.0 / 3], atol = 0.0001)
                p3 = newparams[:x3]
                @test isapprox(p3[(1,1)], [1.0, 0.0])
                @test isapprox(p3[(1,2)], [0.5, 0.5])
                @test isapprox(p3[(2,1)], [0.0, 1.0])
                @test isapprox(p3[(2,2)], [0.5, 0.5])
                @test isapprox(p3[(3,1)], [0.5, 0.5])
                @test isapprox(p3[(3,2)], [0.0, 1.0])
            end

            @testset "with children observed" begin
                # sf3 = Cat([1,2], [0.1, 0.9])
                # mod3 = SimpleNumeric(sf3)
                # x3 = Variable(:x3, mod3)
                # sf4 = DiscreteCPT([1,2], Dict((1,) => [0.9, 0.1], (2,) => [0.1, 0.9]))
                # mod4 = SimpleNumeric(sf4)
                # x4 = Variable(:x4, mod4)
                # sf5 = DiscreteCPT([1,2], Dict((1,) => [0.9, 0.1], (2,) => [0.1, 0.9]))
                # mod5 = SimpleNumeric(sf5)
                # x5 = Variable(:x5, mod5)
                
                # fivecpdnet = Network([x3,x4,x5], Placeholder[], Placeholder[], Dict(x4=>[x3], x5=>[x3]))
                data = [
                    Dict(:x4 => HardScore(1), :x5 => HardScore(1)), 
                    Dict(:x4 => HardScore(2), :x5 => HardScore(2))
                    ]
                ((converged, numiters), newparams) = em(fivecpdnet, data)
                p1s = newparams[:x1]
                p2s = newparams[:x2]
                p3s = newparams[:x3]
                # belief on :x3 calculated from learned network
                learned_belief3 = 
                    p2s[1] * p1s[1] * p3s[(1,1)] .+
                    p2s[1] * p1s[2] * p3s[(1,2)] .+
                    p2s[2] * p1s[1] * p3s[(2,1)] .+
                    p2s[2] * p1s[2] * p3s[(2,2)] .+
                    p2s[3] * p1s[1] * p3s[(3,1)] .+
                    p2s[3] * p1s[2] * p3s[(3,2)]

                # belief on :x3 computed analytically
                # prior distribution over node 3
                prior31 = 0.2 * 0.1 * 0.3 + 0.2 * 0.9 * 0.6 + 0.3 * 0.1 * 0.4 + 0.3 * 0.9 * 0.7 + 0.5 * 0.1 * 0.5 + 0.5 * 0.9 * 0.8
                prior32 = 1 - prior31 
                # Posteriors for each data case are computed by considering the appropriate prior and lambda collect_messages.
                # The these posteriors are summed over the data cases. (Note: each case is individually normalized so this makes sense.)
                case1p1 = prior31 * 0.15 * 0.35
                case1p2 = prior32 * 0.25 * 0.45
                case2p1 = prior31 * 0.85 * 0.65
                case2p2 = prior32 * 0.75 * 0.55
                case1post3 = normalize([case1p1, case1p2])
                case2post3 = normalize([case2p1, case2p2])
                post3 = normalize(case1post3 .+ case2post3)
                @test isapprox(learned_belief3, post3; atol = 0.01)
            end
          
        end
#=
        @testset "learning with Separable" begin
            # Testing separable models is challenging, because it can be hard to calculate what the maximizing alphas should be.
            # Therefore, we use extreme cases to test.
            @testset "with one component" begin
                alphas = [1.0]
                cpd1 = Dict((1,) => [0.2, 0.8])
                cpds::SepCPTs = [cpd1]
                sf1 = Cat([1], [1.0])
                mod1 = SimpleNumeric(sf1)
                x1 = Variable(:x1, mod1)
                sf2 = Separable([1,2], alphas, cpds)()
                mod2 = SimpleNumeric(sf2)
                x2 = Variable(:x2, mod2)
                net = Network([x1,x2], Placeholder[], Placeholder[], Dict(x2=>[x1]))
            
                data = [Dict(:x2 => 1), Dict(:x2 => 1), Dict(:x2 => 2), Dict(:x2 => 1)]
                cs = em(net, data)[2]
                @test isapprox(cs[(:x2)][2], [[0.75, 0.25]]; atol = 0.01)
            end

            @testset "when each parent predicts a different child value" begin
                alphas = [0.2, 0.3, 0.5]
                cpd1 = Dict((1,) => [1.0, 0.0, 0.0])
                cpd2 = Dict((1,) => [0.0, 1.0, 0.0])
                cpd3 = Dict((1,) => [0.0, 0.0, 1.0])
                cpds::SepCPTs = [cpd1, cpd2, cpd3]
                sf1 = Cat([1], [1.0])
                mod1 = SimpleNumeric(sf1)
                x1 = Variable(:x1, mod1)
                sf2 = Cat([1], [1.0])
                mod2 = SimpleNumeric(sf2)
                x2 = Variable(:x2, mod2)
                sf3 = Cat([1], [1.0])
                mod3 = SimpleNumeric(sf3)
                x3 = Variable(:x3, mod3)
                sf4 = Separable([1,2,3], alphas, cpds)
                x4 = Variable(:x4, mod4)
                net = Network([x1,x2,x3,x4], Placeholder[], Placeholder[], Dict(x4=>[x1,x2,x3]))
                
                data = [Dict(:x4 => 1), Dict(:x4 => 2), Dict(:x4 => 2), Dict(:x4 => 3)]
                newparams = em(net, data)[2]
                (as, cs1, cs2, cs3) = newparams[:x4]
                @test as == [0.25, 0.5, 0.25]
                @test cs1 == [[1.0, 0.0, 0.0]]
                @test cs2 == [[0.0, 1.0, 0.0]]
                @test cs3 == [[0.0, 0.0, 1.0]]
            end
        end
        =#
        #=
        @testset "Learning with If" begin
            sfc1 = Cat([1, 2], [0.1, 0.9])
            modc1 = SimpleNumeric(sfc1)
            vc1 = Variable(:c1, modc1)
            sfc2 = Cat([1, 2, 3], [0.2, 0.3, 0.5])
            modc2 = SimpleNumeric(sfc2)
            vc2 = Variable(:c2, modc2)
            sff = Flip(0.4)
            modf = SimpleNumeric(sff)
            vf = Variable(:f, modf)
            sfi = If{Int}
            modi = SimpleModel(sfi)
            vi = Variable(:i, modi)
            net = Network([vc1,vc2,vf,vi], Placeholder[], Placeholder[], Dict(vi=>[vf,vc1,vc2]))

            data = [Dict(:i => 1), Dict(:i => 2)]
            newparams = em(net, data)[2]
            pc1 = newparams[:c1]
            pc2 = newparams[:c2]
            pf = newparams[:f]
            pi = newparams[:i] 
            @test pi === nothing
            # The problem is underconstrained, but we can check that we have learned not to predict 3
            # We can also check that 1 and 2 have similar probability.
            # To avoid predicting 3, we need either f to always be true (second value) or c2 to always be 1 or 2.
            @test isapprox(pf[1], 0.0; atol = 0.05) || isapprox(pc2[3], 0.0; atol = 0.05)
            p1 = pf[1] * pc2[1] + pf[2] * pc1[1]
            p2 = pf[1] * pc2[2] + pf[2] * pc1[2]
            @test isapprox(p1, p2; atol = 0.05)
        end
        # Parameter sharing currently doesn't work
        @testset "parameter sharing" begin
            sf = Cat([1,2], [0.5, 0.5])
            m = sf()
            x1 = m(:x1)
            x2 = m(:x2)
            net = Network(Tuple{}, Nothing)
            add_variable!(net, x1)
            add_variable!(net, x2)
            data = [Dict(:x1 => 1, :x2 => 1), Dict(:x1 => 2, :x2 => 1)]
            newparams = em(net, data)[2]
            @test newparams[:x1] == [0.75, 0.25]
            @test newparams[:x2] == [0.75, 0.25]
        end
        =#
#= we do not currently have validation set code
        @testset "with validation set" begin
            @testset "retains the old parameters when learning makes validation set worse" begin
                m = Cat([1,2], [0.5, 0.5])
                x = m()(:x)
                net = Network([x], Placeholder[], Placeholder[], Dict())

                data = [Dict(:x => 1)]
                validation = [Dict(:x => 2)]
                ((conv, iteration), newparams) = em(net, data; validationset = validation)
                @test conv
                @test iteration == 1
                @test newparams[:x] == [0.5, 0.5]
            end

            @testset "when training set and validation set are the same, behaves like before" begin
                m = Cat([1,2], [0.5, 0.5])
                x = m()(:x)
                net = Network([x], Placeholder[], Placeholder[], Dict())
                
                data = [Dict(:x => 1), Dict(:x => 1), Dict(:x => 1), Dict(:x => 2)]
                validation = [Dict(:x => 1), Dict(:x => 1), Dict(:x => 1), Dict(:x => 2)]
                ((conv, iteration), newparams) = em(net, data; validationset = validation)
                @test conv
                @test iteration == 2
                @test newparams[:x] == [0.75, 0.25]
            end
=#
        end

    end


