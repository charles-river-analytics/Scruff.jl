using Test
using Scruff
using Scruff.Utils
using Scruff.RTUtils
using Scruff.Models
using Scruff.SFuncs
using Scruff.Operators
import Scruff.Algorithms: VE, ve, infer, probability, greedy_order, unconnected_neighbors, cost, copy_graph, eliminate

@testset "VE" begin

    @testset "vegraph" begin
        g = Graph()
        add_node!(g, 1, 2)
        add_node!(g, 2, 3)
        add_node!(g, 3, 2)
        add_node!(g, 4, 2)
        add_node!(g, 5, 5)
        add_node!(g, 6, 1)
        add_undirected!(g, 1, 3)
        add_undirected!(g, 2, 3)
        add_undirected!(g, 3, 4)
        add_undirected!(g, 3, 5)
        add_undirected!(g, 1, 4)
        @testset "Construction" begin
            @test g.nodes == [1, 2, 3, 4, 5, 6]
            @test g.sizes == Dict(1 => 2, 2 => 3, 3 => 2, 4 => 2, 5 => 5, 6 => 1)
            @test g.edges == Dict(1 => [3, 4], 2 => [3], 3 => [1, 2, 4, 5],
                                    4 => [3, 1], 5 => [3], 6 => [])
        end

        @testset "Unconnected neighbors" begin
            @test unconnected_neighbors(g, 3) == [(1,2), (1,5), (2,4), (2,5), (4,5)]
            @test unconnected_neighbors(g, 6) == []
            @test cost(g, 3) == 5
            @test cost(g, 6) == 0
        end

        @testset "Elimination" begin
            h = copy_graph(g)
            eliminate(h, 3)
            @test h.nodes == [1, 2, 4, 5, 6]
            @test h.sizes == Dict(1 => 2, 2 => 3, 4 => 2, 5 => 5, 6 => 1)
            @test h.edges == Dict(1 => [4, 2, 5], 2 => [1, 4, 5],
                                    4 => [1, 2, 5], 5 => [1, 2, 4], 6 => [])
        end

        @testset "Greedy elimination order" begin
            @testset "With all variables eliminated" begin
                ord = greedy_order(g)
                @test Set(ord) == Set([1,2,3,4,5,6])
                @test length(ord) == 6
                # Only 3 has unconnected neighbors.
                # It cannot be eliminated before at least 3 of its neighbors
                # have been eliminated.
                inds = indexin([1,2,4,5,3], ord)
                count = 0
                if inds[1] < inds[4] count += 1 end
                if inds[2] < inds[4] count += 1 end
                if inds[3] < inds[4] count += 1 end
                if inds[4] < inds[4] count += 1 end
                @test count >= 3
            end

            @testset "With uneliminated variables" begin
                ord = greedy_order(g, [5, 6])
                @test Set(ord) == Set([1,2,3,4])
                @test length(ord) == 4
                # 3 must be eliminated last because all the others
                # are a disconnected neighbor from 5
                @test ord[4] == 3
            end
        end
    end

    @testset "range" begin
        dn1 = Cat([1,2], [0.1, 0.9])
        dv1 = dn1()(:dv1)
        dn2 = Cat([1,2,3], [0.2, 0.3, 0.5])
        dv2 = dn2()(:dv2)
        dn3 = DiscreteCPT([1,2],
            Dict((1,1) => [0.3, 0.7], (2,1) => [0.6, 0.4],
                 (1,2) => [0.4, 0.6], (2,2) => [0.7, 0.3],
                 (1,3) => [0.5, 0.5], (2,3) => [0.8, 0.2]))
        dv3 = dn3()(:dv3)
        cn1 = Normal(-0.1, 1.0)
        cv1 = cn1()(:cv1)
        cn2 = CLG(Dict((1,) => ((1.5,), -0.3, 0.5), (2,) => ((0.7,), 0.4, 0.5)))
        cv2 = cn2()(:cv2)

        network = InstantNetwork(Variable[dv1,dv2,dv3,cv1,cv2], VariableGraph(dv3=>[dv1,dv2],cv2=>[dv3,cv1]))
        runtime = Runtime(network)
        ensure_all!(runtime, 0)
        order = topsort(get_initial_graph(network))

        iv = Vector{Int}()
        fv = Vector{Float64}()
        dr1 = Operators.support(dn1, (), 10, iv)
        dr2 = Operators.support(dn2, (), 10, iv)
        dr3 = Operators.support(dn3, (dr1, dr2), 10, iv)
        cr1 = Operators.support(cn1, (), 10, fv)
        cr2 = Operators.support(cn2, (dr3, cr1), 10, fv)

        dx1 = Operators.support(dn1, (), 20, dr1)
        dx2 = Operators.support(dn2, (), 20, dr2)
        dx3 = Operators.support(dn3, (dx1, dx2), 20, dr3)
        cx1 = Operators.support(cn1, (), 20, cr1)
        cx2 = Operators.support(cn2, (dx3, cx1), 20, cr2)

        @testset "Setting initial ranges" begin
            set_ranges!(runtime, Dict{Symbol, Score}(), 10)
            @test get_range(runtime, dv1) == dr1
            @test get_range(runtime, dv2) == dr2
            @test get_range(runtime, dv3) == dr3
            @test get_range(runtime, cv1) == cr1
            @test get_range(runtime, cv2) == cr2
        end

        @testset "Setting expanded ranges" begin
            set_ranges!(runtime, Dict{Symbol, Score}(), 20)
            @test get_range(runtime, dv1) == dx1
            @test get_range(runtime, dv2) == dx2
            @test get_range(runtime, dv3) == dx3
            @test get_range(runtime, cv1) == cx1
            @test get_range(runtime, cv2) == cx2
        end

        @testset "Ranges from previous instance" begin
            ensure_all!(runtime, 2)
            @test get_range(runtime, dv1) == dx1
        end
        
    end

    @testset "ve" begin
        dn1 = Cat([1,2], [0.1, 0.9])
        i11 = indexin(1, dn1.range)[1]
        i12 = indexin(2, dn1.range)[1]
        dv1 = dn1()(:dv1)
        dn2 = Cat([1,2,3], [0.2, 0.3, 0.5])
        i21 = indexin(1, dn2.range)[1]
        i22 = indexin(2, dn2.range)[1]
        i23 = indexin(3, dn2.range)[1]
        dv2 = dn2()(:dv2)
        dn3 = DiscreteCPT([1,2], Dict((1,1) => [0.3, 0.7], (2,1) => [0.4, 0.6],
                                (1,2) => [0.5, 0.5], (2,2) => [0.6, 0.4],
                                (1,3) => [0.7, 0.3], (2,3) => [0.8, 0.2]))
        i31 = indexin(1, dn3.sfs[1,1].range)[1]
        i32 = indexin(2, dn3.sfs[1,1].range)[1]
        dv3 = dn3()(:dv3)
        cn1 = Normal(-0.1, 1.0)
        cv1 = cn1()(:cv1)
        cn2 = CLG(Dict((1,) => ((1.5,), -0.3, 0.5), (2,) => ((0.7,), 0.4, 0.5)))
        cv2 = cn2()(:cv2)
        net1 = InstantNetwork(Variable[dv1,dv2,dv3], VariableGraph(dv3=>[dv1,dv2]))
        ord1 = topsort(get_initial_graph(net1))

        @testset "A discrete network" begin
            
            @testset "With one query variable and bounds" begin
                runtime = Runtime(net1)
                ensure_all!(runtime, 0)
                set_ranges!(runtime, Dict{Symbol, Score}(), 10)
                ((l,u), ids) = ve(runtime, ord1, [dv3]; bounds = true)
                pa = 0.1 * 0.2 * 0.3 + 0.1 * 0.3 * 0.5 + 0.1 * 0.5 * 0.7 +
                        0.9 * 0.2 * 0.4 + 0.9 * 0.3 * 0.6 + 0.9 * 0.5 * 0.8
                pb = 1 - pa
                @test length(l.keys) == 1
                @test l.keys[1] == ids[dv3]
                @test l.dims == (2,)
                @test length(l.entries) == 2
                @test isapprox(l.entries[i31], pa, atol = 0.0000001)
                @test isapprox(l.entries[i32], pb, atol = 0.0000001)
                @test length(u.keys) == 1
                @test u.keys[1] == ids[dv3]
                @test u.dims == (2,)
                @test length(u.entries) == 2
                @test isapprox(u.entries[i31], pa, atol = 0.0000001)
                @test isapprox(u.entries[i32], pb, atol = 0.0000001)
            end
            
            @testset "With one query variable and no bounds" begin
                runtime = Runtime(net1)
                ensure_all!(runtime, 0)
                set_ranges!(runtime, Dict{Symbol, Score}(), 10)
                (l,ids) = ve(runtime, ord1, [dv3]; bounds = false)
                pa = 0.1 * 0.2 * 0.3 + 0.1 * 0.3 * 0.5 + 0.1 * 0.5 * 0.7 +
                        0.9 * 0.2 * 0.4 + 0.9 * 0.3 * 0.6 + 0.9 * 0.5 * 0.8
                pb = 1 - pa
                @test length(l.keys) == 1
                @test l.keys[1] == ids[dv3]
                @test l.dims == (2,)
                @test length(l.entries) == 2
                @test isapprox(l.entries[i31], pa, atol = 0.0000001)
                @test isapprox(l.entries[i32], pb, atol = 0.0000001)
            end

            @testset "With disconnected variable" begin
                x = Cat([1,2], [0.5, 0.5])()(:x)
                y = Cat([1,2], [0.2, 0.8])()(:y)
                net = InstantNetwork(Variable[x,y], VariableGraph())
                run = Runtime(net)
                ensure_all!(run)
                ord = topsort(get_initial_graph(net))
                set_ranges!(run, Dict{Symbol, Score}(), 2)
                (l,ids) = ve(run, ord, [x]; bounds = false)
                @test l.entries == [0.5, 0.5]
            end

            @testset "With two query variables" begin
                runtime = Runtime(net1)
                ensure_all!(runtime, 0)
                set_ranges!(runtime, Dict{Symbol, Score}(), 10)
                (l,ids) = ve(runtime, ord1, [dv3, dv1]; bounds = false)
                ppa = 0.1 * 0.2 * 0.3 + 0.1 * 0.3 * 0.5 + 0.1 * 0.5 * 0.7
                ppb = 0.1 * 0.2 * 0.7 + 0.1 * 0.3 * 0.5 + 0.1 * 0.5 * 0.3
                pqa = 0.9 * 0.2 * 0.4 + 0.9 * 0.3 * 0.6 + 0.9 * 0.5 * 0.8
                pqb = 0.9 * 0.2 * 0.6 + 0.9 * 0.3 * 0.4 + 0.9 * 0.5 * 0.2
                @test l.dims == (2,2)
                @test length(l.keys) == 2
                @test length(l.entries) == 4
                k1 = l.keys[1]
                k2 = l.keys[2]
                @test k1 == ids[dv1] && k2 == ids[dv3] || k1 == ids[dv3] && k2 == ids[dv1]
                if k1 == ids[dv1]
                    @test isapprox(l.entries[(i11-1)*2 + i31], ppa, atol = 0.000001)
                    @test isapprox(l.entries[(i11-1)*2 + i32], ppb, atol = 0.000001)
                    @test isapprox(l.entries[(i12-1)*2 + i31], pqa, atol = 0.000001)
                    @test isapprox(l.entries[(i12-1)*2 + i32], pqb, atol = 0.000001)
                else
                    @test isapprox(l.entries[(i31-1)*2 + i11], ppa, atol = 0.000001)
                    @test isapprox(l.entries[(i31-1)*2 + i12], pqa, atol = 0.000001)
                    @test isapprox(l.entries[(i32-1)*2 + i11], ppb, atol = 0.000001)
                    @test isapprox(l.entries[(i32-1)*2 + i12], pqb, atol = 0.000001)
                end
                
            end

            @testset "with hard evidence" begin
                runtime = Runtime(net1)
                ensure_all!(runtime)
                set_ranges!(runtime, Dict{Symbol, Score}(), 10)
                inst1 = current_instance(runtime, dv1)
                post_evidence!(runtime, inst1, HardScore(1))
                (l, ids) = ve(runtime, ord1, [dv3]; bounds = false)
                p31 = 0.1 * (0.2 * 0.3 + 0.3 * 0.5 + 0.5 * 0.7)
                p32 = 0.1 * (0.2 * 0.7 + 0.3 * 0.5 + 0.5 * 0.3)
                @test isapprox(l.entries[i31], p31, atol = 0.000001)
                @test isapprox(l.entries[i32], p32, atol = 0.000001)
            end

            @testset "with soft evidence" begin
                runtime = Runtime(net1)
                ensure_all!(runtime)
                set_ranges!(runtime, Dict{Symbol, Score}(), 10)
                inst1 = current_instance(runtime, dv1)
                post_evidence!(runtime, inst1, SoftScore([1,2], [3.0, 5.0]))
                (l, ids) = ve(runtime, ord1, [dv3]; bounds = false)
                p31 = 0.1 * 3.0 * (0.2 * 0.3 + 0.3 * 0.5 + 0.5 * 0.7) +
                        0.9 * 5.0 * (0.2 * 0.4 + 0.3 * 0.6 + 0.5 * 0.8)
                p32 = 0.1 * 3.0 * (0.2 * 0.7 + 0.3 * 0.5 + 0.5 * 0.3) +
                        0.9 * 5.0 * (0.2 * 0.6 + 0.3 * 0.4 + 0.5 * 0.2)
                @test isapprox(l.entries[i31], p31, atol = 0.000001)
                @test isapprox(l.entries[i32], p32, atol = 0.000001)
            end

            @testset "with separable models" begin
                sf1 = Cat([1,2], [0.1, 0.9])
                z1 = sf1()(:z1)
                sf2 = DiscreteCPT([1,2], Dict((1,) => [0.2, 0.8], (2,) => [0.3, 0.7]))
                z2 = sf2()(:z2)
                sf3 = DiscreteCPT([1,2], Dict((1,) => [0.4, 0.6], (2,) => [0.5, 0.5]))
                z3 = sf3()(:z3)
                cpt1 = Dict((1,) => [0.6, 0.4], (2,) => [0.7, 0.3])
                cpt2 = Dict((1,) => [0.8, 0.2], (2,) => [0.9, 0.1])
                cpts :: SepCPTs = [cpt1, cpt2]
                sf4 = Separable([1,2], [0.75, 0.25], cpts)
                z4 = sf4()(:z4)
                net = InstantNetwork(Variable[z1,z2,z3,z4], VariableGraph(z2=>[z1],z3=>[z1],z4=>[z2,z3]))
                ord = topsort(get_initial_graph(net))
            
                aceg = 0.1 * 0.2 * 0.4 * (0.75 * 0.6 + 0.25 * 0.8)
                aceh = 0.1 * 0.2 * 0.4 * (0.75 * 0.4 + 0.25 * 0.2)
                acfg = 0.1 * 0.2 * 0.6 * (0.75 * 0.6 + 0.25 * 0.9)
                acfh = 0.1 * 0.2 * 0.6 * (0.75 * 0.4 + 0.25 * 0.1)
                adeg = 0.1 * 0.8 * 0.4 * (0.75 * 0.7 + 0.25 * 0.8)
                adeh = 0.1 * 0.8 * 0.4 * (0.75 * 0.3 + 0.25 * 0.2)
                adfg = 0.1 * 0.8 * 0.6 * (0.75 * 0.7 + 0.25 * 0.9)
                adfh = 0.1 * 0.8 * 0.6 * (0.75 * 0.3 + 0.25 * 0.1)
                bceg = 0.9 * 0.3 * 0.5 * (0.75 * 0.6 + 0.25 * 0.8)
                bceh = 0.9 * 0.3 * 0.5 * (0.75 * 0.4 + 0.25 * 0.2)
                bcfg = 0.9 * 0.3 * 0.5 * (0.75 * 0.6 + 0.25 * 0.9)
                bcfh = 0.9 * 0.3 * 0.5 * (0.75 * 0.4 + 0.25 * 0.1)
                bdeg = 0.9 * 0.7 * 0.5 * (0.75 * 0.7 + 0.25 * 0.8)
                bdeh = 0.9 * 0.7 * 0.5 * (0.75 * 0.3 + 0.25 * 0.2)
                bdfg = 0.9 * 0.7 * 0.5 * (0.75 * 0.7 + 0.25 * 0.9)
                bdfh = 0.9 * 0.7 * 0.5 * (0.75 * 0.3 + 0.25 * 0.1)
        
                @testset "without evidence" begin
                    run = Runtime(net)
                    ensure_all!(run)
                    set_ranges!(run, Dict{Symbol, Score}(), 10)
                    (l, ids) = ve(run, ord, [z4]; bounds = false)
                    g = aceg + acfg + adeg + adfg + bceg + bcfg + bdeg + bdfg
                    h = aceh + acfh + adeh + adfh + bceh + bcfh + bdeh + bdfh
                    es = normalize(l.entries)
                    r4 = get_range(run, current_instance(run, z4))
                    i1 = indexin(1, r4)[1]
                    i2 = indexin(2, r4)[1]
                    @test isapprox(es[i1], g)
                    @test isapprox(es[i2], h)
                end

                @testset "with evidence" begin
                    run = Runtime(net)
                    ensure_all!(run)
                    set_ranges!(run, Dict{Symbol, Score}(), 10)
                    inst4 = current_instance(run, z4)
                    post_evidence!(run, inst4, HardScore(2))
                    (l, ids) = ve(run, ord, [z1]; bounds = false)
                    ah = aceh + acfh + adeh + adfh
                    bh = bceh + bcfh + bdeh + bdfh
                    es = normalize(l.entries)
                    r1 = get_range(run, current_instance(run, z1))
                    i1 = indexin(1, r1)[1]
                    i2 = indexin(2, r1)[1]
                    @test isapprox(es[i1], ah / (ah + bh))
                    @test isapprox(es[i2], bh / (ah + bh))
                end

            end
            
        end

        @testset "Using the VE instant algorithm" begin
            @testset "Basic" begin
                v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
                v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
                net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
                runtime = Runtime(net)
                default_initializer(runtime)
                alg = VE([v1,v2])
                infer(alg, runtime)
                i1 = current_instance(runtime, v1)
                i2 = current_instance(runtime, v2)
                @test isapprox(probability(alg, runtime, i1, :a), 0.1)
                @test isapprox(probability(alg, runtime, i1, :b), 0.9)
                @test isapprox(probability(alg, runtime, i2, 1), 0.1 * 0.2 + 0.9 * 0.3)
                @test isapprox(probability(alg, runtime, i2, 2), 0.1 * 0.8 + 0.9 * 0.7)
            end
            
            @testset "With placeholder" begin
                p1 = Placeholder{Symbol}(:p1)
                v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
                net = InstantNetwork(Variable[v2], VariableGraph(v2 => [p1]), Placeholder[p1])
                runtime = Runtime(net)
                default_initializer(runtime, 10, Dict(p1.name => Cat([:a,:b], [0.1, 0.9])))
                alg = VE([v2])
                infer(alg, runtime)
                i2 = current_instance(runtime, v2)
                @test isapprox(probability(alg, runtime, i2, 1), 0.1 * 0.2 + 0.9 * 0.3)
                @test isapprox(probability(alg, runtime, i2, 2), 0.1 * 0.8 + 0.9 * 0.7)
            end
    
            @testset "With evidence" begin
                v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
                v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
                net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
                runtime = Runtime(net)
                default_initializer(runtime)
                alg = VE([v1,v2])
                infer(alg, runtime, Dict{Symbol, Score}(:v2 => HardScore(2)))
                i1 = current_instance(runtime, v1)
                i2 = current_instance(runtime, v2)
                p1 = 0.1 * 0.8
                p2 = 0.9 * 0.7
                z = p1 + p2
                @test isapprox(probability(alg, runtime, i1, :a), p1 / z)
                @test isapprox(probability(alg, runtime, i1, :b), p2 / z)
                @test isapprox(probability(alg, runtime, i2, 1), 0.0)
                @test isapprox(probability(alg, runtime, i2, 2), 1.0)
            end
        end  
          
    end

end
