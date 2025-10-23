using Test
# using Profile, ProfileView, Gtk4

using Scruff
using Scruff.Utils
using Scruff.RTUtils
using Scruff.Models
using Scruff.SFuncs
using Scruff.Algorithms
using Scruff.Operators
import Scruff: make_initial, make_transition


@testset "Filtering" begin

    struct Tree
        x :: Int
        left :: Union{Nothing, Tree}
        right :: Union{Nothing, Tree}
        Tree(x) = new(x, nothing, nothing)
        Tree(x,l,r) = new(x, l, r)
    end

    struct Model1 <: VariableTimeModel{Tuple{}, Tuple{}, Tree} end
    global make_initial(::Model1, t) = Constant(Tree(0))
    global make_transition(::Model1, parts, t) = Constant(Tree(t))
    struct Model2 <: VariableTimeModel{Tuple{}, Tuple{Tree, Tree}, Tree} end
    global make_initial(::Model2, t) = Constant(Tree(0))
    global make_transition(::Model2, parts, t) = Det(Tuple{Tree, Tree}, Tree, (l,r) -> Tree(t, l, r))

    @testset "Window utilities" begin
        
        @testset "construct an instant network from instances" begin
            usf = Constant(1)
            umodel = HomogeneousModel(usf, usf)
            uvar = umodel(:u)
            vsf = DiscreteCPT([:a,:b], Dict((1,) => [0.5, 0.5]))
            vmodel = HomogeneousModel(vsf, vsf)
            vvar = vmodel(:v)
            ph = Placeholder{Int}(:p)
            dynnet = DynamicNetwork(Variable[uvar,vvar], VariableGraph(vvar => [uvar]), 
                VariableGraph(vvar => [uvar]), VariableParentTimeOffset(), Placeholder[ph], Placeholder[ph])
            dynrun = Runtime(dynnet)
            inst1 = instantiate!(dynrun, ph, 1)
            # this instance won't be used to create the instant network
            instantiate!(dynrun, uvar, 1) 
            inst2 = instantiate!(dynrun, uvar, 2)
            inst3 = instantiate!(dynrun, vvar, 3)
            inst4 = instantiate!(dynrun, uvar, 4) 
            set_value!(dynrun, inst1, :x, 1)
            set_value!(dynrun, inst2, :x, 2)
            set_value!(dynrun, inst3, :x, 3)
            set_value!(dynrun, inst4, :x, 4)

            run = instant_runtime_from_instances(dynrun, [inst3, inst1, inst2, inst4])
            net = get_network(run)
            phs = get_placeholders(net)
            vars = get_variables(net)
            p1 = Placeholder{Int}(:p_1)
            @test phs == [p1]
            @test length(vars) == 3
            varnames = [var.name for var in vars]
            i2 = findfirst(x -> x == :u_2, varnames)
            i3 = findfirst(x -> x == :v_3, varnames)
            i4 = findfirst(x -> x == :u_4, varnames)
            @test !isnothing(i2)
            @test !isnothing(i3)
            @test !isnothing(i4)
            v2 = vars[i2]
            v3 = vars[i3] 
            v4 = vars[i4]
            @test v2.model isa SimpleModel
            @test v3.model isa SimpleModel
            @test v4.model isa SimpleModel
            @test v2.model.sf == usf
            @test v3.model.sf == vsf
            @test v4.model.sf == usf
            @test get_parents(net, p1) == Node[]
            @test get_parents(net, v2) == Node[]
            @test get_parents(net, v3) == Node[v2]
            @test get_parents(net, v4) == Node[]
            @test has_instance(run, p1)
            @test has_instance(run, v2)
            @test has_instance(run, v3)
            @test has_instance(run, v4)
            i1 = current_instance(run, p1)
            i2 = current_instance(run, v2)
            i3 = current_instance(run, v3)
            i4 = current_instance(run, v4)
            @test get_value(run, i1, :x) == 1
            @test get_value(run, i2, :x) == 2
            @test get_value(run, i3, :x) == 3
            @test get_value(run, i4, :x) == 4
        end

        @testset "restore values in the dynamic network" begin
            usf = Constant(1)
            umodel = HomogeneousModel(usf, usf)
            uvar = umodel(:u)
            vsf = DiscreteCPT([:a,:b], Dict((1,) => [0.5, 0.5]))
            vmodel = HomogeneousModel(vsf, vsf)
            vvar = vmodel(:v)
            ph = Placeholder{Int}(:p)
            dynnet = DynamicNetwork(Variable[uvar,vvar], VariableGraph(vvar => [uvar]), 
                VariableGraph(vvar => [uvar]), VariableParentTimeOffset(), Placeholder[ph], Placeholder[ph])
            dynrun = Runtime(dynnet)
            dyninst1 = instantiate!(dynrun, ph, 1)
            # this instance won't be used to create the instant network
            instantiate!(dynrun, uvar, 1) 
            dyninst2 = instantiate!(dynrun, uvar, 2)
            dyninst3 = instantiate!(dynrun, vvar, 3)
            dyninst4 = instantiate!(dynrun, uvar, 4) 
            set_value!(dynrun, dyninst1, :x, 1)
            set_value!(dynrun, dyninst2, :x, 2)
            set_value!(dynrun, dyninst3, :x, 3)
            set_value!(dynrun, dyninst4, :x, 4)

            instrun = instant_runtime_from_instances(dynrun, 
                [dyninst3, dyninst1, dyninst2, dyninst4])
            instnet = get_network(instrun)
            phs = get_placeholders(instnet)
            instvars = get_variables(instnet)
            instp1 = Placeholder{Int}(:p_1)
            varnames = [var.name for var in instvars]
            i2 = findfirst(x -> x == :u_2, varnames)
            i3 = findfirst(x -> x == :v_3, varnames)
            i4 = findfirst(x -> x == :u_4, varnames)
            instv2 = instvars[i2]
            instv3 = instvars[i3] 
            instv4 = instvars[i4]

            instinst1 = current_instance(instrun, instp1)
            instinst2 = current_instance(instrun, instv2)
            instinst3 = current_instance(instrun, instv3)
            instinst4 = current_instance(instrun, instv4)
            set_value!(instrun, instinst1, :x, -1)
            set_value!(instrun, instinst2, :x, -2)
            set_value!(instrun, instinst3, :x, -3)
            set_value!(instrun, instinst4, :x, -4)
            retrieve_values_from_instant_runtime!(dynrun, instrun)
            @test get_value(dynrun, dyninst1, :x) == -1
            @test get_value(dynrun, dyninst2, :x) == -2
            @test get_value(dynrun, dyninst3, :x) == -3
            @test get_value(dynrun, dyninst4, :x) == -4
        end

        @testset "make an initial instant network" begin
            ph = Placeholder{Int}(:p)
            usf = DiscreteCPT([1], Dict((0,) => [1.0]))
            umodel = HomogeneousModel(usf, usf)
            uvar = umodel(:u)
            vsf = DiscreteCPT([:a,:b], Dict((1,) => [0.5, 0.5]))
            vmodel = HomogeneousModel(vsf, vsf)
            vvar = vmodel(:v)
            dynnet = DynamicNetwork(Variable[uvar,vvar], VariableGraph(uvar => [ph], vvar => [uvar]), 
                VariableGraph(uvar => [ph], vvar => [uvar]), VariableParentTimeOffset(), Placeholder[ph], Placeholder[ph])
            dynrun = Runtime(dynnet)
            ensure_all!(dynrun)
            instrun = initial_instant_runtime(dynrun)
            instnet = get_network(instrun)
            phnames = map(get_name, get_placeholders(instnet))
            @test length(phnames) == 1
            @test :p_0 in phnames
            varnames = map(get_name, get_variables(instnet))
            @test length(varnames) == 2
            @test :u_0 in varnames
            @test :v_0 in varnames
            p_0 = get_node(instnet, :p_0)
            u_0 = get_node(instnet, :u_0)
            v_0 = get_node(instnet, :v_0)
            @test p_0 isa Placeholder
            @test u_0 isa Variable
            @test v_0 isa Variable
            @test has_instance(instrun, p_0)
            @test has_instance(instrun, u_0)
            @test has_instance(instrun, v_0)
            @test get_time(current_instance(instrun, p_0)) == 0
            @test get_time(current_instance(instrun, u_0)) == 0
            @test get_time(current_instance(instrun, v_0)) == 0
        end
        
    end 
       
    @testset "Particle filter" begin
        
        @testset "Synchronous" begin
            
            @testset "Initialization step" begin
                c = Cat([1,2], [0.1, 0.9])
                d = DiscreteCPT([:a, :b], Dict((1,) => [0.2, 0.8], (2,) => [0.3, 0.7]))
                m1 = HomogeneousModel(c, c)
                m2 = HomogeneousModel(d, d)
                v1 = m1(:v1)
                v2 = m2(:v2)
                net = DynamicNetwork(Variable[v1, v2], VariableGraph(v2 => [v1]), VariableGraph(v2 => [v1]))
                filter = SyncPF(1000)
                runtime = Runtime(net)
                init_filter(filter, runtime)
                @test isapprox(probability(filter, runtime, v1, 1), 0.1; atol = 0.05)
                # Can't currently answer joint probability queries
                # @test isapprox(probability(filter, runtime, Queryable[v1,v2], x -> x[1] == 1 && x[2] == :a), 0.1 * 0.2; atol = 0.05)
            end

            @testset "Filter step" begin
                p101 = 0.1
                p102 = 0.9
                p20a = 0.5
                p20b = 0.5
                # Observe v21 = :a
                prior111 = p101 * 0.2 + p102 * 0.3
                prior112 = p101 * 0.8 + p102 * 0.7
                q111 = prior111 * 0.4
                q112 = prior112 * 0.9
                post111 = q111 / (q111 + q112)
                post112 = q112 / (q111 + q112)
                # Observe v22 = :b
                prior121 = post111 * 0.2 + post112 * 0.3
                prior122 = post111 * 0.8 + post112 * 0.7
                q121 = prior121 * 0.6
                q122 = prior122 * 0.1
                post121 = q121 / (q121 + q122)
                post122 = q122 / (q121 + q122)

                c1 = Cat([1,2], [0.1, 0.9])
                d1 = DiscreteCPT([1, :2], Dict((1,) => [0.2, 0.8], (2,) => [0.3, 0.7]))
                c2 = Cat([:a,:b], [0.5, 0.5])
                d2 = DiscreteCPT([:a, :b], Dict((1,) => [0.4, 0.6], (2,) => [0.9, 0.1]))
                m1 = HomogeneousModel(c1, d1)
                m2 = HomogeneousModel(c2, d2)
                v1 = m1(:v1)
                v2 = m2(:v2)
                vars = Variable[v1, v2]
                net = DynamicNetwork(vars, VariableGraph(), VariableGraph(v1 => [v1], v2 => [v1]))
                filter = SyncPF(1000)
                runtime = Runtime(net)
                init_filter(filter, runtime)
                @test isapprox(probability(filter, runtime, v1, 1), p101; atol = 0.05)
                @test isapprox(probability(filter, runtime, v1, 2), p102; atol = 0.05)
                @test isapprox(probability(filter, runtime, v2, :a), p20a; atol = 0.05)
                @test isapprox(probability(filter, runtime, v2, :b), p20b; atol = 0.05)
                filter_step(filter, runtime, vars, 1, Dict{Symbol, Score}(:v2 => HardScore(:a)))
                @test isapprox(probability(filter, runtime, v1, 1), post111; atol = 0.05)
                @test isapprox(probability(filter, runtime, v1, 2), post112; atol = 0.05)
                @test isapprox(probability(filter, runtime, v2, :a), 1.0; atol = 0.05)
                @test isapprox(probability(filter, runtime, v2, :b), 0.0; atol = 0.05)
                filter_step(filter, runtime, vars, 2, Dict{Symbol, Score}(:v2 => HardScore(:b)))
                @test isapprox(probability(filter, runtime, v1, 1), post121; atol = 0.05)
                @test isapprox(probability(filter, runtime, v1, 2), post122; atol = 0.05)
                @test isapprox(probability(filter, runtime, v2, :a), 0.0; atol = 0.05)
                @test isapprox(probability(filter, runtime, v2, :b), 1.0; atol = 0.05)
            end
            
        end

        @testset "Asynchronous PF" begin
            # These tests are designed to test whether variables are instantiated correctly
            # We use a four-node network v1 -> v2 -> v3 -> v4. Each variable also depends on its own previous state.
            # To keep track of instantiation, each sfunc is a constant whose value represents the instantiation pattern as a tree.
            # Instantiations happen at times 2, 3, and 5
            # We consider three orders of instantiation:
            # 1) v2, v3, v4 - no extra instances should be created
            # 2) v3, v2, v4 - when v4 is instantiated, the coherent PF should also instantiate v3
            # 3) v3, v1, v4 - when v4 is instantiated, the coherent PF should also instantiate v2 and v3 - difficult because it has to recognize an ancestor
            v1 = Model1()(:v1)
            v2 = Model2()(:v2)
            v3 = Model2()(:v3)
            v4 = Model2()(:v4)
            net = DynamicNetwork(Variable[v1,v2,v3,v4], VariableGraph(), VariableGraph(v2 => [v2, v1], v3 => [v3, v2], v4 => [v4, v3]))
            noev = Dict{Symbol, Score}()
    
            @testset "Non-coherent PF" begin
                @testset "Order 2-3-4" begin
                    filter = AsyncPF(10, 10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v2], 2, noev)
                    filter_step(filter, runtime, Variable[v3], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t22 = Tree(2, t0, t0)
                    t33 = Tree(3, t0, t22)
                    t45 = Tree(5, t0, t33)
                    @test probability(filter, runtime, v1, t0) == 1.0
                    @test probability(filter, runtime, v2, t22) == 1.0
                    @test probability(filter, runtime, v3, t33) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
    
                @testset "Order 3-2-4" begin
                    filter = AsyncPF(10, 10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v3], 2, noev)
                    filter_step(filter, runtime, Variable[v2], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t32 = Tree(2, t0, t0)
                    t23 = Tree(3, t0, t0)
                    t45 = Tree(5, t0, t32)
                    @test probability(filter, runtime, v1, t0) == 1.0
                    @test probability(filter, runtime, v2, t23) == 1.0
                    @test probability(filter, runtime, v3, t32) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
    
                @testset "Order 3-1-4" begin
                    filter = AsyncPF(10, 10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v3], 2, noev)
                    filter_step(filter, runtime, Variable[v1], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t32 = Tree(2, t0, t0)
                    t13 = Tree(3, nothing, nothing)
                    t45 = Tree(5, t0, t32)
                    @test probability(filter, runtime, v1, t13) == 1.0
                    @test probability(filter, runtime, v2, t0) == 1.0
                    @test probability(filter, runtime, v3, t32) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
                
            end
    
            @testset "Coherent PF" begin
                
                @testset "Order 2-3-4" begin
                    filter = CoherentPF(10, 10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v2], 2, noev)
                    filter_step(filter, runtime, Variable[v3], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t22 = Tree(2, t0, t0)
                    t33 = Tree(3, t0, t22)
                    t45 = Tree(5, t0, t33)
                    @test probability(filter, runtime, v1, t0) == 1.0
                    @test probability(filter, runtime, v2, t22) == 1.0
                    @test probability(filter, runtime, v3, t33) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
    
                @testset "Order 3-2-4" begin
                    filter = CoherentPF(10, 10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v3], 2, noev)
                    filter_step(filter, runtime, Variable[v2], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t32 = Tree(2, t0, t0)
                    t23 = Tree(3, t0, t0)
                    t35 = Tree(5, t32, t23) # extra instance added
                    t45 = Tree(5, t0, t35)
                    @test probability(filter, runtime, v1, t0) == 1.0
                    @test probability(filter, runtime, v2, t23) == 1.0
                    @test probability(filter, runtime, v3, t35) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
    
                @testset "Order 3-1-4" begin
                    filter = CoherentPF(10, 10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v3], 2, noev)
                    filter_step(filter, runtime, Variable[v1], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t32 = Tree(2, t0, t0)
                    t13 = Tree(3, nothing, nothing)
                    t25 = Tree(5, t0, t13) # added
                    t35 = Tree(5, t32, t25) # added
                    t45 = Tree(5, t0, t35)
                    @test probability(filter, runtime, v1, t13) == 1.0
                    @test probability(filter, runtime, v2, t25) == 1.0
                    @test probability(filter, runtime, v3, t35) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
                
            end
            
        end
    
    end

    
    @testset "BP filter" begin
        
        @testset "Synchronous" begin
            
            # @testset "Initialization step" begin
            #     c = Cat([1,2], [0.1, 0.9])
            #     d = DiscreteCPT([:a, :b], Dict((1,) => [0.2, 0.8], (2,) => [0.3, 0.7]))
            #     m1 = HomogeneousModel(c, c)
            #     m2 = HomogeneousModel(d, d)
            #     v1 = m1(:v1)
            #     v2 = m2(:v2)
            #     net = DynamicNetwork(Variable[v1, v2], VariableGraph(v2 => [v1]), VariableGraph(v2 => [v1]))
            #     filter = SyncBP()
            #     runtime = Runtime(net)
            #     init_filter(filter, runtime)
            #     @test isapprox(probability(filter, runtime, v1, 1), 0.1; atol = 0.05)
            #     # Can't currently answer joint probability queries
            #     # @test isapprox(probability(filter, runtime, Queryable[v1,v2], x -> x[1] == 1 && x[2] == :a), 0.1 * 0.2; atol = 0.05)
            # end

            @testset "Filter step" begin
                
                p101 = 0.1
                p102 = 0.9
                p20a = 0.5
                p20b = 0.5
                # Observe v21 = :a
                prior111 = p101 * 0.2 + p102 * 0.3
                prior112 = p101 * 0.8 + p102 * 0.7
                q111 = prior111 * 0.4
                q112 = prior112 * 0.9
                post111 = q111 / (q111 + q112)
                post112 = q112 / (q111 + q112)
                # Observe v22 = :b
                prior121 = post111 * 0.2 + post112 * 0.3
                prior122 = post111 * 0.8 + post112 * 0.7
                q121 = prior121 * 0.6
                q122 = prior122 * 0.1
                post121 = q121 / (q121 + q122)
                post122 = q122 / (q121 + q122)

                @testset "Without evidence" begin
                    c1 = Cat([1,2], [0.1, 0.9])
                    d1 = DiscreteCPT([1, :2], Dict((1,) => [0.2, 0.8], (2,) => [0.3, 0.7]))
                    c2 = Cat([:a,:b], [0.5, 0.5])
                    d2 = DiscreteCPT([:a, :b], Dict((1,) => [0.4, 0.6], (2,) => [0.9, 0.1]))
                    m1 = HomogeneousModel(c1, d1)
                    m2 = HomogeneousModel(c2, d2)
                    v1 = m1(:v1)
                    v2 = m2(:v2)
                    vars = Variable[v1, v2]
                    net = DynamicNetwork(vars, VariableGraph(), VariableGraph(v1 => [v1], v2 => [v1]))
                    filter = SyncBP()
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    @test isapprox(probability(filter, runtime, v1, 1), p101; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v1, 2), p102; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v2, :a), p20a; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v2, :b), p20b; atol = 0.05)
                    filter_step(filter, runtime, vars, 1, Dict{Symbol, Score}(:v2 => HardScore(:a)))
                    @test isapprox(probability(filter, runtime, v1, 1), post111; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v1, 2), post112; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v2, :a), 1.0; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v2, :b), 0.0; atol = 0.05)
                    filter_step(filter, runtime, vars, 2, Dict{Symbol, Score}(:v2 => HardScore(:b)))
                    @test isapprox(probability(filter, runtime, v1, 1), post121; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v1, 2), post122; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v2, :a), 0.0; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v2, :b), 1.0; atol = 0.05)
                end
                
            end
            
        end

        @testset "Asynchronous BP" begin
            # These tests are designed to test whether variables are instantiated correctly
            # We use a four-node network v1 -> v2 -> v3 -> v4. Each variable also depends on its own previous state.
            # To keep track of instantiation, each sfunc is a constant whose value represents the instantiation pattern as a tree.
            # Instantiations happen at times 2, 3, and 5
            # We consider three orders of instantiation:
            # 1) v2, v3, v4 - no extra instances should be created
            # 2) v3, v2, v4 - when v4 is instantiated, the coherent PF should also instantiate v3
            # 3) v3, v1, v4 - when v4 is instantiated, the coherent PF should also instantiate v2 and v3 - difficult because it has to recognize an ancestor
    
            v1 = Model1()(:v1)
            v2 = Model2()(:v2)
            v3 = Model2()(:v3)
            v4 = Model2()(:v4)
            net = DynamicNetwork(Variable[v1,v2,v3,v4], VariableGraph(), VariableGraph(v2 => [v2, v1], v3 => [v3, v2], v4 => [v4, v3]))
            noev = Dict{Symbol, Score}()
    
            @testset "Non-coherent PF" begin
                @testset "Order 2-3-4" begin
                    filter = AsyncBP(10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v2], 2, noev)
                    filter_step(filter, runtime, Variable[v3], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t22 = Tree(2, t0, t0)
                    t33 = Tree(3, t0, t22)
                    t45 = Tree(5, t0, t33)
                    @test probability(filter, runtime, v1, t0) == 1.0
                    @test probability(filter, runtime, v2, t22) == 1.0
                    @test probability(filter, runtime, v3, t33) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
    
                @testset "Order 3-2-4" begin
                    filter = AsyncBP(10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v3], 2, noev)
                    filter_step(filter, runtime, Variable[v2], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t32 = Tree(2, t0, t0)
                    t23 = Tree(3, t0, t0)
                    t45 = Tree(5, t0, t32)
                    @test probability(filter, runtime, v1, t0) == 1.0
                    @test probability(filter, runtime, v2, t23) == 1.0
                    @test probability(filter, runtime, v3, t32) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
    
                @testset "Order 3-1-4" begin
                    filter = AsyncBP(10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v3], 2, noev)
                    filter_step(filter, runtime, Variable[v1], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t32 = Tree(2, t0, t0)
                    t13 = Tree(3, nothing, nothing)
                    t45 = Tree(5, t0, t32)
                    @test probability(filter, runtime, v1, t13) == 1.0
                    @test probability(filter, runtime, v2, t0) == 1.0
                    @test probability(filter, runtime, v3, t32) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
                
            end
    
            @testset "Coherent BP" begin
                
                @testset "Order 2-3-4" begin
                    filter = CoherentBP(10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v2], 2, noev)
                    filter_step(filter, runtime, Variable[v3], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t22 = Tree(2, t0, t0)
                    t33 = Tree(3, t0, t22)
                    t45 = Tree(5, t0, t33)
                    @test probability(filter, runtime, v1, t0) == 1.0
                    @test probability(filter, runtime, v2, t22) == 1.0
                    @test probability(filter, runtime, v3, t33) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
    
                @testset "Order 3-2-4" begin
                    filter = CoherentBP(10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v3], 2, noev)
                    filter_step(filter, runtime, Variable[v2], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t32 = Tree(2, t0, t0)
                    t23 = Tree(3, t0, t0)
                    t35 = Tree(5, t32, t23) # extra instance added
                    t45 = Tree(5, t0, t35)
                    @test probability(filter, runtime, v1, t0) == 1.0
                    @test probability(filter, runtime, v2, t23) == 1.0
                    @test probability(filter, runtime, v3, t35) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
    
                @testset "Order 3-1-4" begin
                    filter = CoherentBP(10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v3], 2, noev)
                    filter_step(filter, runtime, Variable[v1], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t32 = Tree(2, t0, t0)
                    t13 = Tree(3, nothing, nothing)
                    t25 = Tree(5, t0, t13) # added
                    t35 = Tree(5, t32, t25) # added
                    t45 = Tree(5, t0, t35)
                    @test probability(filter, runtime, v1, t13) == 1.0
                    @test probability(filter, runtime, v2, t25) == 1.0
                    @test probability(filter, runtime, v3, t35) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
                
            end
            
        end
    
    end
   
    @testset "Loopy filter" begin
        
        @testset "Synchronous" begin
            
            @testset "Initialization step" begin
                c = Cat([1,2], [0.1, 0.9])
                d = DiscreteCPT([:a, :b], Dict((1,) => [0.2, 0.8], (2,) => [0.3, 0.7]))
                m1 = HomogeneousModel(c, c)
                m2 = HomogeneousModel(d, d)
                v1 = m1(:v1)
                v2 = m2(:v2)
                net = DynamicNetwork(Variable[v1, v2], VariableGraph(v2 => [v1]), VariableGraph(v2 => [v1]))
                filter = SyncLoopy()
                runtime = Runtime(net)
                init_filter(filter, runtime)
                @test isapprox(probability(filter, runtime, v1, 1), 0.1; atol = 0.05)
                # Can't currently answer joint probability queries
                # @test isapprox(probability(filter, runtime, Queryable[v1,v2], x -> x[1] == 1 && x[2] == :a), 0.1 * 0.2; atol = 0.05)
            end

            @testset "Filter step" begin
                @testset "Without evidence" begin
                    p101 = 0.1
                    p102 = 0.9
                    p20a = 0.5
                    p20b = 0.5
                    # Observe v21 = :a
                    prior111 = p101 * 0.2 + p102 * 0.3
                    prior112 = p101 * 0.8 + p102 * 0.7
                    q111 = prior111 * 0.4
                    q112 = prior112 * 0.9
                    post111 = q111 / (q111 + q112)
                    post112 = q112 / (q111 + q112)
                    # Observe v22 = :b
                    prior121 = post111 * 0.2 + post112 * 0.3
                    prior122 = post111 * 0.8 + post112 * 0.7
                    q121 = prior121 * 0.6
                    q122 = prior122 * 0.1
                    post121 = q121 / (q121 + q122)
                    post122 = q122 / (q121 + q122)

                    c1 = Cat([1,2], [0.1, 0.9])
                    d1 = DiscreteCPT([1, :2], Dict((1,) => [0.2, 0.8], (2,) => [0.3, 0.7]))
                    c2 = Cat([:a,:b], [0.5, 0.5])
                    d2 = DiscreteCPT([:a, :b], Dict((1,) => [0.4, 0.6], (2,) => [0.9, 0.1]))
                    m1 = HomogeneousModel(c1, d1)
                    m2 = HomogeneousModel(c2, d2)
                    v1 = m1(:v1)
                    v2 = m2(:v2)
                    vars = Variable[v1, v2]
                    net = DynamicNetwork(vars, VariableGraph(), VariableGraph(v1 => [v1], v2 => [v1]))
                    filter = SyncLoopy()
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    @test isapprox(probability(filter, runtime, v1, 1), p101; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v1, 2), p102; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v2, :a), p20a; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v2, :b), p20b; atol = 0.05)
                    filter_step(filter, runtime, vars, 1, Dict{Symbol, Score}(:v2 => HardScore(:a)))
                    @test isapprox(probability(filter, runtime, v1, 1), post111; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v1, 2), post112; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v2, :a), 1.0; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v2, :b), 0.0; atol = 0.05)
                    filter_step(filter, runtime, vars, 2, Dict{Symbol, Score}(:v2 => HardScore(:b)))
                    @test isapprox(probability(filter, runtime, v1, 1), post121; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v1, 2), post122; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v2, :a), 0.0; atol = 0.05)
                    @test isapprox(probability(filter, runtime, v2, :b), 1.0; atol = 0.05)
                end
            end
        end

        @testset "Asynchronous Loopy" begin
            # These tests are designed to test whether variables are instantiated correctly
            # We use a four-node network v1 -> v2 -> v3 -> v4. Each variable also depends on its own previous state.
            # To keep track of instantiation, each sfunc is a constant whose value represents the instantiation pattern as a tree.
            # Instantiations happen at times 2, 3, and 5
            # We consider three orders of instantiation:
            # 1) v2, v3, v4 - no extra instances should be created
            # 2) v3, v2, v4 - when v4 is instantiated, the coherent PF should also instantiate v3
            # 3) v3, v1, v4 - when v4 is instantiated, the coherent PF should also instantiate v2 and v3 - difficult because it has to recognize an ancestor
            v1 = Model1()(:v1)
            v2 = Model2()(:v2)
            v3 = Model2()(:v3)
            v4 = Model2()(:v4)
            net = DynamicNetwork(Variable[v1,v2,v3,v4], VariableGraph(), VariableGraph(v2 => [v2, v1], v3 => [v3, v2], v4 => [v4, v3]))
            noev = Dict{Symbol, Score}()
    
            @testset "Non-coherent Loopy" begin
                @testset "Order 2-3-4" begin
                    filter = AsyncLoopy(10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v2], 2, noev)
                    filter_step(filter, runtime, Variable[v3], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t22 = Tree(2, t0, t0)
                    t33 = Tree(3, t0, t22)
                    t45 = Tree(5, t0, t33)
                    @test probability(filter, runtime, v1, t0) == 1.0
                    @test probability(filter, runtime, v2, t22) == 1.0
                    @test probability(filter, runtime, v3, t33) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
    
                @testset "Order 3-2-4" begin
                    filter = AsyncLoopy(10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v3], 2, noev)
                    filter_step(filter, runtime, Variable[v2], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t32 = Tree(2, t0, t0)
                    t23 = Tree(3, t0, t0)
                    t45 = Tree(5, t0, t32)
                    @test probability(filter, runtime, v1, t0) == 1.0
                    @test probability(filter, runtime, v2, t23) == 1.0
                    @test probability(filter, runtime, v3, t32) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
    
                @testset "Order 3-1-4" begin
                    filter = AsyncLoopy(10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v3], 2, noev)
                    filter_step(filter, runtime, Variable[v1], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t32 = Tree(2, t0, t0)
                    t13 = Tree(3, nothing, nothing)
                    t45 = Tree(5, t0, t32)
                    @test probability(filter, runtime, v1, t13) == 1.0
                    @test probability(filter, runtime, v2, t0) == 1.0
                    @test probability(filter, runtime, v3, t32) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
                
            end
    
            @testset "Coherent Loopy" begin
                
                @testset "Order 2-3-4" begin
                    filter = CoherentLoopy(10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v2], 2, noev)
                    filter_step(filter, runtime, Variable[v3], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t22 = Tree(2, t0, t0)
                    t33 = Tree(3, t0, t22)
                    t45 = Tree(5, t0, t33)
                    @test probability(filter, runtime, v1, t0) == 1.0
                    @test probability(filter, runtime, v2, t22) == 1.0
                    @test probability(filter, runtime, v3, t33) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
    
                @testset "Order 3-2-4" begin
                    filter = CoherentLoopy(10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v3], 2, noev)
                    filter_step(filter, runtime, Variable[v2], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t32 = Tree(2, t0, t0)
                    t23 = Tree(3, t0, t0)
                    t35 = Tree(5, t32, t23) # extra instance added
                    t45 = Tree(5, t0, t35)
                    @test probability(filter, runtime, v1, t0) == 1.0
                    @test probability(filter, runtime, v2, t23) == 1.0
                    @test probability(filter, runtime, v3, t35) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
    
                @testset "Order 3-1-4" begin
                    filter = CoherentLoopy(10, Int)
                    runtime = Runtime(net)
                    init_filter(filter, runtime)
                    filter_step(filter, runtime, Variable[v3], 2, noev)
                    filter_step(filter, runtime, Variable[v1], 3, noev)
                    filter_step(filter, runtime, Variable[v4], 5, noev)
                    t0 = Tree(0, nothing, nothing)
                    t32 = Tree(2, t0, t0)
                    t13 = Tree(3, nothing, nothing)
                    t25 = Tree(5, t0, t13) # added
                    t35 = Tree(5, t32, t25) # added
                    t45 = Tree(5, t0, t35)
                    @test probability(filter, runtime, v1, t13) == 1.0
                    @test probability(filter, runtime, v2, t25) == 1.0
                    @test probability(filter, runtime, v3, t35) == 1.0
                    @test probability(filter, runtime, v4, t45) == 1.0
                end
                
            end
            
        end

    end
       
    @testset "Range limited filter" begin
        
        @testset "Runs discrete without complaining" begin
            cat1 = Cat([1,2,3,4], [0.1, 0.2, 0.3, 0.4])
            cat2 = Cat([6,7,8], [0.2, 0.3, 0.5])
            chain1 = Chain(Tuple{Int}, Int, x -> Constant(x[1] + 1))
            chain2 = Chain(Tuple{Int}, Int, x -> Constant(x[1] * 2))
            v1 = HomogeneousModel(cat1, chain1)(:v1)
            v2 = HomogeneousModel(cat2, chain2)(:v2)
            net = DynamicNetwork([v1, v2], VariableGraph(), VariableGraph(v1 => [v1], v2 => [v2]))

            @testset "with no limits" begin
                limits = Dict{Symbol, Integer}()
                filter = RangeLimited(SyncBP(), limits)
                variables = [v1, v2]
                runtime = Runtime(net)
                init_filter(filter, runtime)
                filter_step(filter, runtime, variables, 1, Dict{Symbol, Score}()) 
                filter_step(filter, runtime, variables, 2, Dict{Symbol, Score}()) 
                @test true
            end

            @testset "with limits greater than range" begin
                limits = Dict{Symbol, Integer}(:v1 => 6, :v2 => 8)
                filter = RangeLimited(SyncBP(), limits)
                variables = [v1, v2]
                runtime = Runtime(net)
                init_filter(filter, runtime)
                filter_step(filter, runtime, variables, 1, Dict{Symbol, Score}()) 
                filter_step(filter, runtime, variables, 2, Dict{Symbol, Score}()) 
                @test true
            end

            @testset "with limits less than range" begin
                limits = Dict{Symbol, Integer}(:v1 => 3, :v2 => 2)
                filter = RangeLimited(SyncBP(), limits)
                variables = [v1, v2]
                runtime = Runtime(net)
                init_filter(filter, runtime)
                filter_step(filter, runtime, variables, 1, Dict{Symbol, Score}()) 
                filter_step(filter, runtime, variables, 2, Dict{Symbol, Score}()) 
                @test true
            end

            @testset "with partial limits" begin
                limits = Dict{Symbol, Integer}(:v1 => 3)
                filter = RangeLimited(SyncBP(), limits)
                variables = [v1, v2]
                runtime = Runtime(net)
                init_filter(filter, runtime)
                filter_step(filter, runtime, variables, 1, Dict{Symbol, Score}()) 
                filter_step(filter, runtime, variables, 2, Dict{Symbol, Score}()) 
                @test true
            end
        end 

        @testset "Runs continuous without complaining" begin
            norm = Normal(0.0, 1.0)
            lg = LinearGaussian((1.0,), 2.0, 1.0)
            v = HomogeneousModel(norm, lg)(:v)
            net = DynamicNetwork([v], VariableGraph(), VariableGraph(v => [v]))

            @testset "with no limits" begin
                limits = Dict{Symbol, Integer}()
                filter = RangeLimited(SyncBP(), limits)
                variables = [v]
                runtime = Runtime(net)
                init_filter(filter, runtime)
                filter_step(filter, runtime, variables, 1, Dict{Symbol, Score}()) 
                filter_step(filter, runtime, variables, 2, Dict{Symbol, Score}()) 
                @test true
            end

            @testset "with limits" begin
                limits = Dict{Symbol, Integer}(:v => 3)
                filter = RangeLimited(SyncBP(), limits)
                variables = [v]
                runtime = Runtime(net)
                init_filter(filter, runtime)
                filter_step(filter, runtime, variables, 1, Dict{Symbol, Score}()) 
                filter_step(filter, runtime, variables, 2, Dict{Symbol, Score}()) 
                @test true
            end

        end 
        
        @testset "Does not limit the range of variables when there are no limits" begin
            norm = Normal(0.0, 1.0)
            lg = LinearGaussian((1.0,), 2.0, 1.0)
            v = HomogeneousModel(norm, lg)(:v)
            net = DynamicNetwork([v], VariableGraph(), VariableGraph(v => [v]))

            limits = Dict{Symbol, Integer}()
            filter = RangeLimited(SyncBP(), limits)
            variables = [v]
            runtime = Runtime(net)
            init_filter(filter, runtime)
            filter_step(filter, runtime, variables, 1, Dict{Symbol, Score}()) 
            filter_step(filter, runtime, variables, 2, Dict{Symbol, Score}()) 

            instance = current_instance(runtime, v)
            bel = get_belief(runtime, instance)
            @test length(support(bel, (), 1000, Float64[])) > 3
        end

        @testset "Appropriately limits the range of continuous variables when there are limits" begin
            norm = Normal(0.0, 1.0)
            lg = LinearGaussian((1.0,), 2.0, 1.0)
            v = HomogeneousModel(norm, lg)(:v)
            net = DynamicNetwork([v], VariableGraph(), VariableGraph(v => [v]))

            limits = Dict{Symbol, Integer}(:v => 3)
            filter = RangeLimited(SyncBP(), limits)
            variables = [v]
            runtime = Runtime(net)
            init_filter(filter, runtime)
            filter_step(filter, runtime, variables, 1, Dict{Symbol, Score}()) 
            filter_step(filter, runtime, variables, 2, Dict{Symbol, Score}()) 

            instance = current_instance(runtime, v)
            bel = get_belief(runtime, instance)
            @test length(support(bel, (), 1000, Float64[])) <= 3
        end

        @testset "Appropriately limits the range of discrete variables when there are limits" begin
            cat1 = Cat([1,2,3,4], [0.1, 0.2, 0.3, 0.4])
            cat2 = Cat([6,7,8], [0.2, 0.3, 0.5])
            chain1 = Chain(Tuple{Int}, Int, x -> Constant(x[1] + 1))
            chain2 = Chain(Tuple{Int}, Int, x -> Constant(x[1] * 2))
            v1 = HomogeneousModel(cat1, chain1)(:v1)
            v2 = HomogeneousModel(cat2, chain2)(:v2)
            net = DynamicNetwork([v1, v2], VariableGraph(), VariableGraph(v1 => [v1], v2 => [v2]))

            limits = Dict{Symbol, Integer}(:v1 => 3, :v2 => 2)
            filter = RangeLimited(SyncBP(), limits)
            variables = [v1, v2]
            runtime = Runtime(net)
            init_filter(filter, runtime)
            filter_step(filter, runtime, variables, 1, Dict{Symbol, Score}()) 
            filter_step(filter, runtime, variables, 2, Dict{Symbol, Score}()) 

            instance1 = current_instance(runtime, v1)
            bel1 = get_belief(runtime, instance1)
            @test length(support(bel1, (), 1000, Int[])) <= 3
            instance2 = current_instance(runtime, v2)
            bel2 = get_belief(runtime, instance2)
            @test length(support(bel2, (), 1000, Int[])) <= 2
        end

        @testset "Produces correct beliefs with no limits - discrete" begin
            p101 = 0.1
            p102 = 0.9
            p20a = 0.5
            p20b = 0.5
            # Observe v21 = :a
            prior111 = p101 * 0.2 + p102 * 0.3
            prior112 = p101 * 0.8 + p102 * 0.7
            q111 = prior111 * 0.4
            q112 = prior112 * 0.9
            post111 = q111 / (q111 + q112)
            post112 = q112 / (q111 + q112)
            # Observe v22 = :b
            prior121 = post111 * 0.2 + post112 * 0.3
            prior122 = post111 * 0.8 + post112 * 0.7
            q121 = prior121 * 0.6
            q122 = prior122 * 0.1
            post121 = q121 / (q121 + q122)
            post122 = q122 / (q121 + q122)

            c1 = Cat([1,2], [0.1, 0.9])
            d1 = DiscreteCPT([1, :2], Dict((1,) => [0.2, 0.8], (2,) => [0.3, 0.7]))
            c2 = Cat([:a,:b], [0.5, 0.5])
            d2 = DiscreteCPT([:a, :b], Dict((1,) => [0.4, 0.6], (2,) => [0.9, 0.1]))
            m1 = HomogeneousModel(c1, d1)
            m2 = HomogeneousModel(c2, d2)
            v1 = m1(:v1)
            v2 = m2(:v2)
            vars = Variable[v1, v2]
            net = DynamicNetwork(vars, VariableGraph(), VariableGraph(v1 => [v1], v2 => [v1]))
            filter = RangeLimited(SyncBP(), Dict{Symbol, Score}())
            runtime = Runtime(net)
            init_filter(filter, runtime)
            @test isapprox(probability(filter, runtime, v1, 1), p101, atol = 0.05)
            @test isapprox(probability(filter, runtime, v1, 2), p102, atol = 0.05)
            @test isapprox(probability(filter, runtime, v2, :a), p20a, atol = 0.05)
            @test isapprox(probability(filter, runtime, v2, :b), p20b, atol = 0.05)
            filter_step(filter, runtime, vars, 1, Dict{Symbol, Score}(:v2 => HardScore(:a)))
            @test isapprox(probability(filter, runtime, v1, 1), post111, atol = 0.05)
            @test isapprox(probability(filter, runtime, v1, 2), post112, atol = 0.05)
            @test isapprox(probability(filter, runtime, v2, :a), 1.0, atol = 0.05)
            @test isapprox(probability(filter, runtime, v2, :b), 0.0, atol = 0.05)
            filter_step(filter, runtime, vars, 2, Dict{Symbol, Score}(:v2 => HardScore(:b)))
            @test isapprox(probability(filter, runtime, v1, 1), post121, atol = 0.05)
            @test isapprox(probability(filter, runtime, v1, 2), post122, atol = 0.05)
            @test isapprox(probability(filter, runtime, v2, :a), 0.0, atol = 0.05)
            @test isapprox(probability(filter, runtime, v2, :b), 1.0, atol = 0.05)

        end

        @testset "Produces correct beliefs with no limits - continuous" begin
            norm = Normal(0.0, 1.0)
            lg = LinearGaussian((1.0,), 2.0, 1.0)
            v = HomogeneousModel(norm, lg)(:v)
            net = DynamicNetwork([v], VariableGraph(), VariableGraph(v => [v]))
            # Although it is difficult to compute the exact target result, we can compare to the non-range limited version and make sure it is similar
            filter1 = SyncBP()
            filter2 = RangeLimited(SyncBP(), Dict{Symbol, Int}())
            runtime1 = Runtime(net)
            init_filter(filter1, runtime1)
            runtime2 = Runtime(net)
            init_filter(filter2, runtime2)
            @test isapprox(mean(filter2, runtime2, v), mean(filter1, runtime1, v)) # These should produce essentially the same result, so no atol
            filter_step(filter1, runtime1, [v], 1, Dict{Symbol, Score}())
            filter_step(filter2, runtime2, [v], 1, Dict{Symbol, Score}())
            @test isapprox(mean(filter2, runtime2, v), mean(filter1, runtime1, v)) 
            filter_step(filter1, runtime1, [v], 2, Dict{Symbol, Score}())
            filter_step(filter2, runtime2, [v], 2, Dict{Symbol, Score}())
            @test isapprox(mean(filter2, runtime2, v), mean(filter1, runtime1, v)) 
        end

        @testset "Produces approximately correct beliefs with limits - continuous" begin
            norm = Normal(0.0, 1.0)
            lg = LinearGaussian((1.0,), 2.0, 1.0)
            v = HomogeneousModel(norm, lg)(:v)
            net = DynamicNetwork([v], VariableGraph(), VariableGraph(v => [v]))
            # Although it is difficult to compute the exact target result, we can compare to the non-range limited version and make sure it is similar
            filter1 = SyncBP()
            filter2 = RangeLimited(SyncBP(), Dict{Symbol, Int}(:v => 10))
            runtime1 = Runtime(net)
            init_filter(filter1, runtime1)
            runtime2 = Runtime(net)
            init_filter(filter2, runtime2)
            @test isapprox(mean(filter2, runtime2, v), mean(filter1, runtime1, v), atol = 0.5) # Now these are only approximately the same
            filter_step(filter1, runtime1, [v], 1, Dict{Symbol, Score}()) # With only 5 values, this is a reasonable error if it is consistently adhered to
            filter_step(filter2, runtime2, [v], 1, Dict{Symbol, Score}())
            @test isapprox(mean(filter2, runtime2, v), mean(filter1, runtime1, v), atol = 0.5) 
            filter_step(filter1, runtime1, [v], 2, Dict{Symbol, Score}())
            filter_step(filter2, runtime2, [v], 2, Dict{Symbol, Score}())
            @test isapprox(mean(filter2, runtime2, v), mean(filter1, runtime1, v), atol = 0.5) 
        end

        @testset "Works with Async" begin
            cat1 = Cat([1,2,3,4], [0.1, 0.2, 0.3, 0.4])
            cat2 = Cat([6,7,8], [0.2, 0.3, 0.5])
            struct ThisVTM1 <: VariableTimeModel{Tuple{}, Tuple{Int}, Int} end
            struct ThisVTM2 <: VariableTimeModel{Tuple{}, Tuple{Int}, Int} end
            make_initial(::ThisVTM1, empty) = cat1 
            make_transition(::ThisVTM1, prev, t) = Constant{Int}(prev[1] + floor(t))
            make_initial(::ThisVTM2, empty) = cat2 
            make_transition(::ThisVTM2, prev, t) = Constant{Int}(prev[1] * floor(t))
            v1 = Variable(:v1, ThisVTM1())
            v2 = Variable(:v2, ThisVTM2())
            net = DynamicNetwork([v1, v2], VariableGraph(), VariableGraph(v1 => [v1], v2 => [v2]))

            limits = Dict{Symbol, Integer}(:v1 => 3, :v2 => 2)
            filter = RangeLimited(AsyncBP(1000000, Float64), limits) # We're limiting the range, so we don't give any suggestion to generating the range in the first place
            variables = [v1, v2]
            runtime = Runtime(net, 0.0)
            init_filter(filter, runtime)
            filter_step(filter, runtime, Variable[v1], 1.0, Dict{Symbol, Score}()) 
            filter_step(filter, runtime, Variable[v2], 3.5, Dict{Symbol, Score}()) 

            # Test appropriate limiting
            instance1 = current_instance(runtime, v1)
            bel1 = get_belief(runtime, instance1)
            @test length(support(bel1, (), 1000, Int[])) <= 3
            instance2 = current_instance(runtime, v2)
            bel2 = get_belief(runtime, instance2)
            @test length(support(bel2, (), 1000, Int[])) <= 2
        end

        # @testset "Profiling filtering algorithms" begin
        #     norm = Normal(0.0, 1.0)
        #     function tf(tup::Tuple{Float64})
        #         x = tup[1]
        #         d = Dict(x - 1.0 => 0.5, x * 1.01 => 0.5)
        #         Cat(d)
        #     end
        #     trans = Chain(Tuple{Float64}, Float64, tf)
        #     v = HomogeneousModel(norm, trans)(:v)
        #     net = DynamicNetwork([v], VariableGraph(), VariableGraph(v => [v]))
        #     num_iterations = 10000 # Choose num_iterations to cause range_unlimited to take a long time
        #     interval = 100

        #     # Choose 1
        #     filter = SyncBP()
        #     # filter = SyncPF(1000)
        #     # filter = RangeLimited(SyncBP(), Dict(:v => 5))
            
        #     runtime = Runtime(net)
        #     init_filter(filter, runtime)
        #     time1 = time()
        #     for i in 1:num_iterations
        #         filter_step(filter, runtime, [v], i, Dict{Symbol, Score}())
        #         if i % interval == 0
        #             # println("\nIteration", i) 
        #             time2 = time()
        #             # println("Time of last ", interval, " iterations:", time2 - time1)
        #             time1 = time2
        #         end
        #     end
        # end
    end
  

end