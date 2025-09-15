using Test

using Logging
using Scruff
using Scruff.Models
using Scruff.Utils
using Scruff.RTUtils
using Scruff.SFuncs
using Scruff.Operators

@testset "test loggers" begin
    x1 = Cat([1,2], [0.1, 0.9])()(:x1)
    x2 = Cat([1,2,3], [0.2, 0.3, 0.5])()(:x2)
    cpd2 = Dict((1,1) => [0.3, 0.7], (1,2) => [0.6, 0.4], (2,1) =>[0.4, 0.6], 
                (2,2) => [0.7, 0.3], (3,1) => [0.5, 0.5], (3,2) => [0.8, 0.2])
    x3 = DiscreteCPT([1,2], cpd2)()(:x3)
    x4 = DiscreteCPT([1,2], Dict((1,) => [0.15, 0.85], (2,) => [0.25, 0.75]))()(:x4)
    x5 = DiscreteCPT([1,2], Dict((1,) => [0.35, 0.65], (2,) => [0.45, 0.55]))()(:x5)
    x6 = DiscreteCPT([1,2], Dict((1,) => [0.65, 0.35], (2,) => [0.75, 0.25]))()(:x6)

    fivecpdnet = InstantNetwork(Variable[x1,x2,x3,x4,x5], VariableGraph(x3=>[x2,x1], x4=>[x3], x5=>[x3]))
    
    @testset "runtime logger" begin
        run = Runtime(fivecpdnet)
        default_initializer(run)
        inst1 = current_instance(run, x1)
        inst3 = current_instance(run, x3)
        inst5 = current_instance(run, x5)
        
        @test_logs(
            (:info, r"distribute_messages!"),
            (:info, r"get_node"),
            (:info, r"collect_messages"),
            (:info, r"get_variables"),
            (:info, r"set_value!"),
            (:info, r"current_instance"),
            match_mode=:any,
            trace_runtime(Scruff.Algorithms.three_pass_BP, run))
        @test_logs(
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            (:info, r"collect_messages"),
            trace_runtime(Scruff.Algorithms.three_pass_BP, run; fnamefilter=x->x == "collect_messages"))
    end

    #= This test takes too long
    @testset "log all" begin
        run = Runtime(fivecpdnet)
        default_initializer(run)
        inst1 = current_instance(run, x1)
        inst3 = current_instance(run, x3)
        inst5 = current_instance(run, x5)

        @test_logs(
            (:info, r"distribute_messages!"),
            (:info, r"has_state"),
            (:info, r"get_network"),
            (:info, r"get_state"),
            (:info, r"collect_messages"),
            (:info, r"current_time"),
            (:info, r"get_variables"),
            (:info, r"set_value!"),
            (:info, r"outgoing_pis"),
            (:info, r"get_parents"),
            (:info, r"get_children"),
            (:info, r"compute_bel"),
            (:info, r"current_instance"),
            match_mode=:any,
            trace_algorithm(Scruff.Algorithms.three_pass_BP, run))
    end
    =#

    @testset "timing logger" begin
        run = Runtime(fivecpdnet)
        default_initializer(run)
        inst1 = current_instance(run, x1)
        inst3 = current_instance(run, x3)
        inst5 = current_instance(run, x5)
        
        @test_logs(
            (:info, "three_pass_BP"),
            (:info, "outgoing_pis"),
            (:info, "get_sfunc"),
            match_mode=:any,
            time_algorithm(Scruff.Algorithms.three_pass_BP, run))
    end
end
