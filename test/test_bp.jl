import Base.timedwait
import Base.isapprox
import PrettyPrint
import Scruff.Operators.bounded_probs
import Scruff.Operators.range

using Test
using Scruff
using Scruff.Utils
using Scruff.RTUtils
using Scruff.SFuncs
using Scruff.Models
using Scruff.Operators
import Scruff.Algorithms: three_pass_BP, loopy_BP, ThreePassBP, LoopyBP, infer, probability

@testset "BP" begin

    @testset "Cat operations" begin
        x = Cat([1,2,3], [0.2, 0.3, 0.5])

        @testset "compute_pi" begin
            p = compute_pi(x, [1,2,3], (), ())
            i1 = indexin(1, p.__compiled_range)[1]
            i2 = indexin(2, p.__compiled_range)[1]
            i3 = indexin(3, p.__compiled_range)[1]
            @test p.params[i1] == 0.2
            @test p.params[i2] == 0.3
            @test p.params[i3] == 0.5
        end
    end

    @testset "DiscreteCPT operations" begin
        cpd1 = Dict((1,) => [0.1, 0.9], (2,) => [0.2, 0.8], (3,) => [0.3, 0.7])
        cpd2 = Dict((1,1) => [0.3, 0.7], (1,2) => [0.6, 0.4], (2,1) =>[0.4, 0.6],
                    (2,2) => [0.7, 0.3], (3,1) => [0.5, 0.5], (3,2) => [0.8, 0.2])
        cpd3entries = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        cpd3 = Dict{Tuple{Int, Int, Int}, Vector{Float64}}()
        for i1 = 1:2
            for i2 = 1:3
                for i3 = 1:2
                    ix = (i1-1)*6 + (i2-1)*2 + i3
                    p = cpd3entries[ix]
                    cpd3[(i1,i2,i3)] = [p, 1-p]
                end
            end
        end
        cpd4entries =
            [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
            0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        cpd4 = Dict{Tuple{Int, Int, Int, Int}, Vector{Float64}}()
        for i1 = 1:2
            for i2 = 1:3
                for i3 = 1:2
                    for i4 = 1:2
                        ix = (i1-1)*12 + (i2-1)*4 + (i3-1)*2 + i4
                        p = cpd4entries[ix]
                        cpd4[(i1,i2,i3,i4)] = [p, 1-p]
                    end
                end
            end
        end

        x1 = DiscreteCPT([1, 2], cpd1)
        x2 = DiscreteCPT([1, 2], cpd2)
        x3 = DiscreteCPT([1, 2], cpd3)
        x4 = DiscreteCPT([1, 2], cpd4)
        i1 = indexin(1, x1.__sfs[1].__compiled_range)[1]
        i2 = indexin(2, x1.__sfs[1].__compiled_range)[1]

        @testset "compute_pi" begin
            @testset "with one parent" begin
                pa = 0.2 * 0.1 + 0.3 * 0.2 + 0.5 * 0.3
                pb = 1 - pa
                pi = compute_pi(x1, [1,2], ([1,2,3],), (Cat([1,2,3], [0.2, 0.3, 0.5]),))
                @test isapprox(pi.params[i1], pa)
                @test isapprox(pi.params[i2], pb)
            end

            @testset "with two parents" begin
                # First parent is outermost (range 2).
                # Second parent is innermost (range 3).
                # If first parent probabilities are [0.1, 0.9]
                # and second parent probablity are [0.2, 0.3, 0.5]
                # we get the following probabilities over x:
                pa = 0.2 * (0.1 * 0.3 + 0.9 * 0.6) +
                    0.3 * (0.1 * 0.4 + 0.9 * 0.7) +
                    0.5 * (0.1 * 0.5 + 0.9 * 0.8)
                pb = 1 - pa
                pi = compute_pi(x2, [1,2], ([1,2,3], [1,2]), (Cat([1,2,3], [0.2, 0.3, 0.5]), Cat([1,2],[0.1, 0.9])))
                @test isapprox(pi.params[i1], pa)
                @test isapprox(pi.params[i2], pb)
            end

            @testset "with three parents" begin
                pa = 0.0
                pi1 = [0.3, 0.7]
                pi2 = [0.2, 0.3, 0.5]
                pi3 = [0.6, 0.4]
                for i = 1:12
                    ix1 = div(i-1,6) + 1
                    ix2 = div(mod(i-1,6), 2) + 1
                    ix3 = mod(i-1,2) + 1
                    parp = pi1[ix1] * pi2[ix2] * pi3[ix3]
                    pa += parp * cpd3entries[i]
                end
                pb = 1 - pa
                pi = compute_pi(x3, [1,2], ([1,2], [1,2,3], [1,2]), (Cat([1,2], pi1), Cat([1,2,3], pi2), Cat([1,2], pi3)))
                @test isapprox(pi.params[i1], pa)
                @test isapprox(pi.params[i2], pb)
            end

            @testset "with four parents" begin
                pa = 0.0
                pi1 = [0.3, 0.7]
                pi2 = [0.2, 0.3, 0.5]
                pi3 = [0.6, 0.4]
                pi4 = [0.8, 0.2]
                for i = 1:24
                    ix1 = div(i-1,12) + 1
                    ix2 = div(mod(i-1,12), 4) + 1
                    ix3 = div(mod(i-1,4), 2) + 1
                    ix4 = mod(i-1, 2) + 1
                    parp = pi1[ix1] * pi2[ix2] * pi3[ix3] * pi4[ix4]
                    pa += parp * cpd4entries[i]
                end
                pb = 1 - pa
                pi = compute_pi(x4, [1,2], ([1,2], [1,2,3], [1,2], [1,2]), (Cat([1,2], pi1), Cat([1,2,3], pi2), Cat([1,2], pi3), Cat([1,2], pi4)))
                @test isapprox(pi.params[i1], pa)
                @test isapprox(pi.params[i2], pb)
            end

        end

        @testset "send_lambda" begin
            @testset "with one parent" begin
                p1 = 0.3 * 0.1 + 0.7 * 0.9
                p2 = 0.3 * 0.2 + 0.7 * 0.8
                p3 = 0.3 * 0.3 + 0.7 * 0.7
                # l1 = send_lambda(x1, [0.3, 0.7], [1,2], ([1,2,3], [1,2]), [[0.2, 0.3, 0.5], [0.1, 0.9]], 1)
                # @test isapprox(l1, [p1, p2, p3])
                # FIXME is this a correct test?
                l1 = send_lambda(x1, SoftScore([1,2], [0.3, 0.7]), [1,2], ([1,2,3],), (Cat([1,2,3], [0.2, 0.3, 0.5]),), 1)
                @test isapprox(get_score(l1, 1), p1)
                @test isapprox(get_score(l1, 2), p2)
                @test isapprox(get_score(l1, 3), p3)
            end

            @testset "with two parents" begin
                # If the lambda for x is [0.3, 0.7], the lambda messages to the parents are:
                p11 = 0.3 * (0.1 * 0.3 + 0.9 * 0.6) +
                    0.7 * (0.1 * 0.7 + 0.9 * 0.4)
                p12 = 0.3 * (0.1 * 0.4 + 0.9 * 0.7) +
                    0.7 * (0.1 * 0.6 + 0.9 * 0.3)
                p13 = 0.3 * (0.1 * 0.5 + 0.9 * 0.8) +
                    0.7 * (0.1 * 0.5 + 0.9 * 0.2)
                p21 = 0.3 * (0.2 * 0.3 + 0.3 * 0.4 + 0.5 * 0.5) +
                    0.7 * (0.2 * 0.7 + 0.3 * 0.6 + 0.5 * 0.5)
                p22 = 0.3 * (0.2 * 0.6 + 0.3 * 0.7 + 0.5 * 0.8) +
                    0.7 * (0.2 * 0.4 + 0.3 * 0.3 + 0.5 * 0.2)
                chlam = SoftScore([1,2], [0.3,0.7])
                range = [1,2]
                parranges = ([1,2,3], [1,2])
                parpis = (Cat([1,2,3], [0.2, 0.3, 0.5]), Cat([1,2], [0.1, 0.9]))
                l1 = send_lambda(x2, chlam, range, parranges, parpis, 1)
                l2 = send_lambda(x2, chlam, range, parranges, parpis, 2)
                @test isapprox([get_score(l1, i) for i in 1:3], [p11, p12, p13])
                @test isapprox([get_score(l2, i) for i in 1:2], [p21, p22])
            end

            @testset "with three parents" begin
                pi1 = [0.3, 0.7]
                pi2 = [0.2, 0.3, 0.5]
                pi3 = [0.6, 0.4]

                p11 = 0.0
                p12 = 0.0
                p21 = 0.0
                p22 = 0.0
                p23 = 0.0
                p31 = 0.0
                p32 = 0.0
                for i = 1:12
                    ix1 = div(i-1,6) + 1
                    ix2 = div(mod(i-1,6), 2) + 1
                    ix3 = mod(i-1,2) + 1
                    p1mod = (0.3 * cpd3entries[i] + 0.7 * (1 - cpd3entries[i])) * pi2[ix2] * pi3[ix3]
                    p2mod = (0.3 * cpd3entries[i] + 0.7 * (1 - cpd3entries[i])) * pi1[ix1] * pi3[ix3]
                    p3mod = (0.3 * cpd3entries[i] + 0.7 * (1 - cpd3entries[i])) * pi1[ix1] * pi2[ix2]
                    if ix1 == 1
                        p11 += p1mod
                    else
                        p12 += p1mod
                    end
                    if ix2 == 1
                        p21 += p2mod
                    elseif ix2 == 2
                        p22 += p2mod
                    else
                        p23 += p2mod
                    end
                    if ix3 == 1
                        p31 += p3mod
                    else
                        p32 += p3mod
                    end
                end
                chlam = SoftScore([1,2], [0.3, 0.7])
                range = [1,2]
                parranges = ([1,2], [1,2,3], [1,2])
                parpis = (Cat([1,2], pi1), Cat([1,2,3], pi2), Cat([1,2], pi3))
                l1 = send_lambda(x3, chlam, range, parranges, parpis, 1)
                l2 = send_lambda(x3, chlam, range, parranges, parpis, 2)
                l3 = send_lambda(x3, chlam, range, parranges, parpis, 3)
                @test isapprox([get_score(l1, i) for i in 1:2], [p11, p12])
                @test isapprox([get_score(l2, i) for i in 1:3], [p21, p22, p23])
                @test isapprox([get_score(l3, i) for i in 1:2], [p31, p32])
            end

            @testset "with four parents" begin
                pi1 = [0.3, 0.7]
                pi2 = [0.2, 0.3, 0.5]
                pi3 = [0.6, 0.4]
                pi4 = [0.8, 0.2]
                p11 = 0.0
                p12 = 0.0
                p21 = 0.0
                p22 = 0.0
                p23 = 0.0
                p31 = 0.0
                p32 = 0.0
                p41 = 0.0
                p42 = 0.0
                for i = 1:24
                    ix1 = div(i-1,12) + 1
                    ix2 = div(mod(i-1,12), 4) + 1
                    ix3 = div(mod(i-1,4), 2) + 1
                    ix4 = mod(i-1, 2) + 1
                    p1mod = (0.3 * cpd4entries[i] + 0.7 * (1 - cpd4entries[i])) * pi2[ix2] * pi3[ix3] * pi4[ix4]
                    p2mod = (0.3 * cpd4entries[i] + 0.7 * (1 - cpd4entries[i])) * pi1[ix1] * pi3[ix3] * pi4[ix4]
                    p3mod = (0.3 * cpd4entries[i] + 0.7 * (1 - cpd4entries[i])) * pi1[ix1] * pi2[ix2] * pi4[ix4]
                    p4mod = (0.3 * cpd4entries[i] + 0.7 * (1 - cpd4entries[i])) * pi1[ix1] * pi2[ix2] * pi3[ix3]
                    if ix1 == 1
                        p11 += p1mod
                    else
                        p12 += p1mod
                    end
                    if ix2 == 1
                        p21 += p2mod
                    elseif ix2 == 2
                        p22 += p2mod
                    else
                        p23 += p2mod
                    end
                    if ix3 == 1
                        p31 += p3mod
                    else
                        p32 += p3mod
                    end
                    if ix4 == 1
                        p41 += p4mod
                    else
                        p42 += p4mod
                    end
                end
                chlam = SoftScore([1,2], [0.3, 0.7])
                range = [1,2]
                parranges = ([1,2], [1,2,3], [1,2], [1,2])
                parpis = (Cat([1,2], pi1), Cat([1,2,3], pi2), Cat([1,2], pi3), Cat([1,2], pi4))
                l1 = send_lambda(x4, chlam, range, parranges, parpis, 1)
                l2 = send_lambda(x4, chlam, range, parranges, parpis, 2)
                l3 = send_lambda(x4, chlam, range, parranges, parpis, 3)
                l4 = send_lambda(x4, chlam, range, parranges, parpis, 4)
                @test isapprox([get_score(l1, i) for i in 1:2], [p11, p12])
                @test isapprox([get_score(l2, i) for i in 1:3], [p21, p22, p23])
                @test isapprox([get_score(l3, i) for i in 1:2], [p31, p32])
                @test isapprox([get_score(l4, i) for i in 1:2], [p41, p42])
            end

        end

    end

    @testset "Separable operations" begin
        alphas = [0.2, 0.3, 0.5]
        cpd1 = Dict((1,) => [0.1, 0.9], (2,) => [0.2, 0.8])
        cpd2 = Dict((1,) => [0.3, 0.7], (2,) => [0.4, 0.6], (3,) => [0.5, 0.5])
        cpd3 = Dict((1,) => [0.6, 0.4], (2,) => [0.7, 0.3])
        cpds :: Array{Dict{I,Array{Float64,1}} where I,1} = [cpd1, cpd2, cpd3]
        range = [1, 2]
        parranges = ([1, 2], [1, 2, 3], [1, 2])
        parpis = (Cat([1,2], [0.8, 0.2]), Cat([1,2,3], [0.5, 0.3, 0.2]), Cat([1,2], [0.9, 0.1]))
        x = Separable([1, 2], alphas, cpds)

        @testset "compute_pi" begin
            # If the parent probabilities are [0.8, 0.2], [0.5, 0.3, 0.2], and
            # [0.9, 0.1], x probabilities are:
            pa = 0.2 * (0.8 * 0.1 + 0.2 * 0.2) +
                0.3 * (0.5 * 0.3 + 0.3 * 0.4 + 0.2 * 0.5) +
                0.5 * (0.9 * 0.6 + 0.1 * 0.7)
            pb = 1 - pa
            ps = compute_pi(x, range, parranges, parpis)
            i1 = indexin(1, ps.__compiled_range)[1]
            i2 = indexin(2, ps.__compiled_range)[1]
            @test isapprox(ps.params[i1], pa)
            @test isapprox(ps.params[i2], pb)
        end

        @testset "send_lambda" begin
            # If the x probabilities are [0.3, 0.7], the lambda messages are computed by:
            p1othera = 0.3 * (0.5 * 0.3 + 0.3 * 0.4 + 0.2 * 0.5) +
                    0.5 * (0.9 * 0.6 + 0.1 * 0.7)
            p11a = 0.2 * 0.1 + p1othera
            p11 = 0.3 * p11a + 0.7 * (1-p11a) # = 0.3 * 0.2 * 0.1 + 0.3 * p1othera + 0.7 * (1 - 0.2 * 0.1) + 0.7 * p1otherb
            p12a = 0.2 * 0.2 + p1othera
            p12 = 0.3 * p12a + 0.7 * (1-p12a)

            p2othera = 0.2 * (0.8 * 0.1 + 0.2 * 0.2) +
                    0.5 * (0.9 * 0.6 + 0.1 * 0.7)
            p21a = 0.3 * 0.3 + p2othera
            p21 = 0.3 * p21a + 0.7 * (1-p21a)
            p22a = 0.3 * 0.4 + p2othera
            p22 = 0.3 * p22a + 0.7 * (1-p22a)
            p23a = 0.3 * 0.5 + p2othera
            p23 = 0.3 * p23a + 0.7 * (1-p23a)
            q21a = 0.3 * 0.3
            q21b = 0.3 * q21a + 0.7 * (1-q21a-p2othera)
            p3othera = 0.2 * (0.8 * 0.1 + 0.2 * 0.2) +
                    0.3 * (0.5 * 0.3 + 0.3 * 0.4 + 0.2 * 0.5)
            p31a = 0.5 * 0.6 + p3othera
            p31 = 0.3 * p31a + 0.7 * (1-p31a)
            p32a = 0.5 * 0.7 + p3othera
            p32 = 0.3 * p32a + 0.7 * (1-p32a)

            chlam = SoftScore([1,2], [0.3, 0.7])
            l1 = send_lambda(x, chlam, range, parranges, parpis, 1)
            l2 = send_lambda(x, chlam, range, parranges, parpis, 2)
            l3 = send_lambda(x, chlam, range, parranges, parpis, 3)
            @test isapprox([get_score(l1, i) for i in 1:2], [p11, p12])
            @test isapprox([get_score(l2, i) for i in 1:3], [p21, p22, p23])
            @test isapprox([get_score(l3, i) for i in 1:2], [p31, p32])
        end

    end

    @testset "BP operations in general" begin
        cpd = Dict((1,1) => [0.3, 0.7], (1,2) => [0.6, 0.4], (2,1) =>[0.4, 0.6],
                (2,2) => [0.7, 0.3], (3,1) => [0.5, 0.5], (3,2) => [0.8, 0.2])
        x = DiscreteCPT([1,2], cpd)

        @testset "compute_lambda" begin
            l1 = SoftScore([1,2], [0.1, 0.2])
            l2 = SoftScore([1,2], [0.3, 0.4])
            l3 = SoftScore([1,2], [0.5, 0.6])
            lam1 = compute_lambda(x, [1,2], [l1, l2, l3])
            lam2 = compute_lambda(x, [1,2], Score{output_type(x)}[])
            @test isapprox(normalize([get_score(lam1, i) for i in 1:2]), normalize([0.1 * 0.3 * 0.5, 0.2 * 0.4 * 0.6]))
            @test isapprox(normalize([get_score(lam2, i) for i in 1:2]), normalize([1.0, 1.0]))
        end

        @testset "compute_lambda avoids underflow" begin
            ls = fill(SoftScore([1,2], [0.000000001, 0.000000001]), 500)
            lam = compute_lambda(x, [1,2], ls)
            @test isapprox(normalize([get_score(lam, i) for i in 1:2]), [0.5, 0.5])
        end

        @testset "compute_bel" begin
            p1 = Cat([1, 2], [0.1, 0.2])
            l1 = SoftScore([1, 2], [0.3, 0.4])
            b = compute_bel(x, [2, 1], p1, l1)
            @test isapprox([cpdf(b, (), i) for i in [2,1]], normalize([0.2 * 0.4, 0.1 * 0.3]))
        end

        @testset "send_pi" begin
            b1 = Cat([1,2], [0.1, 0.2])
            l1 = SoftScore([1,2], [0.3, 0.4])
            l2 = SoftScore([1,2], [0.0, 0.0])
            p1 = send_pi(x, [2, 1], b1, l1)
            p2 = send_pi(x, [2, 1], b1, l2)
            @test isapprox([cpdf(p1, (), i) for i in [2,1]], normalize([0.2 / 0.4, 0.1 / 0.3]))
            @test all(y -> !isinf(y), [cpdf(p2, (), i) for i in [2,1]])
        end

        @testset "outgoing_pis" begin
            b1 = Cat([1,2], [0.1, 0.2])
            l1 = SoftScore([1,2], [0.3, 0.4])
            l2 = SoftScore([1,2], [0.5, 0.6])
            p1 = send_pi(x, [2,1], b1, l1)
            p2 = send_pi(x, [2,1], b1, l2)
            op = outgoing_pis(x, [2,1], b1, [l1, l2])
            @test all(i -> cpdf(op[1], (), i) == cpdf(p1, (), i), [2,1])
            @test all(i -> cpdf(op[2], (), i) == cpdf(p2, (), i), [2,1])
        end

        @testset "outgoing_lambdas" begin
            lam = SoftScore([1,2], [0.3, 0.7])
            incoming_pis = (Cat([1,2,3], [0.2, 0.3, 0.5]), Cat([1,2],[0.1, 0.9]))
            l1 = send_lambda(x, lam, [1,2], ([1,2,3], [1,2]), incoming_pis, 1)
            l2 = send_lambda(x, lam, [1,2], ([1,2,3], [1,2]), incoming_pis, 2)
            ols = outgoing_lambdas(x, lam, [1,2], ([1,2,3], [1,2]), incoming_pis)
            @test length(ols) == 2
            @test all(i -> get_score(ols[1], i) == get_score(l1, i), [1,2])
            @test all(i -> get_score(ols[2], i) == get_score(l2, i), [1,2])
        end

    end

    @testset "Three pass BP" begin
        x1m = Cat([1,2], [0.1, 0.9])()
        x2m = Cat([1,2,3], [0.2, 0.3, 0.5])()
        cpd2 = Dict((1,1) => [0.3, 0.7], (1,2) => [0.6, 0.4], (2,1) =>[0.4, 0.6],
                    (2,2) => [0.7, 0.3], (3,1) => [0.5, 0.5], (3,2) => [0.8, 0.2])
        x3m = DiscreteCPT([1,2], cpd2)()
        x4m = DiscreteCPT([1,2], Dict((1,) => [0.15, 0.85], (2,) => [0.25, 0.75]))()
        x5m = DiscreteCPT([1,2], Dict((1,) => [0.35, 0.65], (2,) => [0.45, 0.55]))()
        x6m = DiscreteCPT([1,2], Dict((1,) => [0.65, 0.35], (2,) => [0.75, 0.25]))()

        x1 = x1m(:x1)
        x2 = x2m(:x2)
        x3 = x3m(:x3)
        x4 = x4m(:x4)
        x5 = x5m(:x5)
        x6 = x6m(:x6)

        singlenet = InstantNetwork(Variable[x1], VariableGraph())

        @testset "single node network, no evidence" begin
            run = Runtime(singlenet)
            default_initializer(run)
            inst = current_instance(run, x1)
            three_pass_BP(run)
            bel = get_belief(run, inst)
            @test isapprox(cpdf(bel, (), 1), 0.1)
            @test isapprox(cpdf(bel, (), 2), 0.9)
        end

        @testset "single node network, with evidence" begin
            run = Runtime(singlenet)
            default_initializer(run)
            inst = current_instance(run, x1)
            post_evidence!(run, inst, HardScore(2))
            three_pass_BP(run)
            bel = get_belief(run, inst)
            @test isapprox(cpdf(bel, (), 1), 0.0)
            @test isapprox(cpdf(bel, (), 2), 1.0)
        end

        twoindepnet = InstantNetwork(Variable[x1, x2], VariableGraph())

        @testset "two independent nodes, no evidence" begin
            run = Runtime(twoindepnet)
            default_initializer(run)
            inst = current_instance(run, x1)
            three_pass_BP(run)
            bel = get_belief(run, inst)
            @test isapprox(cpdf(bel, (), 1), 0.1)
            @test isapprox(cpdf(bel, (), 2), 0.9)
        end

        @testset "two independent nodes, with evidence on other variable" begin
            run = Runtime(twoindepnet)
            default_initializer(run)
            inst1 = current_instance(run, x1)
            inst2 = current_instance(run, x2)
            post_evidence!(run, inst2, HardScore(2))
            three_pass_BP(run)
            bel = get_belief(run, inst1)
            @test isapprox(cpdf(bel, (), 1), 0.1)
            @test isapprox(cpdf(bel, (), 2), 0.9)
        end

        parchildnet = InstantNetwork(Variable[x1, x6], VariableGraph(x6=>[x1]))

        @testset "parent and child, no evidence" begin
            run = Runtime(parchildnet)
            default_initializer(run)
            inst1 = current_instance(run, x1)
            inst6 = current_instance(run, x6)
            three_pass_BP(run)
            bel1 = get_belief(run, inst1)
            @test isapprox(cpdf(bel1, (), 1), 0.1)
            @test isapprox(cpdf(bel1, (), 2), 0.9)
            bel6 = get_belief(run, inst6)
            @test isapprox(cpdf(bel6, (), 1), 0.1 * 0.65 + 0.9 * 0.75)
            @test isapprox(cpdf(bel6, (), 2), 0.1 * 0.35 + 0.9 * 0.25)
        end

        @testset "parent and child, evidence on parent" begin
            run = Runtime(parchildnet)
            default_initializer(run)
            inst1 = current_instance(run, x1)
            inst6 = current_instance(run, x6)
            post_evidence!(run, inst1, HardScore(2))
            three_pass_BP(run)
            bel1 = get_belief(run, inst1)
            @test isapprox(cpdf(bel1, (), 1), 0.0)
            @test isapprox(cpdf(bel1, (), 2), 1.0)
            bel6 = get_belief(run, inst6)
            @test isapprox(cpdf(bel6, (), 1), 0.75)
            @test isapprox(cpdf(bel6, (), 2), 0.25)
        end

        @testset "parent and child, evidence on child" begin
            run = Runtime(parchildnet)
            default_initializer(run)
            inst1 = current_instance(run, x1)
            inst6 = current_instance(run, x6)
            post_evidence!(run, inst6, HardScore(2))
            three_pass_BP(run)
            p1 = 0.1 * 0.35
            p2 = 0.9 * 0.25
            z = p1 + p2
            bel1 = get_belief(run, inst1)
            @test isapprox(cpdf(bel1, (), 1), p1 / z)
            @test isapprox(cpdf(bel1, (), 2), p2 / z)
            bel6 = get_belief(run, inst6)
            @test isapprox(cpdf(bel6, (), 1), 0.0)
            @test isapprox(cpdf(bel6, (), 2), 1.0)
        end

        @testset "parent and child, intervention on parent" begin
            run = Runtime(parchildnet)
            default_initializer(run)
            inst1 = current_instance(run, x1)
            inst6 = current_instance(run, x6)
            post_intervention!(run, inst1, Constant(2))
            three_pass_BP(run)
            bel1 = get_belief(run, inst1)
            @test isapprox(cpdf(bel1, (), 1), 0.0)
            @test isapprox(cpdf(bel1, (), 2), 1.0)
            bel6 = get_belief(run, inst6)
            @test isapprox(cpdf(bel6, (), 1), 0.75)
            @test isapprox(cpdf(bel6, (), 2), 0.25)
        end

        @testset "parent and child, intervention on child" begin
            run = Runtime(parchildnet)
            default_initializer(run)
            inst1 = current_instance(run, x1)
            inst6 = current_instance(run, x6)
            post_intervention!(run, inst6, Constant(2))
            three_pass_BP(run)
            p1 = 0.1 * 0.35
            p2 = 0.9 * 0.25
            z = p1 + p2
            bel1 = get_belief(run, inst1)
            @test isapprox(cpdf(bel1, (), 1), 0.1)
            @test isapprox(cpdf(bel1, (), 2), 0.9)
            bel6 = get_belief(run, inst6)
            @test isapprox(cpdf(bel6, (), 1), 0.0)
            @test isapprox(cpdf(bel6, (), 2), 1.0)
        end

        @testset "running bp twice with different evidence" begin
            run = Runtime(parchildnet)
            default_initializer(run)
            inst1 = current_instance(run, x1)
            inst6 = current_instance(run, x6)
            post_evidence!(run, inst1, HardScore(2))
            three_pass_BP(run)
            delete_evidence!(run, inst1)
            post_evidence!(run, inst6, HardScore(2))
            three_pass_BP(run)
            p1 = 0.1 * 0.35
            p2 = 0.9 * 0.25
            z = p1 + p2
            bel1 = get_belief(run, inst1)
            @test isapprox(cpdf(bel1, (), 1), p1 / z)
            @test isapprox(cpdf(bel1, (), 2), p2 / z)
            bel6 = get_belief(run, inst6)
            @test isapprox(cpdf(bel6, (), 1), 0.0)
            @test isapprox(cpdf(bel6, (), 2), 1.0)
        end

        @testset "with soft evidence on root and evidence on child" begin
            run = Runtime(parchildnet)
            default_initializer(run)
            inst1 = current_instance(run, x1)
            inst6 = current_instance(run, x6)
            post_evidence!(run, inst1, SoftScore([1,2], [0.3, 0.7]))
            post_evidence!(run, inst6, HardScore(2))
            three_pass_BP(run)
            # soft evidence is interpreted as an additional lambda message
            p1 = 0.1 * 0.3 * 0.35
            p2 = 0.9 * 0.7 * 0.25
            z = p1 + p2
            bel1 = get_belief(run, inst1)
            @test isapprox(cpdf(bel1, (), 1), p1 / z)
            @test isapprox(cpdf(bel1, (), 2), p2 / z)
            bel6 = get_belief(run, inst6)
            @test isapprox(cpdf(bel6, (), 1), 0.0)
            @test isapprox(cpdf(bel6, (), 2), 1.0)
        end

        fivecpdnet = InstantNetwork(Variable[x1,x2,x3,x4,x5], VariableGraph(x3=>[x2,x1], x4=>[x3], x5=>[x3]))

        acfhj = 0.1 * 0.2 * 0.3 * 0.15 * 0.35
        acfhk = 0.1 * 0.2 * 0.3 * 0.15 * 0.65
        acfij = 0.1 * 0.2 * 0.3 * 0.85 * 0.35
        acfik = 0.1 * 0.2 * 0.3 * 0.85 * 0.65
        acghj = 0.1 * 0.2 * 0.7 * 0.25 * 0.45
        acghk = 0.1 * 0.2 * 0.7 * 0.25 * 0.55
        acgij = 0.1 * 0.2 * 0.7 * 0.75 * 0.45
        acgik = 0.1 * 0.2 * 0.7 * 0.75 * 0.55
        adfhj = 0.1 * 0.3 * 0.4 * 0.15 * 0.35
        adfhk = 0.1 * 0.3 * 0.4 * 0.15 * 0.65
        adfij = 0.1 * 0.3 * 0.4 * 0.85 * 0.35
        adfik = 0.1 * 0.3 * 0.4 * 0.85 * 0.65
        adghj = 0.1 * 0.3 * 0.6 * 0.25 * 0.45
        adghk = 0.1 * 0.3 * 0.6 * 0.25 * 0.55
        adgij = 0.1 * 0.3 * 0.6 * 0.75 * 0.45
        adgik = 0.1 * 0.3 * 0.6 * 0.75 * 0.55
        aefhj = 0.1 * 0.5 * 0.5 * 0.15 * 0.35
        aefhk = 0.1 * 0.5 * 0.5 * 0.15 * 0.65
        aefij = 0.1 * 0.5 * 0.5 * 0.85 * 0.35
        aefik = 0.1 * 0.5 * 0.5 * 0.85 * 0.65
        aeghj = 0.1 * 0.5 * 0.5 * 0.25 * 0.45
        aeghk = 0.1 * 0.5 * 0.5 * 0.25 * 0.55
        aegij = 0.1 * 0.5 * 0.5 * 0.75 * 0.45
        aegik = 0.1 * 0.5 * 0.5 * 0.75 * 0.55
        bcfhj = 0.9 * 0.2 * 0.6 * 0.15 * 0.35
        bcfhk = 0.9 * 0.2 * 0.6 * 0.15 * 0.65
        bcfij = 0.9 * 0.2 * 0.6 * 0.85 * 0.35
        bcfik = 0.9 * 0.2 * 0.6 * 0.85 * 0.65
        bcghj = 0.9 * 0.2 * 0.4 * 0.25 * 0.45
        bcghk = 0.9 * 0.2 * 0.4 * 0.25 * 0.55
        bcgij = 0.9 * 0.2 * 0.4 * 0.75 * 0.45
        bcgik = 0.9 * 0.2 * 0.4 * 0.75 * 0.55
        bdfhj = 0.9 * 0.3 * 0.7 * 0.15 * 0.35
        bdfhk = 0.9 * 0.3 * 0.7 * 0.15 * 0.65
        bdfij = 0.9 * 0.3 * 0.7 * 0.85 * 0.35
        bdfik = 0.9 * 0.3 * 0.7 * 0.85 * 0.65
        bdghj = 0.9 * 0.3 * 0.3 * 0.25 * 0.45
        bdghk = 0.9 * 0.3 * 0.3 * 0.25 * 0.55
        bdgij = 0.9 * 0.3 * 0.3 * 0.75 * 0.45
        bdgik = 0.9 * 0.3 * 0.3 * 0.75 * 0.55
        befhj = 0.9 * 0.5 * 0.8 * 0.15 * 0.35
        befhk = 0.9 * 0.5 * 0.8 * 0.15 * 0.65
        befij = 0.9 * 0.5 * 0.8 * 0.85 * 0.35
        befik = 0.9 * 0.5 * 0.8 * 0.85 * 0.65
        beghj = 0.9 * 0.5 * 0.2 * 0.25 * 0.45
        beghk = 0.9 * 0.5 * 0.2 * 0.25 * 0.55
        begij = 0.9 * 0.5 * 0.2 * 0.75 * 0.45
        begik = 0.9 * 0.5 * 0.2 * 0.75 * 0.55

        @testset "five node non-loopy network with discrete CPD nodes and no evidence" begin
            run = Runtime(fivecpdnet)
            default_initializer(run)
            inst1 = current_instance(run, x1)
            inst3 = current_instance(run, x3)
            inst5 = current_instance(run, x5)
            three_pass_BP(run)
            a = acfhj + acfhk + acfij + acfik + acghj + acghk + acgij + acgik +
                adfhj + adfhk + adfij + adfik + adghj + adghk + adgij + adgik +
                aefhj + aefhk + aefij + aefik + aeghj + aeghk + aegij + aegik
            b = 1-a
            f = acfhj + acfhk + acfij + acfik + adfhj + adfhk + adfij + adfik + aefhj + aefhk + aefij + aefik +
                bcfhj + bcfhk + bcfij + bcfik + bdfhj + bdfhk + bdfij + bdfik + befhj + befhk + befij + befik
            g = 1-f
            j = acfhj + acfij + acghj + acgij + adfhj + adfij + adghj + adgij + aefhj + aefij + aeghj + aegij +
                bcfhj + bcfij + bcghj + bcgij + bdfhj + bdfij + bdghj + bdgij + befhj + befij + beghj + begij
            k = 1-j
            bel1 = get_belief(run, inst1)
            @test isapprox(cpdf(bel1, (), 1), a)
            @test isapprox(cpdf(bel1, (), 2), b)
            bel3 = get_belief(run, inst3)
            @test isapprox(cpdf(bel3, (), 1), f)
            @test isapprox(cpdf(bel3, (), 2), g)
            bel5 = get_belief(run, inst5)
            @test isapprox(cpdf(bel5, (), 1), j)
            @test isapprox(cpdf(bel5, (), 2), k)
        end

        @testset "five node non-loopy network with discrete CPD nodes and evidence at root" begin
            run = Runtime(fivecpdnet)
            default_initializer(run)
            inst1 = current_instance(run, x1)
            inst5 = current_instance(run, x5)
            post_evidence!(run, inst1, HardScore(2))
            three_pass_BP(run)
            bj = bcfhj + bcfij + bcghj + bcgij + bdfhj + bdfij + bdghj + bdgij + befhj + befij + beghj + begij
            bk = bcfhk + bcfik + bcghk + bcgik + bdfhk + bdfik + bdghk + bdgik + befhk + befik + beghk + begik
            bel5 = get_belief(run, inst5)
            @test isapprox(cpdf(bel5, (), 1), bj / (bj + bk))
            @test isapprox(cpdf(bel5, (), 2), bk / (bj + bk))
        end

        @testset "five node non-loopy network with discrete CPD nodes and evidence at leaves" begin
            run = Runtime(fivecpdnet)
            default_initializer(run)
            inst1 = current_instance(run, x1)
            inst4 = current_instance(run, x4)
            inst5 = current_instance(run, x5)
            post_evidence!(run, inst4, HardScore(1))
            post_evidence!(run, inst5, HardScore(2))
            three_pass_BP(run)
            ahk = acfhk + acghk + adfhk + adghk + aefhk + aeghk
            bhk = bcfhk + bcghk + bdfhk + bdghk + befhk + beghk
            bel1 = get_belief(run, inst1)
            @test isapprox(cpdf(bel1, (), 1), ahk / (ahk + bhk))
            @test isapprox(cpdf(bel1, (), 2), bhk / (ahk + bhk))
        end

        @testset "five node non-loopy network with discrete CPD nodes and evidence at root and leaf" begin
            run = Runtime(fivecpdnet)
            default_initializer(run)
            inst1 = current_instance(run, x1)
            inst2 = current_instance(run, x2)
            inst3 = current_instance(run, x3)
            inst4 = current_instance(run, x4)
            inst5 = current_instance(run, x5)
            post_evidence!(run, inst1, HardScore(1))
            post_evidence!(run, inst5, HardScore(2))
            three_pass_BP(run)
            ack = acfhk + acfik + acghk + acgik
            adk = adfhk + adfik + adghk + adgik
            aek = aefhk + aefik + aeghk + aegik
            afj = acfhj + acfij + adfhj + adfij + aefhj + aefij
            afk = acfhk + acfik + adfhk + adfik + aefhk + aefik
            agj = acghj + acgij + adghj + adgij + aeghj + aegij
            agk = acghk + acgik + adghk + adgik + aeghk + aegik
            ahk = acfhk + acghk + adfhk + adghk + aefhk + aeghk
            aik = acfik + acgik + adfik + adgik + aefik + aegik
            af = afj + afk
            ag = agj + agk
            bel2 = get_belief(run, inst2)
            @test isapprox(cpdf(bel2, (), 1), ack / (ack + adk + aek))
            @test isapprox(cpdf(bel2, (), 2), adk / (ack + adk + aek))
            @test isapprox(cpdf(bel2, (), 3), aek / (ack + adk + aek))
            bel3 = get_belief(run, inst3)
            @test isapprox(cpdf(bel3, (), 1), afk / (afk + agk))
            @test isapprox(cpdf(bel3, (), 2), agk / (afk + agk))
            bel4 = get_belief(run, inst4)
            @test isapprox(cpdf(bel4, (), 1), ahk / (ahk + aik))
            @test isapprox(cpdf(bel4, (), 2), aik / (ahk + aik))
        end

        y1 = Cat([1,2], [0.1, 0.9])()(:y1)
        y2 = Cat([1,2,3], [0.2, 0.3, 0.5])()(:y2)
        y3 = Cat([1,2], [0.8, 0.2])()(:y3)
        cpt1 = Dict((1,) => [0.1, 0.9], (2,) => [0.2, 0.8])
        cpt2 = Dict((1,) => [0.3, 0.7], (2,) => [0.4, 0.6], (3,) => [0.5, 0.5])
        cpt3 = Dict((1,) => [0.6, 0.4], (2,) => [0.7, 0.3])
        cpts::SepCPTs = [cpt1, cpt2, cpt3]
        y4= Separable([1,2], [0.5, 0.2, 0.3], cpts)()(:y4)
        y5 = DiscreteCPT([1,2], Dict((1,) => [0.35, 0.65], (2,) => [0.45, 0.55]))()(:y5)

        fivesepnet = InstantNetwork(Variable[y1,y2,y3,y4,y5], VariableGraph(y4=>[y1,y2,y3], y5=>[y4]))

        acfhj = 0.1 * 0.2 * 0.8 * (0.5 * 0.1 + 0.2 * 0.3 + 0.3 * 0.6) * 0.35
        acfhk = 0.1 * 0.2 * 0.8 * (0.5 * 0.1 + 0.2 * 0.3 + 0.3 * 0.6) * 0.65
        acfij = 0.1 * 0.2 * 0.8 * (0.5 * 0.9 + 0.2 * 0.7 + 0.3 * 0.4) * 0.45
        acfik = 0.1 * 0.2 * 0.8 * (0.5 * 0.9 + 0.2 * 0.7 + 0.3 * 0.4) * 0.55
        acghj = 0.1 * 0.2 * 0.2 * (0.5 * 0.1 + 0.2 * 0.3 + 0.3 * 0.7) * 0.35
        acghk = 0.1 * 0.2 * 0.2 * (0.5 * 0.1 + 0.2 * 0.3 + 0.3 * 0.7) * 0.65
        acgij = 0.1 * 0.2 * 0.2 * (0.5 * 0.9 + 0.2 * 0.7 + 0.3 * 0.3) * 0.45
        acgik = 0.1 * 0.2 * 0.2 * (0.5 * 0.9 + 0.2 * 0.7 + 0.3 * 0.3) * 0.55
        adfhj = 0.1 * 0.3 * 0.8 * (0.5 * 0.1 + 0.2 * 0.4 + 0.3 * 0.6) * 0.35
        adfhk = 0.1 * 0.3 * 0.8 * (0.5 * 0.1 + 0.2 * 0.4 + 0.3 * 0.6) * 0.65
        adfij = 0.1 * 0.3 * 0.8 * (0.5 * 0.9 + 0.2 * 0.6 + 0.3 * 0.4) * 0.45
        adfik = 0.1 * 0.3 * 0.8 * (0.5 * 0.9 + 0.2 * 0.6 + 0.3 * 0.4) * 0.55
        adghj = 0.1 * 0.3 * 0.2 * (0.5 * 0.1 + 0.2 * 0.4 + 0.3 * 0.7) * 0.35
        adghk = 0.1 * 0.3 * 0.2 * (0.5 * 0.1 + 0.2 * 0.4 + 0.3 * 0.7) * 0.65
        adgij = 0.1 * 0.3 * 0.2 * (0.5 * 0.9 + 0.2 * 0.6 + 0.3 * 0.3) * 0.45
        adgik = 0.1 * 0.3 * 0.2 * (0.5 * 0.9 + 0.2 * 0.6 + 0.3 * 0.3) * 0.55
        aefhj = 0.1 * 0.5 * 0.8 * (0.5 * 0.1 + 0.2 * 0.5 + 0.3 * 0.6) * 0.35
        aefhk = 0.1 * 0.5 * 0.8 * (0.5 * 0.1 + 0.2 * 0.5 + 0.3 * 0.6) * 0.65
        aefij = 0.1 * 0.5 * 0.8 * (0.5 * 0.9 + 0.2 * 0.5 + 0.3 * 0.4) * 0.45
        aefik = 0.1 * 0.5 * 0.8 * (0.5 * 0.9 + 0.2 * 0.5 + 0.3 * 0.4) * 0.55
        aeghj = 0.1 * 0.5 * 0.2 * (0.5 * 0.1 + 0.2 * 0.5 + 0.3 * 0.7) * 0.35
        aeghk = 0.1 * 0.5 * 0.2 * (0.5 * 0.1 + 0.2 * 0.5 + 0.3 * 0.7) * 0.65
        aegij = 0.1 * 0.5 * 0.2 * (0.5 * 0.9 + 0.2 * 0.5 + 0.3 * 0.3) * 0.45
        aegik = 0.1 * 0.5 * 0.2 * (0.5 * 0.9 + 0.2 * 0.5 + 0.3 * 0.3) * 0.55
        bcfhj = 0.9 * 0.2 * 0.8 * (0.5 * 0.2 + 0.2 * 0.3 + 0.3 * 0.6) * 0.35
        bcfhk = 0.9 * 0.2 * 0.8 * (0.5 * 0.2 + 0.2 * 0.3 + 0.3 * 0.6) * 0.65
        bcfij = 0.9 * 0.2 * 0.8 * (0.5 * 0.8 + 0.2 * 0.7 + 0.3 * 0.4) * 0.45
        bcfik = 0.9 * 0.2 * 0.8 * (0.5 * 0.8 + 0.2 * 0.7 + 0.3 * 0.4) * 0.55
        bcghj = 0.9 * 0.2 * 0.2 * (0.5 * 0.2 + 0.2 * 0.3 + 0.3 * 0.7) * 0.35
        bcghk = 0.9 * 0.2 * 0.2 * (0.5 * 0.2 + 0.2 * 0.3 + 0.3 * 0.7) * 0.65
        bcgij = 0.9 * 0.2 * 0.2 * (0.5 * 0.8 + 0.2 * 0.7 + 0.3 * 0.3) * 0.45
        bcgik = 0.9 * 0.2 * 0.2 * (0.5 * 0.8 + 0.2 * 0.7 + 0.3 * 0.3) * 0.55
        bdfhj = 0.9 * 0.3 * 0.8 * (0.5 * 0.2 + 0.2 * 0.4 + 0.3 * 0.6) * 0.35
        bdfhk = 0.9 * 0.3 * 0.8 * (0.5 * 0.2 + 0.2 * 0.4 + 0.3 * 0.6) * 0.65
        bdfij = 0.9 * 0.3 * 0.8 * (0.5 * 0.8 + 0.2 * 0.6 + 0.3 * 0.4) * 0.45
        bdfik = 0.9 * 0.3 * 0.8 * (0.5 * 0.8 + 0.2 * 0.6 + 0.3 * 0.4) * 0.55
        bdghj = 0.9 * 0.3 * 0.2 * (0.5 * 0.2 + 0.2 * 0.4 + 0.3 * 0.7) * 0.35
        bdghk = 0.9 * 0.3 * 0.2 * (0.5 * 0.2 + 0.2 * 0.4 + 0.3 * 0.7) * 0.65
        bdgij = 0.9 * 0.3 * 0.2 * (0.5 * 0.8 + 0.2 * 0.6 + 0.3 * 0.3) * 0.45
        bdgik = 0.9 * 0.3 * 0.2 * (0.5 * 0.8 + 0.2 * 0.6 + 0.3 * 0.3) * 0.55
        befhj = 0.9 * 0.5 * 0.8 * (0.5 * 0.2 + 0.2 * 0.5 + 0.3 * 0.6) * 0.35
        befhk = 0.9 * 0.5 * 0.8 * (0.5 * 0.2 + 0.2 * 0.5 + 0.3 * 0.6) * 0.65
        befij = 0.9 * 0.5 * 0.8 * (0.5 * 0.8 + 0.2 * 0.5 + 0.3 * 0.4) * 0.45
        befik = 0.9 * 0.5 * 0.8 * (0.5 * 0.8 + 0.2 * 0.5 + 0.3 * 0.4) * 0.55
        beghj = 0.9 * 0.5 * 0.2 * (0.5 * 0.2 + 0.2 * 0.5 + 0.3 * 0.7) * 0.35
        beghk = 0.9 * 0.5 * 0.2 * (0.5 * 0.2 + 0.2 * 0.5 + 0.3 * 0.7) * 0.65
        begij = 0.9 * 0.5 * 0.2 * (0.5 * 0.8 + 0.2 * 0.5 + 0.3 * 0.3) * 0.45
        begik = 0.9 * 0.5 * 0.2 * (0.5 * 0.8 + 0.2 * 0.5 + 0.3 * 0.3) * 0.55

        @testset "five node non-loopy network with separable CPD nodes and no evidence" begin
            run = Runtime(fivesepnet)
            default_initializer(run)
            inst4 = current_instance(run, y4)
            three_pass_BP(run)
            h = acfhj + acfhk + acghj + acghk + adfhj + adfhk + adghj + adghk + aefhj + aefhk + aeghj + aeghk +
                bcfhj + bcfhk + bcghj + bcghk + bdfhj + bdfhk + bdghj + bdghk + befhj + befhk + beghj + beghk
            i = acfij + acfik + acgij + acgik + adfij + adfik + adgij + adgik + aefij + aefik + aegij + aegik +
                bcfij + bcfik + bcgij + bcgik + bdfij + bdfik + bdgij + bdgik + befij + befik + begij + begik
                bel4 = get_belief(run, inst4)
                @test isapprox(cpdf(bel4, (), 1), h)
                @test isapprox(cpdf(bel4, (), 2), i)
        end

        @testset "five node non-loopy network with separable CPD nodes and evidence at root" begin
            run = Runtime(fivesepnet)
            default_initializer(run)
            inst1 = current_instance(run, y1)
            inst5 = current_instance(run, y5)
            post_evidence!(run, inst1, HardScore(2))
            three_pass_BP(run)
            bj = bcfhj + bcfij + bcghj + bcgij + bdfhj + bdfij + bdghj + bdgij + befhj + befij + beghj + begij
            bk = bcfhk + bcfik + bcghk + bcgik + bdfhk + bdfik + bdghk + bdgik + befhk + befik + beghk + begik
            bel5 = get_belief(run, inst5)
            @test isapprox(cpdf(bel5, (), 1), bj / (bj + bk))
            @test isapprox(cpdf(bel5, (), 2), bk / (bj + bk))
        end

        @testset "five node non-loopy network with separable CPD nodes and evidence at leaves" begin
            run = Runtime(fivesepnet)
            default_initializer(run)
            inst1 = current_instance(run, y1)
            inst5 = current_instance(run, y5)
            post_evidence!(run, inst5, HardScore(2))
            three_pass_BP(run)
            ak = acfhk + acfik + acghk + acgik + adfhk + adfik + adghk + adgik + aefhk + aefik + aeghk + aegik
            bk = bcfhk + bcfik + bcghk + bcgik + bdfhk + bdfik + bdghk + bdgik + befhk + befik + beghk + begik
            bel1 = get_belief(run, inst1)
            @test isapprox(cpdf(bel1, (), 1), ak / (ak + bk))
            @test isapprox(cpdf(bel1, (), 2), bk / (ak + bk))
        end

        @testset "five node non-loopy network with separable CPD nodes and evidence at root and leaf" begin
            run = Runtime(fivesepnet)
            default_initializer(run)
            inst1 = current_instance(run, y1)
            inst4 = current_instance(run, y4)
            inst5 = current_instance(run, y5)
            post_evidence!(run, inst1, HardScore(2))
            post_evidence!(run, inst5, HardScore(2))
            three_pass_BP(run)
            bhk = bcfhk + bcghk + bdfhk + bdghk + befhk + beghk
            bik = bcfik + bcgik + bdfik + bdgik + befik + begik
            bel4 = get_belief(run, inst4)
            @test isapprox(cpdf(bel4, (), 1), bhk / (bhk + bik))
            @test isapprox(cpdf(bel4, (), 2), bik / (bhk + bik))
        end

        @testset "five node non-loopy network with separable CPD nodes and evidence on separable node" begin
            run = Runtime(fivesepnet)
            default_initializer(run)
            inst1 = current_instance(run, y1)
            inst4 = current_instance(run, y4)
            post_evidence!(run, inst4, HardScore(2))
            three_pass_BP(run)
            ai = acfij + acgij + adfij + adgij + aefij + aegij +
                acfik + acgik + adfik + adgik + aefik + aegik
            bi = bcfij + bcgij + bdfij + bdgij + befij + begij +
                bcfik + bcgik + bdfik + bdgik + befik + begik
            bel1 = get_belief(run, inst1)
            @test isapprox(cpdf(bel1, (), 1), ai / (ai + bi))
            @test isapprox(cpdf(bel1, (), 2), bi / (ai + bi))
        end

        z1 = Cat([1,2], [0.1, 0.9])()(:z1)
        z2 = DiscreteCPT([1,2], Dict((1,) => [0.2, 0.8], (2,) => [0.3, 0.7]))()(:z2)
        z3 = DiscreteCPT([1,2], Dict((1,) => [0.4, 0.6], (2,) => [0.5, 0.5]))()(:z3)
        cpt1 = Dict((1,) => [0.6, 0.4], (2,) => [0.7, 0.3])
        cpt2 = Dict((1,) => [0.8, 0.2], (2,) => [0.9, 0.1])
        cpts = [cpt1, cpt2]
        z4 = Separable([1,2], [0.75, 0.25], cpts)()(:z4)
        loopynet = InstantNetwork(Variable[z1,z2,z3,z4], VariableGraph(z2=>[z1], z3=>[z1], z4=>[z2,z3]))

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

        @testset "four node loopy network with separable nodes and no evidence" begin
            run = Runtime(loopynet)
            default_initializer(run)
            inst4 = current_instance(run, z4)
            three_pass_BP(run)
            g = aceg + acfg + adeg + adfg + bceg + bcfg + bdeg + bdfg
            h = aceh + acfh + adeh + adfh + bceh + bcfh + bdeh + bdfh
            bel4 = get_belief(run, inst4)
            @test isapprox(cpdf(bel4, (), 1), g)
            @test isapprox(cpdf(bel4, (), 2), h)
        end

        @testset "four node loopy network with separable nodes and evidence at root (exact)" begin
            run = Runtime(loopynet)
            default_initializer(run)
            inst1 = current_instance(run, z1)
            inst4 = current_instance(run, z4)
            post_evidence!(run, inst1, HardScore(2))
            three_pass_BP(run)
            bg = bceg + bcfg + bdeg + bdfg
            bh = bceh + bcfh + bdeh + bdfh
            bel4 = get_belief(run, inst4)
            @test isapprox(cpdf(bel4, (), 1), bg / (bg + bh))
            @test isapprox(cpdf(bel4, (), 2), bh / (bg + bh))
        end

        @testset "four nodeloopy network with separable nodes and evidence at leaves (approximate)" begin
            run = Runtime(loopynet)
            default_initializer(run)
            inst1 = current_instance(run, z1)
            inst4 = current_instance(run, z4)
            post_evidence!(run, inst4, HardScore(2))
            three_pass_BP(run)
            ah = aceh + acfh + adeh + adfh
            bh = bceh + bcfh + bdeh + bdfh
            bel1 = get_belief(run, inst1)
            @test isapprox(cpdf(bel1, (), 1), ah / (ah + bh); atol = 0.0001)
            @test isapprox(cpdf(bel1, (), 2), bh / (ah + bh); atol = 0.0001)
        end

    end

    @testset "loopy BP" begin
        # Loopy BP is hard to test directly.
        # Even on loopy networks, it is not guaranteed to have less error than three pass BP.
        # We can only test that it behaves like three pass BP where expected, which is without evidence,
        # and differently where not.
        x1 = Cat([1,2], [0.1, 0.9])()(:x1)
        x2 = DiscreteCPT([1,2], Dict((1,) => [0.2, 0.8], (2,) => [0.8, 0.2]))()(:x2)
        x3 = DiscreteCPT([1,2], Dict((1,) => [0.3, 0.7], (2,) => [0.7, 0.3]))()(:x3)
        x4 = DiscreteCPT([1,2], Dict((1,1) => [0.1, 0.9], (1,2) => [0.1, 0.9], (2,1) => [0.1, 0.9], (2,2) => [0.95, 0.05]))()(:x4)
        loopynet = InstantNetwork(Variable[x1,x2,x3,x4], VariableGraph(x2=>[x1], x3=>[x1], x4=>[x3,x2]))

        @testset "marginal query" begin
            # On marginal queries, there are no lambda messages, so loopy BP converges in one pass
            # Therefore it should have the same result as three pass BP.
            runloopy = Runtime(loopynet)
            default_initializer(runloopy)
            loopy_BP(runloopy; epsilon = 0.000001)
            runthree = Runtime(loopynet)
            default_initializer(runthree)
            three_pass_BP(runthree)
            # runve = Runtime(loopynet)
            # order = topsort(get_initial_graph(loopynet))
            # ensure_all!(runve)
            # set_ranges!(runve, 2, 1, order)
            # exact = ve(runve, order, [x4])

            loopyinst4 = current_instance(runloopy, x4)
            loopybel = get_belief(runloopy, loopyinst4)
            threeinst4 = current_instance(runthree, x4)
            threebel = get_belief(runthree, threeinst4)
            # exactbel = exact.entries
            for i in 1:2
                # @test abs(loopybel[i] - exactbel[i]) <= abs(threebel[i] - exactbel[i])
                @test cpdf(loopybel, (), i) == cpdf(threebel, (), i)
            end
        end

        @testset "marginal query" begin
            # On conditional queries with evidence, loopy BP could run more iterations.
            # We design the network to make sure this happens.
            # Therefore it should have different results from three pass BP.
            runloopy = Runtime(loopynet)
            default_initializer(runloopy)
            loopyinst4 = current_instance(runloopy, x4)
            post_evidence!(runloopy, loopyinst4, HardScore(2))
            loopy_BP(runloopy; epsilon = 0.000001)
            runthree = Runtime(loopynet)
            default_initializer(runthree)
            threeinst4 = current_instance(runthree, x4)
            post_evidence!(runthree, threeinst4, HardScore(2))
            three_pass_BP(runthree)

            loopyinst1 = current_instance(runloopy, x1)
            loopybel = get_belief(runloopy, loopyinst1)
            threeinst1 = current_instance(runthree, x1)
            threebel = get_belief(runthree, threeinst1)
            for x in [1,2]
                @test cpdf(loopybel, (), x) != cpdf(threebel, (), x)
            end
        end

        @testset "Correctly handle ranges not in alphabetical order" begin
            # create network
            p = Cat([:N, :AN],[1.0, 0.0])()(:p)
            o = DiscreteCPT([:B, :I], Dict((:N,) => [1.0, 0.0], (:AN,) => [0.2, 0.8]))()(:o)
            a = DiscreteCPT([:SK, :SH],  Dict((:B,) => [1.0, 0.0], (:I,) => [0.0,1.0]))()(:a)
            m = DiscreteCPT([:P, :M1, :M2], Dict((:SK,) => [0.9, 0.099, 0.001], (:SH,) => [0.2, 0.05, 0.75]))()(:m)

            network = InstantNetwork(Variable[p,o,a,m], VariableGraph(o=>[p], a=>[o], m=>[a]))

            # create runtime
            runtime = Runtime(network)
            default_initializer(runtime)

            # run algorithm
            loopy_BP(runtime; epsilon = 0.000001)

            # get beliefs
            p_i = current_instance(runtime,get_node(network,:p))
            belief = get_belief(runtime, p_i)
            @test isapprox([cpdf(belief, (), x) for x in [:N, :AN]], [1.0, 0.0];atol=0.1)

            o_i = current_instance(runtime, get_node(network,:o))
            belief = get_belief(runtime, o_i)
            @test isapprox([cpdf(belief, (), x) for x in [:B, :I]], [1.0, 0.0];atol=0.1)

            a_i = current_instance(runtime, get_node(network,:a))
            belief = get_belief(runtime, a_i)
            @test isapprox([cpdf(belief, (), x) for x in [:SK, :SH]], [1.0, 0.0];atol=0.1)

            m_i = current_instance(runtime, get_node(network,:m))
            belief = get_belief(runtime, m_i)
            @test isapprox([cpdf(belief, (), x) for x in [:P, :M1,:M2]], [0.9, 0.099, 0.001];atol=0.1)
        end
        
    end

    @testset "BP with determinisitic variables" begin
        
        @testset "Det" begin
            c1 = Cat([1.1, 2.2], [0.4, 0.6])
            c2 = Cat([3.3, 4.4, 5.5], [0.2, 0.3, 0.5])
            f(i,j) = Int(floor(i + j))
            d = Det(Tuple{Float64, Float64}, Int, f)
            vc1 = c1()(:c1)
            vc2 = c2()(:c2)
            vd = d()(:d)
            net = InstantNetwork(Variable[vc1,vc2,vd], VariableGraph(vd=>[vc1,vc2]))

            @testset "Prior probabilities" begin
                run = Runtime(net)
                default_initializer(run)
                # post_evidence!(run, current_instance(run, vd), SoftScore([5,6], [0.9, 0.1]))
                three_pass_BP(run)
                bc1 = get_belief(run, current_instance(run, vc1))
                bc2 = get_belief(run, current_instance(run, vc2))
                bd = get_belief(run, current_instance(run, vd))
                @test isapprox(cpdf(bc1, (), 1.1), 0.4)
                @test isapprox(cpdf(bc1, (), 2.2), 0.6)
                @test isapprox(cpdf(bc2, (), 3.3), 0.2)
                @test isapprox(cpdf(bc2, (), 4.4), 0.3)
                @test isapprox(cpdf(bc2, (), 5.5), 0.5)
                @test isapprox(cpdf(bd, (), 4), 0.4 * 0.2)
                @test isapprox(cpdf(bd, (), 5), 0.4 * 0.3 + 0.6 * 0.2)
                @test isapprox(cpdf(bd, (), 6), 0.4 * 0.5 + 0.6 * 0.3)
                @test isapprox(cpdf(bd, (), 7), 0.6 * 0.5)
            end

            @testset "Posterior probabilities" begin
                run = Runtime(net)
                default_initializer(run)
                post_evidence!(run, current_instance(run, vd), SoftScore([5,6], [0.9, 0.1]))
                three_pass_BP(run)
                bc1 = get_belief(run, current_instance(run, vc1))
                bc2 = get_belief(run, current_instance(run, vc2))
                bd = get_belief(run, current_instance(run, vd))
                p14 = 0.4 * 0.3 * 0.9
                p15 = 0.4 * 0.5 * 0.1
                p23 = 0.6 * 0.2 * 0.9
                p24 = 0.6 * 0.3 * 0.1
                p1 = p14 + p15
                p2 = p23 + p24
                p3 = p23
                p4 = p14 + p24
                p5 = p15
                pd5 = p14 + p23
                pd6 = p15 + p24
                z1 = p1 + p2
                z2 = p3 + p4 + p5
                zd = pd5 + pd6
                @test isapprox([cpdf(bc1, (), x) for x in [1.1, 2.2]], [p1, p2] ./ z1)
                @test isapprox([cpdf(bc2, (), x) for x in [3.3, 4.4, 5.5]], [p3, p4, p5] ./ z2)
                @test isapprox([cpdf(bd, (), x) for x in [4, 5, 6, 7]], [0, pd5, pd6, 0] ./ zd)
            end

        #     @testset "Downsampling in Det" begin
        #         function f(x_tmin1::Float64, func::Symbol)
        #             delta_t = 1
        #             if (func==:F2) # square: x_t = t^2 <=> x_t = x_tmin1 + 2*sqrt(x_tmin1)*delta_t + delta_t^2
        #                 x_t = floor(x_tmin1 + 2 * sqrt(x_tmin1) * delta_t + delta_t^2)
        #             elseif (func==:F3) # linear: x_t = at + k <=> x_t = x_tmin1 + a * delta_t
        #                 a = 5 # slope
        #                 x_t = floor(x_tmin1 + a * delta_t)
        #             else # include propagation constant x_t = x_tmin1 # :F1
        #                 x_t = floor(x_tmin1)
        #             end
        #             return x_t
        #         end
        #         p = Det(Tuple{Float64, Symbol}, Float64, f)
        #         parranges = ([100.0], [:F1, :F2, :F3])
        #         pis = ((1.0,), (0.2,0.3,0.5))
        #         @test support(p, parranges, 100, Float64[])==[100.0, 105.0, 121.0]

        #         # test downsampling
        #         samples = support(p, parranges, 2, Float64[])
        #         @test length(samples)==2
        #         @test setdiff(samples, [100.0, 105.0, 121.0]) |> isempty == true # check if samples is contained in the original range

        #         #create network
        #         a0 = Cat([:F1,:F2,:F3], [1.0,0.0,0.0])()(:a0)
        #         b0 = DiscreteCPT([100.0], Dict((:F1,) => [1.0], (:F2,) => [1.0], (:F3,) => [1.0]))()(:b0)
        #         aCPD = Dict((:F1, ) => [0.95, 0.045, 0.005],
        #                         (:F2, ) => [0.95, 0.045, 0.005],
        #                         (:F3, ) => [0.45, 0.1, 0.45])
        #         a1 = DiscreteCPT([:F1,:F2,:F3], aCPD)()(:a1)
        #         b1 = Det(Tuple{Float64, Symbol}, Float64, f)()(:b1)
        #         a2 = DiscreteCPT([:F1,:F2,:F3], aCPD)()(:a2)
        #         b2 = Det(Tuple{Float64, Symbol}, Float64, f)()(:b2)

        #         network = InstantNetwork(Variable[a0,b0,a1,b1,a2,b2], VariableGraph(b0=>[a0], a1=>[a0], b1=>[b0,a1], a2=>[a1], b2=>[b1,a2]))

        #         runtime = Runtime(network)
        #         Scruff.RTUtils.default_initializer(runtime, 1, 3) # limit to 3 possible values of Det
        #         @test length(get_range(runtime, current_instance(runtime,b0))) == 1
        #         @test length(get_range(runtime, current_instance(runtime,b1))) == 3
        #         @test length(get_range(runtime, current_instance(runtime,b2))) == 3

        #         post_evidence!(runtime, current_instance(runtime,a1), HardScore(:F2))

        #         three_pass_BP(runtime)

        #         a0_i = current_instance(runtime,a0)
        #         belief_a0 = get_belief(runtime, a0_i)
        #         #println("belief a0=$belief_a0")
        #         b0_i = current_instance(runtime,b0)
        #         belief_b0 = get_belief(runtime, b0_i)
        #         #println("belief b0=$belief_b0 and range = $(get_value(runtime, current_instance(runtime,b0), :range))")

        #         a1_i = current_instance(runtime,a1)
        #         belief_a1 = get_belief(runtime, a1_i)
        #         #println("belief a1=$belief_a1")
        #         b1_i = current_instance(runtime,b1)
        #         belief_b1 = get_belief(runtime, b1_i)
        #         #println("belief b1=$belief_b1 and range = $(get_value(runtime, current_instance(runtime,b1), :range))")

        #         a2_i = current_instance(runtime,a2)
        #         belief_a2 = get_belief(runtime, a2_i)
        #         #println("belief a2=$belief_a2")
        #         b2_i = current_instance(runtime,b2)
        #         belief_b2 = get_belief(runtime, b2_i)
        #         #println("belief b2=$belief_b2 and range = $(get_value(runtime, current_instance(runtime,b2), :range))")
        #     end

        end

        @testset "If" begin
            c1 = Cat([1, 2], [0.1, 0.9])
            c2 = Cat([1, 2, 3], [0.2, 0.3, 0.5])
            f = Flip(0.4)
            i = If{Int}()
            vc1 = c1()(:c1)
            vc2 = c2()(:c2)
            vf = f()(:f)
            vi = i()(:i)
            net = InstantNetwork(Variable[vc1,vc2,vf,vi], VariableGraph(vi=>[vf, vc1, vc2]))

            @testset "Marginal probabilities without evidence" begin
                run = Runtime(net)
                default_initializer(run)
                three_pass_BP(run)
                bc1 = get_belief(run, current_instance(run, vc1))
                bc2 = get_belief(run, current_instance(run, vc2))
                bf = get_belief(run, current_instance(run, vf))
                bi = get_belief(run, current_instance(run, vi))
                @test isapprox([cpdf(bc1, (), x) for x in [1,2]], [0.1, 0.9])
                @test isapprox([cpdf(bc2, (), x) for x in [1,2,3]], [0.2, 0.3, 0.5])
                @test isapprox([cpdf(bf, (), x) for x in [false, true]], [0.6, 0.4])
                @test isapprox([cpdf(bi, (), x) for x in [1,2,3]], [0.4 * 0.1 + 0.6 * 0.2, 0.4 * 0.9 + 0.6 * 0.3, 0.6 * 0.5])
            end

            @testset "Posterior probabilities with evidence" begin
                run = Runtime(net)
                default_initializer(run)
                post_evidence!(run, current_instance(run, vi), HardScore(1))
                three_pass_BP(run)
                bc1 = get_belief(run, current_instance(run, vc1))
                bc2 = get_belief(run, current_instance(run, vc2))
                bf = get_belief(run, current_instance(run, vf))
                bi = get_belief(run, current_instance(run, vi))
                # The following are the probabilities of joint states consistent with vi = 1
                p11t = 0.1 * 0.2 * 0.4
                p12t = 0.1 * 0.3 * 0.4
                p13t = 0.1 * 0.5 * 0.4
                p11f = 0.1 * 0.2 * 0.6
                p21f = 0.9 * 0.2 * 0.6
                z = p11t + p12t + p13t + p11f + p21f
                @test isapprox([cpdf(bc1, (), x) for x in [1,2]], [(p11t + p12t + p13t + p11f) / z, p21f / z])
                @test isapprox([cpdf(bc2, (), x) for x in [1,2,3]], [(p11t + p11f + p21f) / z, p12t / z, p13t / z])
                @test isapprox([cpdf(bf, (), x) for x in [false, true]], [(p11f + p21f) / z, (p11t + p12t + p13t) / z])
                @test isapprox([cpdf(bi, (), x) for x in [1,2,3]], [1.0, 0.0, 0.0])
            end
        end

    end

    @testset "BP with Apply" begin
        ff1((x,)) = x + x
        ff2((x,)) = x
        f1 = Det(Tuple{Int}, Int, ff1)
        f2 = Det(Tuple{Int}, Int, ff2)
        gg1(x, y) = x + y
        gg2(x, y) = x
        g1 = Det(Tuple{Int,Int}, Int, gg1)
        g2 = Det(Tuple{Int,Int}, Int, gg2)
        h1 = f2
        h2params = Dict((1,) => [0.4,0.6], (2,) => [0.7, 0.3])
        h2 = DiscreteCPT([1,2], h2params)
        xrange = [1,2]
        yrange = [1,2,3]
        xyrange = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
        frange = [f1, f2]
        grange = [g1, g2]
        hrange = [h1, h2]
        xpi = [0.1, 0.9]
        ypi = [0.2, 0.3, 0.5]
        xypi = [0.1 * 0.2, 0.1 * 0.3, 0.1 * 0.5, 0.9 * 0.2, 0.9 * 0.3, 0.9 * 0.5]
        fpi = [0.2, 0.8]
        gpi = [0.3, 0.7]
        hpi = [0.7, 0.3]
        x = Cat(xrange, xpi)
        y = Cat(yrange, ypi)
        xy = Cat(xyrange, xypi)
        f = Cat(frange, fpi)
        g = Cat(grange, gpi)
        h = Cat(hrange, hpi)
        appf = Apply{Tuple{Int}, Int}()
        appg = Apply{Tuple{Int, Int}, Int}()
        apph = appf
        appfrange = [1,2,4]
        appgrange = [1,2,3,4,5]
        apphrange = [1,2]
        lf = [0.1, 0.2, 0.7]
        lg = [0.1, 0.2, 0.25, 0.3, 0.15]
        lh = [0.75, 0.25]
        lamappf = SoftScore(appfrange, lf)
        lamappg = SoftScore(appgrange, lg)
        lamapph = SoftScore(apphrange, lh)

        @testset "compute_pi" begin
            @testset "with one arg parent and deterministic sfuncs" begin
                pi = compute_pi(appf, appfrange, (frange, xrange), (f, x))
                p1 = cpdf(pi, (), 1)
                p2 = cpdf(pi, (), 2)
                p4 = cpdf(pi, (), 4)
                @test isapprox(p1, 0.8 * 0.1)
                @test isapprox(p2, 0.8 * 0.9 + 0.2 * 0.1)
                @test isapprox(p4, 0.2 * 0.9)
            end

            @testset "with two arg parents and deterministic sfuncs" begin
                pi = compute_pi(appg, appgrange, (grange, xyrange), (g, xy))
                p1 = cpdf(pi, (), 1)
                p2 = cpdf(pi, (), 2)
                p3 = cpdf(pi, (), 3)
                p4 = cpdf(pi, (), 4)
                p5 = cpdf(pi, (), 5)
                @test isapprox(p1, gpi[2] * xpi[1]) # Only 1
                @test isapprox(p2, gpi[2] * xpi[2] + gpi[1] * xpi[1] * ypi[1]) # Only 2 or 1+1
                @test isapprox(p3, gpi[1] * (xpi[1] * ypi[2] + xpi[2] * ypi[1])) # 1+2 or 2+1
                @test isapprox(p4, gpi[1] * (xpi[1] * ypi[3] + xpi[2] * ypi[2])) # 1+3 or 2+2
                @test isapprox(p5, gpi[1] * xpi[2] * ypi[3]) # 2+3
            end

            @testset "with stochastic functions" begin
                pi = compute_pi(apph, apphrange, (hrange, xrange), (h, x))
                q1 = cpdf(pi, (), 1)
                q2 = cpdf(pi, (), 2)
                p1given1 = hpi[1] + hpi[2] * h2params[(1,)][1]
                p1given2 = hpi[2] * h2params[(2,)][1]
                p2given1 = hpi[2] * h2params[(1,)][2]
                p2given2 = hpi[1] + hpi[2] * h2params[(2,)][2]
                p1 = xpi[1] * p1given1 + xpi[2] * p1given2
                p2 = xpi[1] * p2given1 + xpi[2] * p2given2
                @test isapprox(q1, p1)
                @test isapprox(q2, p2)
            end
        end

        @testset "compute_lambda" begin
            @testset "on sfunc parent" begin
                @testset "with one arg parent and deterministic functions" begin
                    lam = send_lambda(appf, lamappf, appfrange, (frange, xrange), (f, x), 1)
                    # Possibilities for f1 = x -> x + x:
                    # 1: impossible
                    # 2: x == 1
                    # 4: x == 2
                    l1 = cpdf(x, (), 1) * get_score(lamappf, 2) + cpdf(x, (), 2) * get_score(lamappf, 4)
                    # Possibilities for f2 = x -> x
                    # 1: x == 1
                    # 2: x == 2
                    # 4: impossible
                    l2 = cpdf(x, (), 1) * get_score(lamappf, 1) + cpdf(x, (), 2) * get_score(lamappf, 2)
                    @test isapprox(get_score(lam, f1), l1)
                    @test isapprox(get_score(lam, f2), l2)
                end

                @testset "with two arg parents and deterministic functions" begin
                    xyrange = [(xy[1], xy[2]) for xy in  Utils.cartesian_product([xrange, yrange])]
                    xy = Cat(xyrange, [p[1] * p[2] for p in Utils.cartesian_product([xpi, ypi])])
                    pi = compute_pi(appg, appgrange, (grange, xyrange), (g, xy))
                    lam = send_lambda(appg, lamappg, appgrange, (grange, xyrange), (g, xy), 1)
                    # Possibilities for g1 = (x,y) -> x + y
                    # 1: impossible
                    # 2: x == 1, y == 1
                    # 3: x == 1, y == 2 or x == 2, y == 1
                    # 4: x == 1, y == 3 or x == 2, y == 2
                    # 5: x == 2, y == 3
                    l1 =
                        cpdf(x, (), 1) * cpdf(y, (), 1) * get_score(lamappg, 2) +
                        (cpdf(x, (), 1) * cpdf(y, (), 2) + cpdf(x, (), 2) * cpdf(y, (), 1)) * get_score(lamappg, 3) +
                        (cpdf(x, (), 1) * cpdf(y, (), 3) + cpdf(x, (), 2) * cpdf(y, (), 2)) * get_score(lamappg, 4) +
                        cpdf(x, (), 2) * cpdf(y, (), 3) * get_score(lamappg, 5)
                    # Possibilities for g2 = (x,y) -> x
                    # 1: x == 1, y == anything
                    # 2: x == 2, y == anything
                    # 3-5: impossible
                    l2 = cpdf(x, (), 1) * get_score(lamappg, 1) + cpdf(x, (), 2) * get_score(lamappg, 2)
                    @test isapprox(get_score(lam, g1), l1)
                    @test isapprox(get_score(lam, g2), l2)
                end

                @testset "with stochastic functions" begin
                    lam = send_lambda(apph, lamapph, apphrange, (hrange, xrange), (h, x), 1)
                    # possibilities for h1 = x -> x
                    # 1: x == 1
                    # 2: x == 2
                    l1 = cpdf(x, (), 1) * get_score(lamapph, 1) + cpdf(x, (), 2) * get_score(lamapph, 2)
                    # possibilities for h2 = DiscreteCpt([1,2], h2params)
                    # 1: x == 1 (prob h2params[(1,)][1]) or x == 2 (prob h2params[(2,)][1])
                    # 2: x == 1 (prob h2params[(1,)][2]) or x == 2 (prob h2params[(2,)][2])
                    l2 =
                        (cpdf(x, (), 1) * h2params[(1,)][1] + cpdf(x, (), 2) * h2params[(2,)][1]) * get_score(lamapph, 1) +
                        (cpdf(x, (), 1) * h2params[(1,)][2] + cpdf(x, (), 2) * h2params[(2,)][2]) * get_score(lamapph, 2)
                    @test isapprox(get_score(lam, h1), l1)
                    @test isapprox(get_score(lam, h2), l2)
                end
            end

            @testset "on args parent" begin
                @testset "with one arg parent and deterministic functions" begin
                    lam = send_lambda(appf, lamappf, appfrange, (frange, xrange), (f, x), 2)
                    # Possibilities for x == 1:
                    # 1: f == f2
                    # 2: f == f1
                    # 4: impossible
                    l1 = cpdf(f, (), f2) * get_score(lamappf, 1) + cpdf(f, (), f1) * get_score(lamappf, 2)
                    # Possibilities for x == 2
                    # 1: impossible
                    # 2: f == f2
                    # 4: f == f1
                    l2 = cpdf(f, (), f2) * get_score(lamappf, 2) + cpdf(f, (), f1) * get_score(lamappf, 4)
                    @test isapprox(get_score(lam, 1), l1)
                    @test isapprox(get_score(lam, 2), l2)
                end

                @testset "with two arg parents and deterministic functions" begin
                    lam = send_lambda(appg, lamappg, appgrange, (grange, xyrange), (g, xy), 2)
                    # Possibilities for (1,1)
                    # 1: g == g2
                    # 2: g == g1
                    l11 = cpdf(g, (), g2) * get_score(lamappg, 1) + cpdf(g, (), g1) * get_score(lamappg, 2)
                    # Possibilities for (1,2)
                    # 1: g == g2
                    # 3: g == g1
                    l12 = cpdf(g, (), g2) * get_score(lamappg, 1) + cpdf(g, (), g1) * get_score(lamappg, 3)
                    # Possibilities for (1,3)
                    # 1: g == g2
                    # 4: g == g1
                    l13 = cpdf(g, (), g2) * get_score(lamappg, 1) + cpdf(g, (), g1) * get_score(lamappg, 4)
                    # Possibilities for (2,1)
                    # 2: g == g2
                    # 3: g == g1
                    l21 = cpdf(g, (), g2) * get_score(lamappg, 2) + cpdf(g, (), g1) * get_score(lamappg, 3)
                    # Possibilities for (2,2)
                    # 2: g == g2
                    # 4: g == g1
                    l22 = cpdf(g, (), g2) * get_score(lamappg, 2) + cpdf(g, (), g1) * get_score(lamappg, 4)
                    # Possibilities for (2,3)
                    # 2: g == g2
                    # 5: g == g1
                    l23 = cpdf(g, (), g2) * get_score(lamappg, 2) + cpdf(g, (), g1) * get_score(lamappg, 5)
                    @test isapprox(get_score(lam, (1,1)), l11)
                    @test isapprox(get_score(lam, (1,2)), l12)
                    @test isapprox(get_score(lam, (1,3)), l13)
                    @test isapprox(get_score(lam, (2,1)), l21)
                    @test isapprox(get_score(lam, (2,2)), l22)
                    @test isapprox(get_score(lam, (2,3)), l23)
                end

                @testset "with stochastic functions" begin
                    lam = send_lambda(apph, lamapph, apphrange, (hrange, xrange), (h, x), 2)
                    # possibilities for 1
                    # 1: h == h1 or h == h2 (prob h2params[(1,)][1])
                    # 2: h == h2 (prob h2params[(1,)][2])
                    l1 =
                        (cpdf(h, (), h1) + cpdf(h, (), h2) * h2params[(1,)][1]) * get_score(lamapph, 1) +
                        cpdf(h, (), h2) * h2params[(1,)][2] * get_score(lamapph, 2)
                    # possibilities for 2
                    # 1: h == h2 (prob h2params[(2,)][1])
                    # 2: h == h1 h == h2 (prob h2params[(2,)][2])
                    l2 =
                        cpdf(h, (), h2) * h2params[(2,)][1] * get_score(lamapph, 1) +
                        (cpdf(h, (), h1) + cpdf(h, (), h2) * h2params[(2,)][2]) * get_score(lamapph, 2)
                    @test isapprox(get_score(lam, 1), l1)
                    @test isapprox(get_score(lam, 2), l2)
                end

            end
        end

    end

    @testset "Using the ThreePassBP instant algorithm" begin
        
        @testset "Basic" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            alg = ThreePassBP()
            infer(alg, runtime)
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            @test isapprox(probability(alg, runtime, i1, :a), 0.1)
            @test isapprox(probability(alg, runtime, i1, :b), 0.9)
            @test isapprox(probability(alg, runtime, i2, 1), 0.1 * 0.2 + 0.9 * 0.3)
            @test isapprox(probability(alg, runtime, i2, 2), 0.1 * 0.8 + 0.9 * 0.7)
        end

        @testset "Mean and Variance" begin
            v1 = Cat([4, 34, 18, 12, 2, 26], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6])()(:v1)
            net = InstantNetwork(Variable[v1], VariableGraph())
            runtime = Runtime(net)
            alg = ThreePassBP()
            infer(alg, runtime, Dict{Symbol, Score}())
            i1 = current_instance(runtime, v1)
            @test isapprox(Scruff.Algorithms.mean(alg, runtime, i1), (4 + 34 + 18 + 12 + 2 + 26)/6)
            @test isapprox(Scruff.Algorithms.variance(alg, runtime, i1), ((4-16)^2 + (34-16)^2 + (18-16)^2 + (12-16)^2 + (2-16)^2 + (26-16)^2)/6) 
        end
    
        @testset "With placeholder" begin
            p1 = Placeholder{Symbol}(:p1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v2], VariableGraph(v2 => [p1]), Placeholder[p1])
            runtime = Runtime(net)
            default_initializer(runtime, 10, Dict(p1.name => Cat([:a,:b], [0.1, 0.9])))
            alg = ThreePassBP()
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
            alg = ThreePassBP()
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

        @testset "With intervention" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            alg = ThreePassBP()
            infer(alg, runtime, Dict{Symbol, Score}(), Dict{Symbol, Dist}(:v2 => Constant(2)))
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            @test isapprox(probability(alg, runtime, i1, :a), 0.1)
            @test isapprox(probability(alg, runtime, i1, :b), 0.9)
            @test isapprox(probability(alg, runtime, i2, 1), 0.0)
            @test isapprox(probability(alg, runtime, i2, 2), 1.0)
        end
        
    end

    @testset "Using the LoopyBP instant algorithm" begin
        
        @testset "Basic" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            alg = LoopyBP()
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
            alg = LoopyBP()
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
            alg = LoopyBP()
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

        @testset "With intervention" begin
            v1 = Cat([:a,:b], [0.1, 0.9])()(:v1)
            v2 = DiscreteCPT([1,2], Dict((:a,) => [0.2, 0.8], (:b,) => [0.3, 0.7]))()(:v2)
            net = InstantNetwork(Variable[v1,v2], VariableGraph(v2 => [v1]))
            runtime = Runtime(net)
            default_initializer(runtime)
            alg = LoopyBP()
            infer(alg, runtime, Dict{Symbol, Score}(), Dict{Symbol, Dist}(:v2 => Constant(2)))
            i1 = current_instance(runtime, v1)
            i2 = current_instance(runtime, v2)
            @test isapprox(probability(alg, runtime, i1, :a), 0.1)
            @test isapprox(probability(alg, runtime, i1, :b), 0.9)
            @test isapprox(probability(alg, runtime, i2, 1), 0.0)
            @test isapprox(probability(alg, runtime, i2, 2), 1.0)
        end
        
    end
    
end
