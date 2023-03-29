using Test
using ..Scruff
using ..Scruff.Utils
using ..Scruff.RTUtils
using AbstractTrees
using DataStructures

@testset "Util" begin
    @testset "Cartesian product" begin
        @testset "With no arrays" begin
            @test cartesian_product(Array{Int,1}[]) == [[]]
        end
        @testset "With single array" begin
            @test cartesian_product([[1,2]]) == [[1], [2]]
        end
        @testset "With two arrays" begin
            @test cartesian_product([[1,2], [3,4,5]]) ==
                    [[1,3], [1,4], [1,5], [2,3], [2,4], [2,5]]
        end
        @testset "With empty array" begin
            @test cartesian_product([[1,2], [], [3,4,5]]) == []
        end
    end

    @testset "Topsort" begin
        @testset "Acyclic" begin
            graph = Dict(2 => [1], 5 => [2, 1], 3 => [2], 4 => [1])
            ord = topsort(graph)
            @test length(ord) == 5
            @test 1 in ord
            @test 2 in ord
            @test 3 in ord
            @test 4 in ord
            @test 5 in ord
            i1 = indexin(1, ord)[1]
            i2 = indexin(2, ord)[1]
            i3 = indexin(3, ord)[1]
            i4 = indexin(4, ord)[1]
            i5 = indexin(5, ord)[1]
            @test i1 < i2
            @test i2 < i5
            @test i2 < i3
            @test i1 < i4
        end

        @testset "Cyclic" begin
            graph = Dict(2 => [1, 3], 5 => [2, 1], 3 => [2], 4 => [1])
            ord = topsort(graph)
            @test length(ord) == 5
            @test 1 in ord
            @test 2 in ord
            @test 3 in ord
            @test 4 in ord
            @test 5 in ord
            i1 = indexin(1, ord)[1]
            i2 = indexin(2, ord)[1]
            i3 = indexin(3, ord)[1]
            i4 = indexin(4, ord)[1]
            i5 = indexin(5, ord)[1]
            # i2 and i3 could be in any order, but must be greater than i1 and less than i5
            @test i1 < i2
            @test i1 < i3
            @test i2 < i5
            @test i3 < i5
            @test i1 < i4
        end
    end

    @testset "Continuous utilities" begin
        @testset "Intervals" begin
            @test make_intervals([0, 1, 3]) == [(-Inf,0.5), (0.5,2), (2,Inf)]
            @test make_intervals(([1])) == [(-Inf, Inf)]
        end
        @testset "Linear value" begin
            @test linear_value([2,-3], -1, [4,-2]) == 13
            @test linear_value([], -1, []) == -1
        end
        @testset "Bounded linear value" begin
            is1 = [(3,5), (-3,-1)]
            cs1 = [[3,-1], [3,-3], [5,-1], [5,-3]]
            vs1 = map(x -> linear_value([2,-3], -1, x), cs1)
            @test bounded_linear_value([2,-3], -1, is1) ==
                (minimum(vs1), maximum(vs1))
        end
        @testset "Normal density" begin
            @test isapprox(normal_density(0, 0, 1), 0.3989, atol = 0.0001)
            @test isapprox(normal_density(1, 1, 1), 0.3989, atol = 0.0001)
            @test isapprox(normal_density(-1, 0, 1), 0.2420, atol = 0.0001)
            @test isapprox(normal_density(-2, 0, 4), 0.2420 / 2, atol = 0.0001)
        end
    end

    @testset "Memo" begin
        count = 0
        function f(x)
            count += 1
            return x
        end
        g = memo(f)
        @test g(1) == 1
        @test count == 1
        @test g(1) == 1
        @test count == 1
        @test g(2) == 2
        @test count == 2
    end

    @testset "factor" begin
        fact1 = Factor{1}((2,), (1,), [0.1, 0.9])
        fact2 = Factor{1}((3,), (2,), [0.2, 0.3, 0.5])
        prod12 = Factor{2}((3, 2), (2, 1), [0.02, 0.18, 0.03, 0.27, 0.05, 0.45])
        prod21 = Factor{2}((2, 3), (1, 2), [0.02, 0.03, 0.05, 0.18, 0.27, 0.45])
        fact23 = Factor{2}((3, 2), (2, 3), [0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        prod1223 = Factor{3}((3, 2, 2), (2, 3, 1),
                                [0.004, 0.036, 0.016, 0.144,
                                0.009, 0.081, 0.021, 0.189,
                                0.02, 0.18, 0.03, 0.27])
        prod2123 = prod1223 # Even though the order of variables in one of the
                            # arguments is different, the result is the same
        prod122321 = Factor{3}((2, 3, 2), (1, 2, 3),
                                [0.00008, 0.00032, 0.00027, 0.00063,
                                0.001, 0.0015, 0.00648, 0.02592,
                                0.02187,   0.05103,   0.081,  0.1215])
        sum1 = Factor{2}((3, 2), (2, 3),
                            [0.00008 + 0.00648, 0.00032 + 0.02592, 0.00027 + 0.02187,
                            0.00063 + 0.05103, 0.001 + 0.081, 0.0015 + 0.1215])
        sum2 = Factor{2}((2, 2), (1, 3),
                            [0.00008 + 0.00027 + 0.001, 0.00032 + 0.00063 + 0.0015,
                            0.00648 + 0.02187 + 0.081, 0.02592 + 0.05103 + 0.1215])
        sum3 = Factor{2}((2, 3), (1, 2),
                            [0.00008 + 0.00032, 0.00027 + 0.00063, 0.001 + 0.0015,
                            0.00648 + 0.02592, 0.02187 + 0.05103, 0.081 + 0.1215])
        @testset "Product" begin
            @testset "Multiplying two independent factors" begin
                @test isapprox(product(fact1, fact2), prod12)
                @test isapprox(product(fact2, fact1), prod21)
            end
            @testset "Multiplying two factors with a shared variable" begin
                @test isapprox(product(prod12, fact23), prod1223)
                @test isapprox(product(prod21, fact23), prod2123)
            end
            @testset "Multiplying factors with two shared variables" begin
                @test isapprox(product(prod1223, prod21), prod122321)
            end
        end

        @testset "Summing out a variable" begin
            @testset "With variables" begin
                @test isapprox(sum_over(prod122321, 1), sum1)
                @test isapprox(sum_over(prod122321, 2), sum2)
                @test isapprox(sum_over(prod122321, 3), sum3)
            end
            @testset "When last variable" begin
                f = sum_over(fact1, 1)
                @test isnothing(f.dims)
                @test isnothing(f.keys)
                @test f.entries == [1.0]
            end
        end
    end
end
