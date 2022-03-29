using Test
using Scruff
using Scruff.Operators
using Scruff.SFuncs
using Scruff.Utils

@testset "Score" begin

    @testset "Computing score" begin

        @testset "Hard score" begin
            s = HardScore(2)
            @test get_score(s, 1) == 0.0
            @test get_score(s, 2) == 1.0
            @test get_log_score(s, 1) == -Inf
            @test get_log_score(s, 2) == 0.0
        end

        @testset "Soft score" begin
            s = SoftScore([:a, :b], [0.1, 0.2])
            @test isapprox(get_score(s, :a), 0.1)
            @test isapprox(get_score(s, :b), 0.2)
            @test get_score(s, :c) == 0.0
            @test isapprox(get_log_score(s, :a), log(0.1))
            @test isapprox(get_log_score(s, :b), log(0.2))
            @test get_log_score(s, :c) == -Inf
        end     

        @testset "Log score" begin
            s = LogScore([:a, :b], [-1.0, -2.0])
            @test isapprox(get_score(s, :a), exp(-1.0))
            @test isapprox(get_score(s, :b), exp(-2.0))
            @test get_log_score(s, :a) == -1.0
            @test get_log_score(s, :b) == -2.0
        end

        @testset "Functional score" begin
            f(x) = 1.0 / x
            s = FunctionalScore{Float64}(f)
            @test isapprox(get_score(s, 2.0), 0.5)
            @test isapprox(get_log_score(s, 2.0), log(0.5))
        end

        @testset "Normal score" begin
            s = NormalScore(1.0, 2.0)
            r = normal_density(3.0, 1.0, 2.0)
            @test isapprox(get_score(s, 3.0), r)
            @test isapprox(get_log_score(s, 3.0), log(r))
        end

        @testset "Parzen score" begin
            s = Parzen([-1.0, 1.0], 2.0)
            r1 = normal_density(0.5, -1.0, 2.0)
            r2 = normal_density(0.5, 1.0, 2.0)
            ans = 0.5 * (r1 + r2)
            @test isapprox(get_score(s, 0.5), ans)
            @test isapprox(get_log_score(s, 0.5), log(ans))
        end
        
    end
end