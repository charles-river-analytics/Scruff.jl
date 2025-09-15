using Test
using Scruff
using Scruff.Operators
using Scruff.MultiInterface
using Scruff.SFuncs
using Scruff.Utils
import Distributions

function test_support(sf::SFunc{I,O}, parranges, target, quality; 
        size = 100, curr = O[]) where {I,O}
    s = support(sf, parranges, size, curr)
    @test Set(s) == Set(target)
    @test support_quality(sf, parranges) == quality
end

function test_sample(sf::SFunc{I,O}, parvals, range, probs;  
        num_samples = 1000, tolerance = 0.1) where {I,O}
    d = Dict{O, Int}()
    for i in 1:num_samples
        x = sample(sf, parvals)
        d[x] = get(d,x,0) + 1
    end
    for (x,p) in zip(range, probs)
        @test isapprox(d[x] / num_samples, p; atol = tolerance)
    end
end

function test_pi(pi, range, probs)
    for (x,p) in zip(range, probs)
        @test isapprox(cpdf(pi, (), x), p)
    end
end

@testset "SFuncs" begin
    
    @testset "Constant" begin
        c = Constant(2)
        test_support(c, (), [2], :CompleteSupport)
        test_sample(c, (), [2], 1.0)
        @test logcpdf(c, (), 2) == 0.0
        @test logcpdf(c, (), 1) == -Inf
        (lfs, ufs) = make_factors(c, [2], (), 1, ())
        @test length(lfs) == 1
        lf = lfs[1]
        @test lf.keys == (1,)
        @test lf.dims == (1,)
        @test lf.entries == [1.0]
        @test ufs == lfs
        #=
        s1 = initial_stats(c)
        @test s1 === nothing
        s2 = expected_stats(c, [2], (), (), SoftScore([2], [1.0]))
        @test s2 === nothing
        s3 = accumulate_stats(c, s1, s2)
        @test s3 === nothing
        ps = maximize_stats(c, s3)
        @test ps === nothing
        =#
        pi = compute_pi(c, [1,2], (), ()) 
        test_pi(pi, [1,2], [0.0, 1.0])
    end

    @testset "Cat" begin
        c = Cat([:a,:b,:c], [0.2, 0.3, 0.5])
        c2 = Cat([:a => 0.2, :b => 0.3, :c => 0.5])
        c3 = Cat([1,1,2], [0.1, 0.3, 0.6]) # must handle duplicates in range correctly
        @test c2.original_range == [:a,:b,:c]
        @test c2.params == [0.2, 0.3, 0.5]
        test_support(c, (), [:a,:b,:c], :CompleteSupport)
        test_support(c3, (), [1,2], :CompleteSupport)
        test_sample(c, (), [:a,:b,:c], [0.2, 0.3, 0.5])
        @test isapprox(logcpdf(c, (), :a), log(0.2))
        @test isapprox(logcpdf(c, (), :b), log(0.3))
        @test isapprox(logcpdf(c, (), :c), log(0.5))
        @test logcpdf(c,(),:d) == -Inf
        @test isapprox(logcpdf(c3, (), 1), log(0.1 + 0.3))
        @test isapprox(logcpdf(c3, (), 2), log(0.6))
        (lfs, ufs) = make_factors(c, [:a, :b, :c], (), 1, ())
        @test length(lfs) == 1
        lf = lfs[1]
        @test lf.keys == (1,)
        @test lf.dims == (3,)
        @test lf.entries == [0.2, 0.3, 0.5]
        @test ufs == lfs
        #=
        @test initial_stats(c) == [0, 0, 0]
        chlam1 = SoftScore(c.range, [0.3, 0.1, 0.2])
        chlam2 = SoftScore(c.range, [0.2, 0.3, 0.1])
        s1 = expected_stats(c, [:a,:b,:c], (), (), chlam1)
        @test isapprox(s1, [0.06, 0.03, 0.1])
        s2 = expected_stats(c, [:a,:b,:c], (), (), chlam2)
        @test isapprox(s2, [0.04, 0.09, 0.05])
        s3 = accumulate_stats(c, s1, s2)
        @test isapprox(s3[1], 0.06 + 0.04)
        @test isapprox(s3[2], 0.03 + 0.09)
        @test isapprox(s3[3], 0.1 + 0.05)
        ps2 = maximize_stats(c, s3)
        z = sum(s3)
        @test isapprox(ps2[1], s3[1] / z)
        @test isapprox(ps2[2], s3[2] / z)
        @test isapprox(ps2[3], s3[3] / z)
        =#
        ps3 = compute_pi(c, [:a, :b, :c], (), ())
        test_pi(ps3, [:a, :b, :c], [0.2, 0.3, 0.5])

        cf = Cat(["abc", "defg"], [0.1, 0.9])
        e = 3 * 0.1 + 4 * 0.9
        @test isapprox(f_expectation(cf, (), length), e)
    end

    @testset "Cat with duplicates" begin
        c = Cat([:a,:b,:a], [0.2, 0.3, 0.5])
        test_support(c, (), [:a,:b], :CompleteSupport)
        test_sample(c, (), [:a,:b], [0.7, 0.3])
    end

    @testset "Flip" begin
        f = Flip(0.7)
        test_support(f, (), [false, true], :CompleteSupport)
        test_sample(f, (), [false, true], [0.3, 0.7])
        (lfs, ufs) = make_factors(f, [false, true], (), 7, ())
        @test length(lfs) == 1
        lf = lfs[1]
        @test lf.keys == (7,)
        @test lf.dims == (2,)
        @test isapprox(lf.entries, [0.3, 0.7])
        @test ufs == lfs
        #=
        @test initial_stats(f) == [0.0, 0.0]
        chlam1 = SoftScore(f.range, [0.2, 0.3])
        chlam2 = SoftScore(f.range, [0.4, 0.5])
        s1 = expected_stats(f, [false, true], (), (), chlam1)
        @test isapprox(s1[1], 0.06)
        @test isapprox(s1[2], 0.21)
        s2 = expected_stats(f, [false, true], (), (), chlam2)
        @test isapprox(s2[1], 0.12)
        @test isapprox(s2[2], 0.35)
        s3 = accumulate_stats(f, s1, s2)
        @test isapprox(s3[1], 0.18)
        @test isapprox(s3[2], 0.56)
        ps = maximize_stats(f, s3)
        @test isapprox(ps[2], 0.56 / 0.74)
        set_params!(f, ps)
        =#
        @test isapprox(compute_pi(f, [false, true], (), ()).params, [0.3, 0.7])
    end

    @testset "Uniform" begin
        u = SFuncs.Uniform(-1.0, 3.0)
        
        test_support(u, (), [-1.0, 0.0, 1.0, 2.0, 3.0], :IncrementalSupport; size = 5)
        test_support(u, (), [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], :IncrementalSupport; 
            size = 9, curr = [-0.5, 0.5, 1.5, 2.5])
        test_support(u, (), [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], :IncrementalSupport; 
            size = 10, curr = [-0.5, 0.5, 1.0, 1.5, 2.5])
        
        cs = [0.0, 0.0, 0.0, 0.0]
        tot = 1000
        for i in 1:tot
            x = sample(u, ())
            cs[Int(floor(x)) + 2] += 1
        end
        for j in 1:4
            @test isapprox(cs[j] / tot, 0.25; atol = 0.1)
        end
        @test isapprox(logcpdf(u, (), 0.0), log(0.25))
        @test isapprox(logcpdf(u, (), 5.0), -Inf64)

        @test isapprox(bounded_probs(u, [-1.0, 0.0, 1.0, 2.0, 3.0], ())[1],
                        [0.125, 0.25, 0.25, 0.25, 0.125])
        @test isapprox(bounded_probs(u, [-1.0, 0.0, 1.0, 2.0, 3.0], ())[2],
                        [0.125, 0.25, 0.25, 0.25, 0.125])
        @test isapprox(bounded_probs(u, [-1.0, -9.0, -8.0], ())[1], [0.0, 0.0, 1.0])
    end
    
    @testset "Normal" begin
        n = SFuncs.Normal(-1.0,1.0)
        dist = Distributions.Normal(-1.0, 1.0)
        empty = Vector{Float64}()

        test_support(n, (), [-1.0], :IncrementalSupport; size = 1)
        test_support(n, (), [-2.0, -1.0, 0.0], :IncrementalSupport; size = 3)
        test_support(n, (), [1.0, 2.0, 3.0, 4.0], :IncrementalSupport; 
            size = 3, curr = [1.0, 2.0, 3.0, 4.0])
        test_support(n, (), [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 
            :IncrementalSupport; size = 5, curr = [1.0, 2.0, 3.0, 4.0])
        range = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 
                4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
        test_support(n, (), range, 
            :IncrementalSupport; size = 15, curr = [1.0, 2.0, 3.0, 4.0])

        c = 0
        tot = 1000
        for i = 1:tot
            if sample(n, ()) < 0.5
                c += 1
            end
        end
        @test isapprox(c / tot, Distributions.cdf(dist, 0.5); atol = 0.05)

        @test isapprox(logcpdf(n, (), 0.5), Distributions.logpdf(dist, 0.5))

        (lfs, ufs) = make_factors(n, range, (), 2, ())
        @test length(lfs) == 1
        @test length(ufs) == 1
        lf = lfs[1]
        uf = ufs[1]
        @test lf.keys == (2,)
        @test uf.keys == lf.keys
        @test lf.dims == (length(range),)
        @test uf.dims == lf.dims
        ls = lf.entries
        us = uf.entries
        @test length(ls) == length(range)
        @test length(us) == length(ls)
        for i = 1:length(range)
            @test ls[i] >= 0
            @test us[i] <= 1
            @test ls[i] <= us[i]
            otherl = 0
            otheru = 0
            for j = 1:length(range)
                if j != i
                    otherl += ls[j]
                    otheru += ls[j]
                end
            end
            @test us[i] <= 1 - otherl
            @test otheru <= 1 - ls[i]
        end

        pi = compute_pi(n, range, (), ())
        probs = [Distributions.pdf(dist, x) for x in range]
        test_pi(pi, range, probs)
    end

    @testset "Det" begin
        f(i :: Float64, j :: Float64) = Int(floor(i+j))
        d = Det(Tuple{Float64, Float64}, Int, f)
        parranges = ([1.1, 2.2], [3.3, 4.4, 5.5])
        pis = (Cat([1.1, 2.2], [0.4, 0.6]), Cat([3.3, 4.4, 5.5], [0.2,0.3,0.5]))
        
        test_support(d, parranges, [4,5,6,7], :CompleteSupport)
        c1 = Constant(1)
        c2 = Constant(4)
        test_sample(d, (1.1, 4.4), [5], 1.0)
        @test logcpdf(d, (1.1, 4.4), 5) == 0.0
        @test logcpdf(d, (1.1, 4.4), 4) == -Inf
        a = [1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0]
        (lfs, ufs) = make_factors(d, [4,5,6,7], parranges, 1, (2, 3))
        @test length(lfs) == 1
        lf = lfs[1]
        @test lf.keys == (2,3,1)
        @test lf.dims == (2,3,4)
        @test lf.entries == a
        @test ufs == lfs
        #=
        s1 = initial_stats(d)
        @test s1 === nothing
        s2 = expected_stats(d, [4,5,6,7], parranges, (), SoftScore(Vector{Int}(), Vector{Float64}()))
        @test s2 === nothing
        s3 = accumulate_stats(d, s1, s2)
        @test s3 === nothing
        ps = maximize_stats(d, s3)
        @test ps === nothing
        set_params!(d, ps)
        =#
        p4 = 0.4 * 0.2
        p5 = 0.4 * 0.3 + 0.6 * 0.2
        p6 = 0.4 * 0.5 + 0.6 * 0.3
        p7 = 0.6 * 0.5
        
        pi1 = compute_pi(d, [4,5,6,7], parranges, pis)
        test_pi(pi1, [4,5,6,7], [p4, p5, p6, p7])

        chlam1 = SoftScore([4,5,6,7], [0.1, 0.2, 0.3, 0.4])
        lam11 = send_lambda(d, chlam1, [4,5,6,7], parranges, pis, 1)
        lam12 = send_lambda(d, chlam1, [4,5,6,7], parranges, pis, 2)
        l11 = 0.2 * 0.1 + 0.3 * 0.2 + 0.5 * 0.3 # pi(v2) * chlam(f(1.1, v2))
        l12 = 0.2 * 0.2 + 0.3 * 0.3 + 0.5 * 0.4 # pi(v2) * chlam(f(2.2, v2))
        l23 = 0.4 * 0.1 + 0.6 * 0.2 # pi(v1) * chlam(f(v1, 3.3))
        l24 = 0.4 * 0.2 + 0.6 * 0.3 # pi(v1) * chlam(f(v1, 4.4))
        l25 = 0.4 * 0.3 + 0.6 * 0.4 # pi(v1) * chlam(f(v1, 5.5))
        @test isapprox(get_score(lam11, 1.1), l11)
        @test isapprox(get_score(lam11, 2.2), l12)
        @test isapprox(get_score(lam12, 3.3), l23)
        @test isapprox(get_score(lam12, 4.4), l24)
        @test isapprox(get_score(lam12, 5.5), l25)

        # test incremental support
        @test issubset([4], support(d, parranges, 3, [4])) == true
        @test issubset([4,5], support(d, parranges, 3, [4,5])) == true
        @test issubset([4,5,6], support(d, parranges, 3, [4,5,6])) == true
        @test issubset([4,5,6,7], support(d, parranges, 3, [4,5,6,7])) == true
        # test size in support
        @test length(support(d, parranges, 3, [4])) == 3
        @test length(support(d, parranges, 3, [4,5])) == 3
        @test length(support(d, parranges, 3, [4,5])) == 3
        @test length(support(d, parranges, 3, [4,5,5,5])) == 3
        @test length(support(d, parranges, 3, [4,5,6,7])) == 4
        @test length(support(d, parranges, 50, collect(1:60))) == 4
        @test length(support(d, parranges, 50, collect(1:30))) == 4
    end

    @testset "DiscreteCPT" begin
        d = Dict((:x,1) => [0.1,0.2,0.7], (:x,2) => [0.2,0.3,0.5], (:x,3) => [0.3,0.4,0.3],
                (:y,1) => [0.4,0.5,0.1], (:y,2) => [0.5,0.1,0.4], (:y,3) => [0.6,0.2,0.2])
        # A bug was uncaught because this range was originally in alphabetical order!
        range = ['c', 'b', 'a'] # note reverse order
        c = DiscreteCPT(range, d)
        pis = ([0.4,0.6], [0.2,0.3,0.5])
        picat1 = Cat([:x, :y], pis[1])
        picat2 = Cat([1,2,3], pis[2])
        picats = (picat1, picat2)
        # Cat range is in arbitrary order so we need to get it directly from the Cat
        parranges = (picat1.__compiled_range, picat2.__compiled_range) 
        ks = collect(keys(d))
        ks1 = unique(first.(ks))
        ks2 = unique(last.(ks))
        test_support(c, parranges, range, :CompleteSupport)
        test_sample(c, (:x,2), ['a', 'b', 'c'], [0.5, 0.3, 0.2]) # note reverse order

        l3 = logcpdf(c, (:x,2), 'b')
        l4 = logcpdf(c, (:x,2), 'd')
        @test isapprox(l3, log(0.3))
        @test isapprox(l4, -Inf)

        k1 = nextkey()
        k2 = nextkey()
        k3 = nextkey()
        (lfs, ufs) = make_factors(c, range, parranges, k1, (k2, k3))
        @test length(ufs) == length(lfs)
        for i = 1:length(ufs)
            @test ufs[i].entries == lfs[i].entries
        end
        @test length(lfs) == 8 # six for the cases and two for the switches
        switchfact1 = lfs[7]
        switchkeys1 = switchfact1.keys
        @test length(switchkeys1) == 2
        @test switchkeys1[1] == k2
        switchkey = switchkeys1[2]
        @test switchfact1.dims == (2,6)
        @test switchfact1.entries ==
            [
                1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            ]
        switchfact2 = lfs[8]
        @test switchfact2.keys == (k3, switchkey)
        @test switchfact2.dims == (3,6)
        @test switchfact2.entries ==
            [
                1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0
            ]
        for i = 1:2
            for j = 1:3
                switchval = (i-1)*3 + j
                infact = lfs[switchval]
                @test infact.keys == (k1, switchkey)
                @test infact.dims == (3,6)
                es = infact.entries
                x = parranges[1][i]
                y = parranges[2][j]
                ps = d[(x,y)]
                for k = 1:3
                    for l = 1:6
                        n = (k-1)*6 + l
                        if l == switchval
                            @test es[n] == ps[k]
                        else
                            @test es[n] == 1.0
                        end
                    end
                end
            end
        end

        chlam1 = SoftScore(range, [0.9,0.8,0.7])
        chlam2 = SoftScore(range, [0.6,0.5,0.4])
        #=
        @test isempty(initial_stats(c))
        
        s1 = expected_stats(c, range, parranges, picats, chlam1)
        
        ks1 = collect(keys(s1))
        @test length(ks1) == 6
        @test (:x,1) in ks1
        @test (:x,2) in ks1
        @test (:x,3) in ks1
        @test (:y,1) in ks1
        @test (:y,2) in ks1
        @test (:y,3) in ks1
        pix2 = cpdf(picats[1], (), :x) * cpdf(picats[2], (), 2)
        nm1 = [cpdf(c, (:x,2), range[i]) * get_score(chlam1, range[i]) for i in 1:3]
        @test isapprox(s1[(:x,2)], pix2 .* nm1)
        
        s2 = expected_stats(c, range, parranges, picats, chlam2)
        nm2 = [cpdf(c, (:x,2), range[i]) * get_score(chlam2, range[i]) for i in 1:3]
        @test isapprox(s2[(:x,2)], pix2 .* nm2)
        s3 = accumulate_stats(c, s1, s2)
        ks3 = collect(keys(s3))
        @test length(ks1) == 6
        @test (:x,1) in ks1
        @test (:x,2) in ks1
        @test (:x,3) in ks1
        @test (:y,1) in ks1
        @test (:y,2) in ks1
        @test (:y,3) in ks1
        @test isapprox(s3[(:x,2)], s1[(:x,2)] .+ s2[(:x,2)])
        ps = maximize_stats(c, s3)
        for i in 1:2
            for j in 1:3
                x = parranges[1][i]
                y = parranges[2][j]
                m = c.inversemaps[1][x]
                n = c.inversemaps[2][y]
                @test ps[(m-1)*3+n] == normalize(s3[(x,y)])
            end
        end
=#
        p = compute_pi(c, range, parranges, picats)
        q1 = pis[1][1] .* (pis[2][1] .* d[(:x, 1)] .+ pis[2][2] .* d[(:x, 2)] .+ pis[2][3] .* d[(:x, 3)])
        q2 = pis[1][2] .* (pis[2][1] .* d[(:y, 1)] .+ pis[2][2] .* d[(:y, 2)] .+ pis[2][3] .* d[(:y, 3)])
        q = q1 .+ q2
        test_pi(p, range, q)

        # FIXME cannot test send_lambda
        l1 = send_lambda(c, chlam1, range, parranges, picats, 1)
        l2 = send_lambda(c, chlam1, range, parranges, picats, 2)
        b1x = 0.0
        b1y = 0.0
        chl1 = [get_score(chlam1, i) for i in range]
        for j = 1:3
            p = cpdf(picats[2], (), parranges[2][j])
            qx = [cpdf(c, (:x, parranges[2][j]), r) for r in range]
            qy = [cpdf(c, (:y, parranges[2][j]), r) for r in range]
            b1x += p * sum(qx .* chl1)
            b1y += p * sum(qy .* chl1)
        end
        @test isapprox(get_score(l1, :x), b1x)
        @test isapprox(get_score(l1, :y), b1y)
        b21 = 0.0
        b22 = 0.0
        b23 = 0.0
        for i = 1:2
            p = cpdf(picats[1], (), parranges[1][i])
            q1 = [cpdf(c, (parranges[1][i], 1), r) for r in range]
            q2 = [cpdf(c, (parranges[1][i], 2), r) for r in range]
            q3 = [cpdf(c, (parranges[1][i], 3), r) for r in range]
            b21 += p * sum(q1 .* chl1)
            b22 += p * sum(q2 .* chl1)
            b23 += p * sum(q3 .* chl1)
        end
        @test isapprox(get_score(l2, 1), b21)
        @test isapprox(get_score(l2, 2), b22)
        @test isapprox(get_score(l2, 3), b23)
    end
    
    @testset "LinearGaussian" begin
        lg = LinearGaussian((-1.0, 1.0, 2.0), 3.0, 1.0)
        pars = ([0.0, 1.0], [2.0], [3.0, 4.0, 5.0])
        v1 = support(lg, pars, 10, Vector{Float64}())
        v2 = support(lg, pars, 100, v1)
        @test support_quality(lg, pars) == :IncrementalSupport
        @test length(v1) >= 10
        @test length(v2) >= 100
        @test all(v -> v in v2, v1)
    end

    @testset "CLG" begin
        d = Dict((:x,1) => ((-1.0, 1.0, 2.0), 3.0, 1.0), (:x,2) => ((-2.0, 4.0, 2.0), 3.0, 1.0),
                (:x,3) => ((-3.0, 2.0, 2.0), 3.0, 1.0), (:y,1) => ((-4.0, 5.0, 2.0), 3.0, 1.0),
                (:y,2) => ((-5.0, 3.0, 2.0), 3.0, 1.0), (:y,3) => ((-6.0, 6.0, 2.0), 3.0, 1.0))
        clg = CLG(d)
        pars = ([:x, :y], [1, 2, 3], [0.0, 1.0], [2.0], [3.0, 4.0, 5.0])
        v1 = support(clg, pars, 10, Vector{Float64}())
        v2 = support(clg, pars, 100, v1)
        v3 = support(clg, pars, 1000, v2)
        @test support_quality(clg, pars) == :IncrementalSupport
        @test length(v1) >= 10
        @test length(v2) >= 100
        @test length(v3) >= 1000
        @test all(v -> v in v2, v1)
        @test all(v -> v in v3, v2)

        # CLG with 1 discrete and 0 continuos parents
        d2 = Dict((:x,) => ((), 0.0, 0.1), (:y,) => ((), 0.5, 0.2), (:z,) => ((), 1.5, 0.2))
        clg2 = CLG(d2)
        pars = ([:x, :y, :z],)
        v2 = support(clg2, pars, 100, Float64[])
    end

    @testset "Mixture" begin
        s1 = Flip(0.9)
        s2 = Cat([true, false], [0.2, 0.8]) # order of values reversed
        mx1 = Mixture([s1, s2], [0.4, 0.6])

        d1 = DiscreteCPT([1,2], Dict(tuple(false) => [0.1, 0.9], tuple(true) => [0.2, 0.8]))
        d2 = DiscreteCPT([1,2], Dict(tuple(false) => [0.3, 0.7], tuple(true) => [0.4, 0.6]))
        mx2 = Mixture([d1, d2], [0.6, 0.4])

        d3 = DiscreteCPT([:a, :b], Dict((1,1) => [0.1, 0.9], (1,2) => [0.2, 0.8], (2,1) => [0.3, 0.7], (2,2) => [0.4, 0.6]))
        d4 = DiscreteCPT([:a, :b], Dict((1,1) => [0.9, 0.1], (1,2) => [0.8, 0.2], (2,1) => [0.7, 0.3], (2,2) => [0.6, 0.4]))
        mx3 = Mixture([d3, d4], [0.6, 0.4])

        vr = support(mx1, tuple(), 100, Vector{Bool}())
        @test support_quality(mx1, tuple()) == :CompleteSupport
        @test length(vr) == 2
        @test false in vr
        @test true in vr

        total = 1000
        n = 0
        for i in 1:total
            if sample(mx1, ())
                n += 1
            end
        end
        pt = 0.3 * 0.2 + 0.7 * 0.6
        @test isapprox(n / total, pt, atol = 0.05)
        
        p1givenf = 0.6 * 0.1 + 0.4 * 0.3
        p1givent = 0.6 * 0.2 + 0.4 * 0.4
        @test isapprox(logcpdf(mx1, tuple(), true), log(pt))
        @test isapprox(logcpdf(mx2, tuple(false), 1), log(p1givenf))
        @test isapprox(logcpdf(mx2, tuple(true), 1), log(p1givent))

        pi = compute_pi(mx2, [1,2], ([false, true],), (Cat([false, true], [0.7, 0.3]),))
        p1 = cpdf(pi, (), 1)
        p2 = cpdf(pi, (), 2)
        @test isapprox(p1, 0.7 * p1givenf + 0.3 * p1givent)
        @test isapprox(p2, 1 - p1)
        p1given11 = 0.6 * 0.1 + 0.4 * 0.9
        p1given12 = 0.6 * 0.2 + 0.4 * 0.8
        p1given21 = 0.6 * 0.3 + 0.4 * 0.7
        p1given22 = 0.6 * 0.4 + 0.4 * 0.6
        q1 = 0.3 * p1given11 + 0.7 * p1given12 # takes into account pi message from parent 2
        q2 = 0.3 * p1given21 + 0.7 * p1given22 # takes into account pi message from parent 2
        lam1 = send_lambda(mx3, 
            SoftScore([:a, :b], [0.4, 0.3]), [:a, :b], ([1,2], [1,2]), 
            (Cat([1,2], [0.1, 0.9]), Cat([1,2], [0.3, 0.7])), 1)
        @test isapprox(get_score(lam1, 1), 0.4 * q1 + 0.3 * (1 - q1))
        @test isapprox(get_score(lam1, 2), 0.4 * q2 + 0.3 * (1 - q2))
    end

    @testset "Separable" begin
        cpt1 = Dict((:x,) => [0.1, 0.9], (:y,) => [0.2, 0.8], (:z,) => [0.3, 0.7])
        cpt2 = Dict((1,) => [0.4, 0.6], (2,) => [0.5, 0.5])
        cpt3 = Dict(('a',) => [0.6, 0.4], ('b',) => [0.7, 0.3])
        s = Separable([false, true], [0.2, 0.3, 0.5], [cpt1, cpt2, cpt3])

        parranges = ([:x, :y, :z], [1, 2], ['a', 'b'])
        myrange = [false, true]
        @test support(s, parranges, 100, Bool[]) == myrange
        @test support_quality(s, parranges) == :CompleteSupport

        parvals = (:y, 1, 'b')
        c1 = Cat([:x, :y, :z], [0.2, 0.3, 0.5])
        c2 = Cat([1, 2], [0.1, 0.9])
        c3 = Cat(['a', 'b'], [0.2, 0.8])
        n = 0
        tot = 1000
        for i = 1:tot
            if sample(s, (:y, 1, 'b'))
                n += 1
            end
        end
        p = 0.2 * 0.8 + 0.3 * 0.6 + 0.5 * 0.3
        @test isapprox(n / tot, p; atol = 0.05)
        @test isapprox(logcpdf(s, parvals, true), log(p))

        k1 = nextkey()
        k2 = nextkey()
        k3 = nextkey()
        k4 = nextkey()
        (lfs,ufs) = make_factors(s, myrange, parranges, k1, (k2, k3, k4))
        @test length(ufs) == length(lfs)
        for i = 1:length(ufs)
            @test ufs[i].entries == lfs[i].entries
        end
        @test length(lfs) == 11 # each component has (num parent values + 1) factors, and one for the mixture
        mixfact = lfs[11]
        @test length(mixfact.keys) == 1
        mixkey = mixfact.keys[1]
        @test mixfact.dims == (3,)
        @test mixfact.entries == [0.2, 0.3, 0.5]
        comp1facts = lfs[1:4]
        comp2facts = lfs[5:7]
        comp3facts = lfs[8:10]
        comp1sw = last(comp1facts)
        comp2sw = last(comp2facts)
        comp3sw = last(comp3facts)
        @test length(comp1sw.keys) == 3
        @test length(comp2sw.keys) == 3
        @test length(comp3sw.keys) == 3
        @test comp1sw.keys[1] == k2
        @test comp2sw.keys[1] == k3
        @test comp3sw.keys[1] == k4
        sw1key = comp1sw.keys[2]
        sw2key = comp2sw.keys[2]
        sw3key = comp3sw.keys[2]
        @test comp1sw.keys[3] == mixkey
        @test comp2sw.keys[3] == mixkey
        @test comp3sw.keys[3] == mixkey
        @test comp1sw.dims == (3,3,3)
        @test comp2sw.dims == (2,2,3)
        @test comp3sw.dims == (2,2,3)
        @test comp1sw.entries ==
        [
            1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
        @test comp2sw.entries ==
        [
            1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 1.0, 1.0, 1.0
        ]
        @test comp3sw.entries ==
        [
            1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 1.0, 1.0, 1.0
        ]
    end

    @testset "Switch" begin
        s1 = LinearSwitch(2, Symbol)
        s2 = If{Symbol}()

        prs1 = ([2,1], [:a, :b], [:c, :b]) # different order
        prs2 = ([false, true], [:a, :b], [:c, :b])
        range = [:c, :b, :a] # different order
        
        v1 = support(s1, prs1, 100, Symbol[])
        v2 = support(s2, prs2, 100, Symbol[])
        @test length(v1) == 3
        @test :a in v1
        @test :b in v1
        @test :c in v1
        @test length(v2) == 3
        @test :a in v2
        @test :b in v2
        @test :c in v2

        ns1 = Dict(:a => 0, :b => 0, :c => 0)
        ns2 = Dict(:a => 0, :b => 0, :c => 0)
        tot = 1000
        for j in 1:tot
            ns1[sample(s1, (2,:a,:b))] += 1
            ns2[sample(s2, (false,:a,:b))] += 1
        end
        @test ns1[:a] == 0
        @test ns1[:b] == tot
        @test ns1[:c] == 0
        @test ns2[:a] == 0
        @test ns2[:b] == tot
        @test ns2[:c] == 0

        @test isapprox(logcpdf(s1, (2,:a,:b), :a), -Inf)
        @test isapprox(logcpdf(s1, (2,:a,:b), :b), 0.0)

        @test support_quality(s1, prs1) == :CompleteSupport
        @test support_quality(s2, prs2) == :CompleteSupport
        @test support_quality(s1, ([2], [:a, :b], [:b, :c])) == :BestEffortSupport
        @test support_quality(s1, ([false], [:a, :b], [:b, :c])) == :BestEffortSupport

        incoming_pis1 = (Cat([2,1], [0.4, 0.6]), Cat([:a, :b], [0.1, 0.9]), Cat([:c, :b], [0.8, 0.2]))
        incoming_pis2 = (Cat([false, true], [0.4, 0.6]), Cat([:a, :b], [0.1, 0.9]), Cat([:c, :b], [0.8, 0.2]))
        pi1 = compute_pi(s1, range, prs1, incoming_pis1)
        pi2 = compute_pi(s2, range, prs2, incoming_pis2)
        p2ac = 0.4 * 0.1 * 0.8
        p2ab = 0.4 * 0.1 * 0.2
        p2bc = 0.4 * 0.9 * 0.8
        p2bb = 0.4 * 0.9 * 0.2
        p1ac = 0.6 * 0.1 * 0.8
        p1ab = 0.6 * 0.1 * 0.2
        p1bc = 0.6 * 0.9 * 0.8
        p1bb = 0.6 * 0.9 * 0.2
        pc = p2ac + p2bc
        pb = p2ab + p2bb + p1bc + p1bb
        pa = p1ac + p1ab
        test_pi(pi1, range, [pc, pb, pa])
        test_pi(pi2, range, [pc, pb, pa])

        chlam = SoftScore(range, [0.1, 0.2, 0.7])
        lam1 = send_lambda(s1, chlam, range, prs1, incoming_pis1, 1)
        lam2 = send_lambda(s1, chlam, range, prs1, incoming_pis1, 2)
        lam3 = send_lambda(s1, chlam, range, prs1, incoming_pis1, 3)
        @test Set(keys(lam1.logscores)) == Set(prs1[1])
        @test Set(keys(lam2.logscores)) == Set(prs1[2])
        @test Set(keys(lam3.logscores)) == Set(prs1[3])
        pac = 0.1 * 0.8 # ignoring first parent for lambda message to first parent
        pab = 0.1 * 0.2
        pbc = 0.9 * 0.8
        pbb = 0.9 * 0.2
        l2ac = pac * 0.1
        l2ab = pab * 0.2
        l2bc = pbc * 0.1
        l2bb = pbb * 0.2
        l1ac = pac * 0.7
        l1ab = pab * 0.7
        l1bc = pbc * 0.2
        l1bb = pbb * 0.2
        l2 = [l2ac + l2bc, l2ab + l2bb]
        l1 = [l1ac + l1ab, l1bc + l1bb]
        @test isapprox(get_score(lam1, 2), sum(l2))
        @test isapprox(get_score(lam1, 1), sum(l1))
        # Since this is a LinearSwitch, parent 2 is the choice for input 1, which is second in the first parent's range
        con2 = 0.4 * sum(l2)
        @test isapprox(get_score(lam2, :a), 0.6 * 0.7 + con2)
        @test isapprox(get_score(lam2, :b), 0.6 * 0.2 + con2)
        # And parent 3 is the choice for input 2, which is first in the first parent's range
        con3 = 0.6 * sum(l1)
        @test isapprox(get_score(lam3, :c), 0.4 * 0.1 + con3)
        @test isapprox(get_score(lam3, :b), 0.4 * 0.2 + con3)
    end

    @testset "Generate" begin
        frng1 = [Flip(0.1), Flip(0.2)]
        f1 = Flip(0.1)
        f2 = Flip(0.2)
        g = Generate{Bool}()
        vs = support(g, ([f1, f2],), 0, Bool[])
        @test support_quality(g, ([f1, f2],)) == :CompleteSupport
        @test length(vs) == 2
        @test Set(vs) == Set([false, true])
        @test isapprox(cpdf(g, (f1,), true), 0.1)
        total = 1000
        n = 0
        for i = 1:total
            if sample(g, (f1,))
                n += 1
            end
        end
        @test isapprox(n / total, 0.1, atol = 0.05)
    end

    @testset "Apply" begin
        frng1 = [Flip(0.1), Flip(0.2)]
        l1 = LinearGaussian((1.0,), 0.0, 1.0)
        l2 = LinearGaussian((1.0,), 1.0, 1.0)
        @test typeof(l2) <: SFunc{<:Tuple{Vararg{Float64}}, Float64}
        frng2 = [l1, l2]
        jrng2 = [(1.0,), (2.0,), (3.0,)]
        a1 = Apply{Tuple{}, Bool}()
        a2 = Apply{Tuple{Float64}, Float64}()
        vs = support(a1, (frng1, Vector{Tuple}[]), 0, Bool[])
        @test support_quality(a1, (frng1, Tuple[])) == :CompleteSupport
        @test support_quality(a2, (frng2, jrng2)) == :IncrementalSupport
        @test length(vs) == 2
        @test Set(vs) == Set([false, true])
        @test isapprox(logcpdf(a2, (l1, (1.0,)), 1.0), log(1 / sqrt(2 * pi)))
        total = 1000
        n = 0
        for i = 1:total
            if sample(a2, (l1, (1.0,))) < 1.0
                n += 1
            end
        end
        @test isapprox(n / total, 0.5, atol = 0.05)
    end

    @testset "Chain" begin
        @testset "With simple I and no J" begin
            sf = Chain(Int, Int, i -> Constant(i+1))
            @test sample(sf, (1,)) == 2
        end

        @testset "With tuple I and no J" begin
            sf = Chain(Tuple{Int}, Int, i -> Constant(i[1]+1))
            @test sample(sf, (1,)) == 2
        end

        @testset "With simple I and tuple J" begin
            sf = Chain(Int, Tuple{Int}, Int, 
                i -> Det(Tuple{Int}, Int, j -> j[1] + i))
            @test sample(sf, (1,2)) == 3
        end
    end

    @testset "Invertible" begin
        i = Invertible{Int,Int}(i -> i + 1, o -> o - 1)
        @test support(i, ([1,2],), 100, Int[]) == [2,3]
        @test support_quality(i, ([1,2],)) == :CompleteSupport
        @test sample(i, (1,)) == 2
        @test cpdf(i, (1,), 2) == 1.0
        @test cpdf(i, (1,), 3) == 0.0
        ps = [1.0,0.0,0.0,0.0,1.0,0.0]
        (bps1, bps2) = Scruff.Operators.bounded_probs(i, [2,3,4], ([1,2],))
        @test bps1 == ps
        @test bps2 == bps1
        (facts1, facts2) = make_factors(i, [2,3,4], ([1,2],), 5, (7,))
        @test facts1 == facts2
        @test length(facts1) == 1
        fact1 = facts1[1]
        @test fact1.keys == (7,5)
        @test fact1.dims == (2,3)
        @test fact1.entries == ps
        parpis = (Cat([1,2], [0.1,0.9]),)
        chlam = SoftScore([2,3,4], [0.2,0.3,0.5])
        pi = compute_pi(i, [2,3,4], ([1,2],), parpis) 
        @test cpdf(pi, (), 2) == 0.1
        @test cpdf(pi, (), 3) == 0.9
        @test cpdf(pi, (), 4) == 0.0
        lam = send_lambda(i, chlam, [2,3,4], ([1,2],), parpis, 1)
        @test get_score(lam, 1) == 0.2
        @test get_score(lam, 2) == 0.3
        @test get_score(lam, 3) == 0.0 # even though it maps to 4, it's not in the parent range
        #=
        @test initial_stats(i) == nothing
        @test accumulate_stats(i, nothing, nothing) == nothing
        @test expected_stats(i, [2,3,4], ([1,2],), parpis, chlam) == nothing
        @test maximize_stats(i, nothing) == nothing
        =#
    end

    @testset "Serial" begin
        sf1 = DiscreteCPT([:a, :b], 
            Dict((1,1) => [0.1, 0.9], (1,2) => [0.2, 0.8], 
                 (2,1) => [0.3, 0.7], (2,2) => [0.4, 0.6]))
        sf2 = DiscreteCPT([false, true], Dict((:a,) => [0.6, 0.4], (:b,) => [0.8, 0.2]))
        sf3 = Invertible{Bool, Int}(b -> b ? 5 : 6, i -> i == 5)
        ser = Serial(Tuple{Int,Int}, Int, (sf1,sf2,sf3))

        total = 1000
        n = 0
        for i in 1:total
            if sample(ser, (1,2)) == 6
                n += 1
            end
        end
        @test isapprox(n / total, 0.2 * 0.6 + 0.8 * 0.8; atol = 0.05)

        prs = ([1,2], [2,1])
        sup = support(ser, prs, 10, Int[]) 
        @test Set(sup) == Set([5,6])
        @test support_quality(ser, prs) == :CompleteSupport

        @test isapprox(cpdf(ser, (1,2), 6), 0.2 * 0.6 + 0.8 * 0.8)

        bps = [
                0.2 * 0.6 + 0.8 * 0.8, 0.2 * 0.4 + 0.8 * 0.2,
                0.1 * 0.6 + 0.9 * 0.8, 0.1 * 0.4 + 0.9 * 0.2,
                0.4 * 0.6 + 0.6 * 0.8, 0.4 * 0.4 + 0.6 * 0.2,
                0.3 * 0.6 + 0.7 * 0.8, 0.3 * 0.4 + 0.7 * 0.2
              ]
        @test isapprox(bounded_probs(ser, [6,5], prs)[1], bps)

        facts = make_factors(ser, [6,5], prs, 3, (1,2))[1]
        fs1 = make_factors(sf1, [:a,:b], prs, 4, (1,2))[1]
        fs2 = make_factors(sf2, [false, true], ([:a,:b],), 5, (4,))[1]
        fs3 = make_factors(sf3, [6,5], ([false, true],), 3, (5,))[1]
        @test length(facts) == length(fs1) + length(fs2) + length(fs3)
        fs = copy(fs1)
        append!(fs, fs2)
        append!(fs, fs3)
        for (computed, actual) in zip(facts, fs)
            @test computed.entries == actual.entries
        end

        parpis = (Cat([1,2], [0.9, 0.1]), Cat([2,1], [0.8, 0.2]))
        pi = compute_pi(ser, [6,5], prs, parpis)
        pi6 = 0.9 * 0.8 * bps[1] + 0.9 * 0.2 * bps[3] + 
                0.1 * 0.8 * bps[5] + 0.1 * 0.2 * bps[7]
        pi5 = 0.9 * 0.8 * bps[2] + 0.9 * 0.2 * bps[4] + 
                0.1 * 0.8 * bps[6] + 0.1 * 0.2 * bps[8]
        @test isapprox(cpdf(pi, (), 6), pi6)
        @test isapprox(cpdf(pi, (), 5), pi5)

        chlam = SoftScore([6,5], [0.7, 0.3])
        lam1 = send_lambda(ser, chlam, [6,5], prs, parpis, 1)
        lam2 = send_lambda(ser, chlam, [6,5], prs, parpis, 2)
        lf = 0.7
        lt = 0.3
        la = 0.6 * lf + 0.4 * lt
        lb = 0.8 * lf + 0.2 * lt
        l11 = 0.8 * (0.2 * la + 0.8 * lb) + 0.2 * (0.1 * la + 0.9 * lb)
        l12 = 0.8 * (0.4 * la + 0.6 * lb) + 0.2 * (0.3 * la + 0.7 * lb)
        @test isapprox(get_score(lam1, 1), l11)
        @test isapprox(get_score(lam1, 2), l12)
        l22 = 0.9 * (0.2 * la + 0.8 * lb) + 0.1 * (0.4 * la + 0.6 * lb)
        l21 = 0.9 * (0.1 * la + 0.9 * lb) + 0.1 * (0.3 * la + 0.7 * lb)
        @test isapprox(get_score(lam2, 1), l21)
        @test isapprox(get_score(lam2, 2), l22)

        #=
        is1 = initial_stats(sf1)
        is2 = initial_stats(sf2)
        is3 = initial_stats(sf3)
        istats = initial_stats(ser)
        @test istats == (is1,is2,is3)

        es1 = expected_stats(sf1, [:a,:b], prs, parpis, SoftScore([:a,:b], [la,lb]))
        pi1 = compute_pi(sf1, [:a,:b], prs, parpis)
        es2 = expected_stats(sf2, [false, true], ([:a,:b],), (pi1,), 
                SoftScore([false, true], [lf,lt]))
        pi2 = compute_pi(sf2, [false, true], ([:a,:b],), (pi1,))
        es3 = expected_stats(sf3, [6,5], ([false, true],), (pi2,), chlam)
        estats = expected_stats(ser, [6,5], prs, parpis, chlam)
        @test length(estats) == 3
        @test keys(estats[1]) == keys(es1)
        for k in keys(estats[1])
            @test isapprox(estats[1][k], es1[k])
        end
        @test keys(estats[2]) == keys(es2)
        for k in keys(estats[2])
            @test isapprox(estats[2][k], es2[k])
        end
        @test estats[3] == es3

        as1 = accumulate_stats(sf1, is1, es1)
        as2 = accumulate_stats(sf2, is2, es2)
        as3 = accumulate_stats(sf3, is3, es3)
        astats = accumulate_stats(ser, istats, estats)
        @test length(astats) == 3
        @test keys(astats[1]) == keys(as1)
        for k in keys(astats[1])
            @test isapprox(astats[1][k], as1[k])
        end
        @test keys(astats[2]) == keys(as2)
        for k in keys(astats[2])
            @test isapprox(astats[2][k], as2[k])
        end
        @test astats[3] == as3

        mp1 = maximize_stats(sf1, as1)
        mp2 = maximize_stats(sf2, as2)
        mp3 = maximize_stats(sf3, as3)
        mparams = maximize_stats(ser, astats)
        @test length(mparams) == 3
        @test isapprox(mparams[1], mp1)
        @test isapprox(mparams[2], mp2)
        @test mparams[3] == mp3
        =#
    end

    @testset "Discrete Distributions.jl" begin
        d = Distributions.Categorical([0.4, 0.3, 0.3])
        sf = DistributionsSF(d)
        N = 1000
        samples = [sample(sf, ()) for _ in 1:N]
        sf_mean = expectation(sf, ())
        @test isapprox(sf_mean, sum(samples) / N; atol=0.15)

        # must handle duplicates in range correctly
        c3 = Discrete([1, 1, 2], 
                      [0.1, 0.3, 0.6]) 
        test_support(c3, (), [1, 2], :CompleteSupport)
        @test isapprox(logcpdf(c3, (), 1), log(0.1 + 0.3))
        @test isapprox(logcpdf(c3, (), 2), log(0.6))
    end

    @testset "Continuous Distributions.jl" begin
        d = Distributions.Normal()
        sf = DistributionsSF(d)
        num_samples = 1024
        samples = [sample(sf, ()) for _ in 1:num_samples]
        @test isapprox(expectation(sf, ()), 0.0)
        @test isapprox(variance(sf, ()), 1.0)
        sf2 = sumsfs((sf, sf))
        @test isapprox(variance(sf2, ()), 2.0)

        cat = Discrete(samples, [1.0/num_samples for _ in 1:num_samples])
        fit_normal = fit_mle(Normal{Float64}, cat)

        @test isapprox(expectation(sf, ()), expectation(fit_normal, ()), atol=0.1)
        @test isapprox(variance(sf, ()), variance(fit_normal, ()), atol=0.1)

        d = Distributions.MvNormal([1.0, 1.0], [1.0 0.5; 0.5 1.0])
        mv_sfunc = DistributionsSF(d)
        sample(mv_sfunc, ())
        sumsfs((mv_sfunc, mv_sfunc))
    end
    
end
