using Test

import Logging

using Scruff
using Scruff.Operators
using Scruff.MultiInterface: @impl

import Scruff.Operators: sample, forward, cpdf, logcpdf
import Scruff.SFuncs: Constant

# Test default implementation of operators

logger = Logging.SimpleLogger(stderr, Logging.Error+1)

struct SF1 <: SFunc{Tuple{Int},Int} end

@impl begin
    struct SF1Forward end
    function forward(sf::SF1, i::Tuple{Int})::Dist{Int}
        Constant(1)
    end
end

@impl begin
    struct SF1Sample end
    function sample(sf::SF1, i::Tuple{Int})::Int
        0
    end
end

struct SF2 <: SFunc{Tuple{Int},Int} end

@impl begin
    struct SF2Forward end
    function forward(sf::SF2, i::Tuple{Int})::Dist{Int}
        Constant(1)
    end
end

struct SF3 <: SFunc{Tuple{Int}, Int} end

struct SF4 <: SFunc{Tuple{Int}, Int} end

@impl begin
    struct SF4Cpdf end
    function cpdf(sf::SF4, i::Tuple{Int}, o::Int)
        0.0
    end
end

@impl begin
    struct SF4Logcpdf end
    function logcpdf(sf::SF4, i::Tuple{Int}, o::Int)
        0.0 # so cpdf is 1.0
    end
end

struct SF5 <: SFunc{Tuple{Int}, Int} end

@impl begin
    struct SF5Cpdf end
    function cpdf(sf::SF5, i::Tuple{Int}, o::Int)
        0.0
    end
end

struct SF6 <: SFunc{Tuple{Int}, Int} end

@impl begin
    struct SF6Logcpdf end
    function logcpdf(sf::SF6, i::Tuple{Int}, o::Int)
        0.0
    end
end

struct SF7 <: SFunc{Tuple{Int}, Int} end

Logging.with_logger(logger) do

@testset "Implementation of sample using forward" begin
    @testset "When explicit sample defined that is different" begin
        # To test the calls, we implement forward and sample in contradictory ways
        # The explicit sample should be used
        @test sample(SF1(), (2,)) == 0
    end

    @testset "When forward is implemented explicitly but not sample" begin
        # sample should use forward
        @test sample(SF2(), (2,)) == 1
    end

    @testset "When neither sample nor forward are implemented explicitly" begin
        # Should throw a MethodError because forward is not found when using the default implemenation of sample``
        @test_throws MethodError sample(SF3(), (2,))
    end
end

@testset "Default implementations of cpdf and logcpdf in terms of each other" begin
    @testset "When both cpdf and logcpdf are implemented explicitly" begin
        # To test the calls, we implement cpdf and logcpdf in contradictory ways
        @test cpdf(SF4(), (2,), 1) == 0.0
        @test logcpdf(SF4(), (2,), 1) == 0.0
    end

    @testset "When only cpdf is implemented explicitly" begin
        @test cpdf(SF5(), (2,), 1) == 0.0
        @test logcpdf(SF5(), (2,), 1) == -Inf64
    end
    
    @testset "When only cpdf is implemented explicitly" begin
        @test cpdf(SF6(), (2,), 1) == 1.0
        @test logcpdf(SF6(), (2,), 1) == 0.0
    end

    # @testset "When neither cpdf nor logcpdf are implemented explicitly" begin
    #     # should detect infinite loop and throw error
    #     @test_throws ErrorException cpdf(SF7(), (2,), 1)
    #     @test_throws ErrorException logcpdf(SF7(), (2,), 1)
    # end
end
end