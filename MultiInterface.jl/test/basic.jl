@interface a(x::I)::I where {I}
@interface b(f::Vector{I}, x::O) where {I, O}
@interface c(f::Vector{O}) where {O}
@interface d(f)

@impl begin
    function a(x::Int)::Int
        c = 1
        return x + c
    end
end

@impl begin
    struct MyB
        precision::AbstractFloat = 0.1
        dummy::Int = 2
    end

    function b(x::Vector{I}, y::Number) where I
        return x.+y.+precision/2
    end
end

@impl begin 
    function b(x::Vector{I}, y::AbstractFloat) where I
        return x.+y
    end
end

@testset "Basic" begin
    @test a(0) == 1
    @test b([1.,2.,3.],2.) == [3.,4.,5.]
    @test b([1.,2.,3.],2.) == [3.,4.,5.]
    @test all(abs.(b(MyB(0.1, 2),[1.,2.,3.], 2.) .- [3.,4.,5.]) .< 0.1)
    println(methods(a))
    @test list_impls(b) == [MyB, B1]
    @test list_impls(b, Tuple{Vector, Int}) == [MyB]
    @test all(b(MyB(0.1, 2),[1.,2.,3.], 2.) == MyB(0.1,2)([1.,2.,3.],2.))
    println(get_method(MyB))
end

module A_
    using MultiInterface
    
    export four

    @interface one(a::Vector{<:T}, b::T) where {T}
    @interface two(a::Signed, b::AbstractFloat)
    @interface three(a, b::Int64)
    @interface four(a::Vector{<:T})::T where {T}
end

module B_
    using MultiInterface
    import ..A_
    import ..A_: one, two

    @impl begin
        struct One_ end
        function one(a::Vector{<:AbstractFloat}, b::AbstractFloat)
            1.0
        end
        function one(a::Vector{<:Integer}, b::Integer)
            1
        end
        function one(a::Vector{Float32}, b::AbstractFloat)
            2.0
        end

    end

    @impl begin
        struct One_NoPolicy end
        function one(a::Vector{Symbol}, b::Symbol)
            :a
        end
    end

    @impl begin
        struct Two_ end
        
        function two(a::Int64, b::Float32)
            (typeof(a), typeof(b))
        end

        function two(a::Int32, b::Float32)
            (typeof(a), typeof(b))
        end

        function two(a::Int64, b::Float64)
            (typeof(a), typeof(b))
        end

        function two(::Signed, ::AbstractFloat)
            (Signed, AbstractFloat)
        end
    end

    @impl begin
        struct Three_ end
         
        function A_.three(a, b::Int64)
            b
        end 
    end

    error = nothing

    try
        @impl begin
            struct Two_Error end
            
            function two(a::Symbol, b::Float64)
                (typeof(a), typeof(b))
            end 
        end
    catch e
        global error = e
    end

    @impl begin
        struct Four_ end

        function A_.four(a::Vector{Float64})::Float64
            isempty(a) ? 0.0 : a[1]
        end
    end
end

using .A_
import .B_

import MultiInterface: get_imp

struct Policy_One <: Policy end
get_imp(policy::Policy_One, ::Type{A_.One}, args...) = B_.One_()
get_imp(policy::Policy_One, ::Type{A_.Two}, args...) = B_.Two_()

@testset "module testing" begin
    @test A_.one(Float64[1.0], 1.0) == 1.0
    @test A_.one(Int[1], 1) == 1
    @test A_.one([:a], :a) == :a

    @test A_.two(-1, 1.0) == (Int64, Float64)
    @test A_.two(1, 1.0f0) == (Int64, Float32)
    @test A_.two(convert(Int32, 1), 1.0f0) == (Int32, Float32)
    @test A_.two(convert(Int32, 1), 1.0) == (Signed, AbstractFloat)

    @test A_.three(:a, 1) == 1
    @test four([1.0,2.0]) == 1.0
    @test four(Float64[]) == 0.0

    # TODO these no longer error
    # @test B_.error isa ErrorException
    # @test startswith(B_.error.msg,"Invalid definition of two")

    with_policy(Policy_One()) do 
        @test A_.one(Float64[1.0], 1.0) == 1.0
        @test A_.one(Float32[1.0], 1.0) == 2.0
        @test A_.one(Int[1], 1) == 1
        @test_throws MethodError A_.one([:a], :a)

        @test A_.two(-1, 1.0) == (Int64, Float64)
        @test A_.two(1, 1.0f0) == (Int64, Float32)
        @test A_.two(convert(Int32, 1), 1.0f0) == (Int32, Float32)
        @test A_.two(convert(Int32, 1), 1.0) == (Signed, AbstractFloat)
    end
end

