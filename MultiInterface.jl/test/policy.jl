@interface a(x::Int)::Int

@impl begin
    struct A1 end
    function a(x::Int)::Int
        return 1
    end
end

@impl begin
    struct A2
        time::AbstractFloat = 0.01
    end
     
    function a(x::Int)::Int
        sleep(time)
        return 2
    end
end

struct Policy1 <: Policy end
get_imp(policy::Policy1, ::Type{A}, args...) = A1()

struct Policy2 <: Policy end
get_imp(policy::Policy2, ::Type{A}, args...) = A2()

@testset "Policy" begin
    with_policy(Policy1()) do
        @test a(0)==1
    end
    with_policy(Policy2()) do
        @test a(0)==2
    end
    with_policy(Policy1()) do
        @test MultiInterface.__policy == Policy1()
        with_policy(Policy2()) do
            @test MultiInterface.__policy == Policy2()
            @test a(0)==2
        end
        @test MultiInterface.__policy == Policy1()
    end
    with_policy(Policy2()) do
        @test MultiInterface.__policy == Policy2()
        with_policy(Policy1()) do
            @test a(0)==1
        end
        @test MultiInterface.__policy == Policy2()
    end
    @test MultiInterface.__policy === nothing
end
