module C_
    using MultiInterface

    abstract type Subby{T} end
    abstract type Subby2{N,T} <: Subby{T} end
    struct SubS{N,T} <: Subby2{N,T} end
    struct SubS2{N,T} <: Subby2{2,SubS{N,T}} end

    Base.print_without_params(x::Type{Subby}) = false
    Base.print_without_params(x::Type{Subby2}) = false
    Base.print_without_params(x::Type{SubS}) = false
    Base.print_without_params(x::Type{SubS2}) = false

    @interface sub(s::Subby{<:T}, t::T)::T where T
    @interface sub2(s::Subby2{2,<:SubS{N,<:T}}, t::T) where {N,T}

    @impl begin
        struct Subby2Sub end
        function sub(s::Subby2{N,T}, t::T)::T where {N,T}
            t
        end
    end

    @impl begin
        struct Subby2Sub1 end
        # This should not break
        function sub(s::Subby2{N,Float64}, t::AbstractFloat)::AbstractFloat where {N}
            t
        end
        
    end

    error = nothing
    try 
        @impl begin
            # this should break
            struct Subby2Sub2 end
            function sub(s::Subby2{N,AbstractFloat}, t::Float64)::Float64 where {N}
                t
            end
        end
    catch e 
        global error = e
    end

    @impl begin
        struct Subby2Sub3 end
        function sub2(s::SubS2{N,T}, t::T)::T where {N,T}
            t
        end
    end
end

@testset "subtype testing" begin
    @test C_.sub(C_.SubS{1,Int64}(), 10) == 10
    @test C_.sub(C_.SubS{1,Symbol}(), :a) == :a
    @test C_.sub(C_.SubS{1, Float64}(), 1.1) == 1.1
    @test C_.sub2(C_.SubS2{10,Int64}(), 2) == 2
    @test C_.error isa ErrorException
    @test occursin("Invalid typing--interface typevars misbound", C_.error.msg)
    @test occursin(r"rule        :  var\"#s\d+\" <: T", C_.error.msg)
    @test occursin("bound rule  :  AbstractFloat <: Float64", C_.error.msg)
end

module D_
    using MultiInterface

    abstract type SupD{S,T} end
    struct SubD{T} <: SupD{Nothing, T} end

    Base.print_without_params(x::Type{SupD}) = false
    Base.print_without_params(x::Type{SubD}) = false

    @interface retval1(x::SubD{T})::T where T

    @impl begin
        struct SubDRetVal1 end
        function retval1(t::SubD{T})::T where T
            T isa Type{<:Integer} && return 5
            T isa Type{Symbol} && return :b
            T
        end
    end

    @interface retval2(x::SubD{<:T})::T where T

    @impl begin
        struct SubDSubRetVal2 end
        function retval2(t::SubD{T})::T where T
            T isa Type{<:Integer} && return 10
            T isa Type{Symbol} && return :a
            T
        end
    end
end

@testset "return value testing" begin
    @test D_.retval1(D_.SubD{Int64}()) == 5
    @test D_.retval1(D_.SubD{Symbol}()) == :b
    @test D_.retval2(D_.SubD{Int64}()) == 10
    @test D_.retval2(D_.SubD{Symbol}()) == :a
end

module E_
    using MultiInterface

    @interface ten(t::T)::T where T
    @interface eleven(x::Vector{T}, t::T) where T
    @interface twelve(x::(Vector{T} where T), t::(T where T))

    tenerror = nothing
    try
        @impl begin
            struct Ten_ end
            function ten(t::Int64)::Float64
                1.0
            end
        end
    catch e
        global tenerror = e
    end

    elevenerror1 = nothing
    try
        @impl begin
            struct Eleven1_ end
            function eleven(x::Vector{T}, t::Float64)::T where T
                t
            end
        end
    catch e
        global elevenerror1 = e
    end

    elevenerror2 = nothing
    try
        @impl begin
            struct Eleven2_ end
            function eleven(x::Vector{Int64}, t::Float64)::Number
                t
            end
        end
    catch e
        global elevenerror2 = e
    end

    @impl begin
        struct Twelve_ end
        function twelve(x::Vector, t::Int64)::Int64
            t
        end
        function twelve(x::Vector{Float64}, t::Int64)
            x
        end
    end
end

@testset "matching and non-matching typevars" begin
    @test E_.tenerror isa ErrorException
    @test contains(E_.tenerror.msg, "Invalid typing--interface typevar bound to multiple types")
    @test contains(E_.tenerror.msg, "typevar:  T")
    @test contains(E_.tenerror.msg, "types  :  Any[:Int64, :Float64]")
    
    @test E_.elevenerror1 isa ErrorException
    @test contains(E_.elevenerror1.msg, "Invalid typing--interface typevar bound to multiple types")
    @test contains(E_.elevenerror1.msg, "typevar:  T")
    @test contains(E_.elevenerror1.msg, "types  :  Any[:T, :Float64]")
    
    @test E_.elevenerror2 isa ErrorException
    @test contains(E_.elevenerror2.msg, "Invalid typing--interface typevar bound to multiple types")
    @test contains(E_.elevenerror2.msg, "typevar:  T")
    @test contains(E_.elevenerror2.msg, "types  :  Any[:Int64, :Float64]")

    @test E_.twelve([:a,:b,"a",1], 10) == 10
    @test E_.twelve([1.0,2.0], 10) == [1.0,2.0]
end

module F_
    using MultiInterface

    @interface thirteen(t::Tuple{AbstractFloat, Signed})
    @interface fourteen(x::Tuple{T}, t::T)::T where T
    @interface fifteen(x::Tuple{S,T}, s::S, t::T) where {S,T}
    @interface sixteen(x::Vector{<:Tuple{S,T}}, s::S, t::T) where {S,T}

    @impl begin
        struct Thirteen_ end
        function thirteen(t::Tuple{Float64, Int64})
            t
        end
    end
    

    @impl begin
        struct Fourteen_ end
        function fourteen(x::Tuple{Symbol}, t::Symbol)::Symbol
            Symbol(string(x[1], t))
        end

        function fourteen(x::Tuple{Tuple{T}}, t::Tuple{T})::Tuple{T} where T
            x[1]
        end
    end

    fourteenerror1 = nothing
    try
        @impl begin
            struct Fourteen_1 end
            function fourteen(x::Tuple{Tuple{T}}, t::T)::T where T
                Symbol(string(x[1], t))
            end
        end
    catch e
        global fourteenerror1 = e
    end

    @impl begin
        struct Fifteen_ end
        function fifteen(x::Tuple{AbstractFloat, Signed}, s::AbstractFloat, t::Signed)
            x
        end
        function fifteen(x::Tuple{Symbol, Symbol}, s::Symbol, t::Symbol)
            x
        end
    end

    fifteenerror1 = nothing
    try
        @impl begin
            struct Fifteen_1 end
            function fifteen(x::Tuple{Float64, Signed}, s::AbstractFloat, t::Signed)
                x
            end
        end
    catch e
        global fifteenerror1 = e
    end

    fifteenerror2 = nothing
    try
        @impl begin
            struct Fifteen_1 end
            function fifteen(x::Tuple, s::Float64, t::Symbol)
                x
            end
        end
    catch e
        global fifteenerror2 = e
    end

    @impl begin
        struct Sixteen_ end
        function sixteen(x::Vector{<:Tuple{AbstractFloat, Signed}}, s::AbstractFloat, t::Signed)
            (s,t)
        end

        function sixteen(x::Vector{Tuple{Float64, Int64}}, s::Float64, t::Int64)
            s
        end

        function sixteen(x::Vector{Tuple{Float32, Int32}}, s::Float32, t::Int32)
            t
        end

        function sixteen(x::Vector{<:Tuple{Float64, B}}, s::Float64, t::B) where B
            B
        end
    end

    sixteenerror1 = nothing
    try
        @impl begin
            struct Sixteen_1 end
            function sixteen(x::Vector{Tuple{Symbol,Symbol}}, s::Symbol, t::Integer)
                x
            end
        end
    catch e
        global sixteenerror1 = e
    end

    sixteenerror2 = nothing
    try
        @impl begin
            struct Sixteen_2 end
            function sixteen(x::Vector{Tuple{A,B}}, s::A, t::Integer) where {A,B}
                x
            end
        end
    catch e
        global sixteenerror2 = e
    end

end

@testset "tuples" begin
    @test F_.thirteen((1.0, 1)) == (1.0, 1)
    @test F_.fourteen((:a,), :b) == :ab
    @test F_.fourteen(((1,),), (2,)) == (1,)
    
    @test F_.fourteenerror1 isa ErrorException
    @test contains(F_.fourteenerror1.msg, "Invalid typing--interface typevar bound to multiple types")
    @test contains(F_.fourteenerror1.msg, "typevar:  T")
    @test contains(F_.fourteenerror1.msg, "types  :  Any[:(Tuple{T}), :T]")

    @test F_.fifteen((1.0, 1), 2.0, 2) == (1.0, 1)
    @test F_.fifteen((:a, :b), :c, :d) == (:a, :b)

    @test F_.fifteenerror1 isa ErrorException
    @test contains(F_.fifteenerror1.msg, "Invalid typing--interface typevar bound to multiple types")
    @test contains(F_.fifteenerror1.msg, "typevar:  S")
    @test contains(F_.fifteenerror1.msg, "types  :  Any[:Float64, :AbstractFloat]")

    @test F_.fifteenerror2 isa ErrorException
    @test contains(F_.fifteenerror2.msg, "Invalid definition of fifteen")
    @test contains(F_.fifteenerror2.msg, "defined :  DataType[Tuple, Float64, Symbol]")
    @test contains(F_.fifteenerror2.msg, "expected:  Type[Tuple{S, T} where {S, T}, Any, Any]")
    @test contains(F_.fifteenerror2.msg, "at      :  Bool[0, 1, 1]")

    @test F_.sixteen(Tuple{Float16, Int16}[(1.0,1)], convert(Float16, 2.0), convert(Int16, 2)) == (2.0,2)
    @test F_.sixteen([(1.0,1)], 2.0, 2) == 2.0
    @test F_.sixteen(Tuple{Float32, Int32}[(1.0,1)], 2.0f0, convert(Int32, 2)) == 2
    @test F_.sixteen([(1.0,"bar")], 2.0, "foo") == String
    @test typeof(F_.sixteenerror1) == ErrorException
    @test contains(F_.sixteenerror1.msg, "Invalid typing--interface typevars misbound")
    @test contains(F_.sixteenerror1.msg, r"var\"#s\d+\" <: T")
    @test contains(F_.sixteenerror1.msg, "bound rule  :  Tuple{Symbol, Symbol} <: Tuple{Symbol, Integer}")
    
    @test typeof(F_.sixteenerror2) == ErrorException
    @test contains(F_.sixteenerror2.msg, "Invalid typing--interface typevars misbound")
    @test contains(F_.sixteenerror2.msg, r"var\"#s\d+\" <: Tuple{S, T}")
    @test contains(F_.sixteenerror2.msg, "bound rule  :  Tuple{A, B} <: Tuple{A, Integer}")
end

module G_
    using MultiInterface

    abstract type A{I,O,P} end
    abstract type B{I,O,P<:AbstractFloat} <: A{I,O,P} end
    abstract type C{I,O} <: B{I,O,Float64} end

    struct D <: C{Symbol, Int64} end
    struct E <: C{Tuple{Symbol, Int64}, String} end
    struct A1{I,O,P} <: A{I,O,P} end 
            
    abstract type AA{I,O,Q,P<:A{I,O,Q}} <: A{I,O,Q} end
    abstract type BB{I,O,Q,P<:A{I,O,Q}} <: A{I,O,P} end
    struct AAA{I,O,Q,P} <: AA{I,O,Q,P} end
    struct CC{D} <: AA{Symbol, Int64, Float64, D} end
    struct DD{D} <: BB{Symbol, Int64, Float64, D} end

    Base.print_without_params(x::Type{A}) = false
    Base.print_without_params(x::Type{B}) = false
    Base.print_without_params(x::Type{C}) = false
    Base.print_without_params(x::Type{D}) = false
    Base.print_without_params(x::Type{E}) = false
    Base.print_without_params(x::Type{A1}) = false
    Base.print_without_params(x::Type{AA}) = false
    Base.print_without_params(x::Type{BB}) = false
    Base.print_without_params(x::Type{AAA}) = false
    Base.print_without_params(x::Type{CC}) = false
    Base.print_without_params(x::Type{DD}) = false

    @interface seventeen(a::A{I,O,P})::O where {I,O,P}
    @interface eighteen(a::B{I,O,P})::P where {I,O,P}
    @interface nineteen(a::AA{I,O,Q,P}, b::I, c::O, d::Q) where {I,O,Q,P<:A{I,O,Q}}

    @impl begin
        struct Seventeen_1 end
        function seventeen(a::B{I,Int64})::Int64 where I
            16
        end
        function seventeen(a::D)::Int64
            17
        end
    end

    @impl begin
        struct Eighteen_1 end
        function eighteen(a::D)::Float64
            18.1
        end
        function eighteen(a::E)::Float64
            18.2
        end
    end

    @impl begin
        struct Nineteen_1 end
        function nineteen(a::AAA{I,O,Q,P}, b::I, c::O, d::Q) where {I,O,Q,P<:A{I,O,Q}}
            (b,c,d)
        end

        function nineteen(a::CC{D}, b::Symbol, c::Int64, d::Float64)::Tuple where {D<:A{Symbol,Int64,Float64}}
            (b,c,d)
        end
    end

    seventeenerror1 = nothing
    try
        @impl begin
            struct Seventeen_2 end
            function seventeen(a::D)::Float64
                16.0
            end
        end
    catch e
        global seventeenerror1 = e
    end

    eightteenerror1 = nothing
    try
        @impl begin
            struct Eightteen_1 end
            function eighteen(a::D)::Int64
                18
            end
        end
    catch e
        global eightteenerror1 = e
    end

    nineteenerror1 = nothing
    try
        @impl begin
            function nineteen(a::AAA, b, c, d)
                (b,c,d)
            end
        end
    catch e
        global nineteenerror1 = e
    end
end

@testset "Complex subtying" begin
    struct BStruct <: G_.B{Symbol, Int64, Float64} end
    struct BStruct2 <: G_.B{Symbol, Symbol, Float64} end

    Base.print_without_params(x::Type{BStruct}) = false
    Base.print_without_params(x::Type{BStruct2}) = false

    @test G_.seventeen(BStruct()) == 16
    @test G_.seventeen(G_.D()) == 17
    @test_throws MethodError G_.seventeen(BStruct2())
    @test typeof(G_.seventeenerror1) == ErrorException

    @test contains(G_.seventeenerror1.msg, "Invalid typing--interface typevar bound to multiple types")
    @test contains(G_.seventeenerror1.msg, "typevar:  O")
    @test contains(G_.seventeenerror1.msg, "types  :  Any[:Int64, :Float64]")
    @test G_.eighteen(G_.D()) == 18.1
    @test G_.eighteen(G_.E()) == 18.2
    
    @test typeof(G_.eightteenerror1) == ErrorException
    @test contains(G_.eightteenerror1.msg, "Invalid typing--interface typevar bound to multiple types")
    @test contains(G_.eightteenerror1.msg, "typevar:  P")
    @test contains(G_.eightteenerror1.msg, "types  :  Any[:Float64, :Int64]")

    @test G_.nineteen(G_.AAA{Int64, Int64, Int64, G_.A1{Int64,Int64,Int64}}(), 1, 2, 3) == (1,2,3)
    @test G_.nineteen(G_.CC{G_.D}(), :a, 1, 1.0) == (:a, 1, 1.0)
    @test_throws MethodError G_.nineteen(G_.CC{G_.D}(), 1, 1, 1.0)

    @test typeof(G_.nineteenerror1) == ErrorException
    @test contains(G_.nineteenerror1.msg, "Invalid definition of nineteen")
    @test contains(G_.nineteenerror1.msg, "defined :  Type[Main.G_.AAA{I, O, Q, P} where {I, O, Q, P}, Any, Any, Any]")
    @test contains(G_.nineteenerror1.msg, "expected:  Type[Main.G_.AA{I, O, Q, P} where {I, O, Q, P<:Main.G_.A{I, O, Q}}, Any, Any, Any]")
    @test contains(G_.nineteenerror1.msg, "defined :  Union{Expr, Symbol}[:(AAA{I, O, Q, P} where {I, O, Q, P}), :Any, :Any, :Any]")
    @test contains(G_.nineteenerror1.msg, "expected:  Union{Expr, Symbol}[:(AA{I, O, Q, P} where P <: Main.G_.A{I, O, Q}), :(I where {I, O, Q, P <: A{I, O, Q}}), :(O where {I, O, Q, P <: A{I, O, Q}}), :(Q where {I, O, Q, P <: A{I, O, Q}})]")
    @test contains(G_.nineteenerror1.msg, "at      :  Bool[0, 1, 1, 1]")

end

module H_
    using MultiInterface

    abstract type A{I,O,P} end
    abstract type B{I,O,P} end
    struct AA{I,O,P} <: A{I,O,P} end
    struct BB{I,O,P} <: B{I,O,P} end

    Base.print_without_params(x::Type{A}) = false
    Base.print_without_params(x::Type{B}) = false
    Base.print_without_params(x::Type{AA}) = false
    Base.print_without_params(x::Type{BB}) = false

    @interface do_a(a::Union{A{Float64}, AA{Float64}})
    @interface do_b(a::Union{B{Float64}, BB{Float64}})
    @interface do_ab(ab::Union{A{Float64}, B{Float64}})
    @interface do_aabb(ab::Union{AA{Float64}, BB{Float64}})

    @interface do_t(t::Union{A{T}, B{T}}) where {T <: Real}

    @impl begin
        struct DoA1 end

        function do_a(a::A{Float64, Int64, Nothing})
            1.0
        end

        function do_a(a::Union{A{Float64, Float64}, A{Float64, Symbol}})
            2.0
        end

        function do_a(a::A{Float64})
            3.0
        end
    end

    doaerror1 = nothing
    try
        @impl begin
            struct DoA2 end

            function do_a(a::Union{A{Float64}, B{Int64}})
                3.0
            end
        end
    catch e
        global doaerror1 = e 
    end


    doaerror2 = nothing
    try
        @impl begin
            struct DoA3 end

            function do_a(a::Union{A{Int64}, A{Float64}})
                4.0
            end
        end
    catch e
        global doaerror2 = e 
    end

    @impl begin
        struct DoT1 end

        function do_t(t::AA{Int64})
            1.0
        end

        function do_t(t::B{Float64})
            2.0
        end

        function do_t(t::A{T}) where {T<:Real}
            3.0
        end

        function do_t(t::B{T}) where {T<:Real}
            4.0
        end
    end

    @impl begin
        struct DoT2 end

        function do_t(t::Union{A{Float64}, B{Int64}})
            5.0
        end
    end

    doterror1 = nothing
    try
        @impl begin
            struct DoT3 end

            function do_t(t::Union{A{Symbol}, B{Int64}})
                5.0
            end
        end
    catch e
        global doterror1 = e
    end
end

@testset "Unions" begin
    @test H_.do_a(H_.AA{Float64, Int64, Nothing}()) == 1.0
    @test H_.do_a(H_.AA{Float64, Float64, Nothing}()) == 2.0
    @test H_.do_a(H_.AA{Float64, Symbol, Nothing}()) == 2.0
    @test H_.do_a(H_.AA{Float64, Nothing, Nothing}()) == 3.0

    @test typeof(H_.doaerror1) == ErrorException
    @test contains(H_.doaerror1.msg, "Invalid definition of do_a")
    @test contains(H_.doaerror1.msg, "defined :  Union[Union{Main.H_.A{Float64, O, P} where {O, P}, Main.H_.B{Int64, O, P} where {O, P}}]")
    @test contains(H_.doaerror1.msg, "expected:  UnionAll[Main.H_.A{Float64, O, P} where {O, P}]")
    @test contains(H_.doaerror1.msg, "defined :  Union{Expr, Symbol}[:(Union{Main.H_.A{Float64, O, P} where {O, P}, Main.H_.B{Int64, O, P} where {O, P}})]")
    @test contains(H_.doaerror1.msg, "expected:  Union{Expr, Symbol}[:(A{Float64, O, P} where {O, P})]")
    @test contains(H_.doaerror1.msg, "at      :  Bool[0]")
   
    @test typeof(H_.doaerror2) == ErrorException
    @test contains(H_.doaerror2.msg, "Invalid definition of do_a")
    @test contains(H_.doaerror2.msg, "defined :  Union[Union{Main.H_.A{Float64, O, P} where {O, P}, Main.H_.A{Int64, O, P} where {O, P}}]")
    @test contains(H_.doaerror2.msg, "expected:  UnionAll[Main.H_.A{Float64, O, P} where {O, P}]")
    @test contains(H_.doaerror2.msg, "defined :  Union{Expr, Symbol}[:(Union{Main.H_.A{Float64, O, P} where {O, P}, Main.H_.A{Int64, O, P} where {O, P}})]")
    @test contains(H_.doaerror2.msg, "expected:  Union{Expr, Symbol}[:(A{Float64, O, P} where {O, P})]")
    @test contains(H_.doaerror2.msg, "at      :  Bool[0]")

    @test H_.do_t(H_.AA{Int64, Symbol, Nothing}()) == 1.0
    @test H_.do_t(H_.BB{Float64, Nothing, Nothing}()) == 2.0
    @test H_.do_t(H_.AA{Integer, Nothing, Nothing}()) == 3.0
    @test H_.do_t(H_.BB{Integer, Nothing, Nothing}()) == 4.0
    @test H_.do_t(H_.BB{Int64, Nothing, Nothing}()) == 5.0

    @test typeof(H_.doterror1) == ErrorException
    @test contains(H_.doterror1.msg, "Invalid definition of do_t")
    @test contains(H_.doterror1.msg, "defined :  Union[Union{Main.H_.A{Symbol, O, P} where {O, P}, Main.H_.B{Int64, O, P} where {O, P}}]")
    @test contains(H_.doterror1.msg, "expected:  UnionAll[Union{Main.H_.A{T, O, P} where {O, P}, Main.H_.B{T, O, P} where {O, P}} where T<:Real]")
    @test contains(H_.doterror1.msg, "defined :  Union{Expr, Symbol}[:(Union{Main.H_.A{Symbol, O, P} where {O, P}, Main.H_.B{Int64, O, P} where {O, P}})]")
    @test contains(H_.doterror1.msg, "expected:  Union{Expr, Symbol}[:(Union{Main.H_.A{T, O, P} where {O, P}, Main.H_.B{T, O, P} where {O, P}} where T <: Real)]")
    @test contains(H_.doterror1.msg, "at      :  Bool[0]")

end

module I_
    __OptVec{T} = Union{Vector{Union{}}, Vector{T}}
    __Opt{T} = Union{Nothing, T}

    using MultiInterface

    abstract type A{I,O,P} end
    abstract type AA{T,P} <: A{Tuple{}, T, P} end
    mutable struct AAA <: AA{Float64, Tuple{Float64, Float64}}
        params::Tuple{Float64, Float64}
    end

    Base.print_without_params(x::Type{A}) = false
    Base.print_without_params(x::Type{AA}) = false
    Base.print_without_params(x::Type{AAA}) = false

    @interface bounded_probs(sf::A{I,O}, 
                            range::__OptVec{<:O}, 
                            parranges::NTuple{N,Vector})::Tuple{Vector{<:AbstractFloat}, Vector{<:AbstractFloat}} where {I,O,P,N}

    @impl begin
        struct BoundedProbs_1 end
        
        function bounded_probs(n::AAA, 
            range::Union{Vector{Union{}}, Vector{Float64}}, 
            parranges::NTuple{N,Vector})::Tuple{Vector{<:AbstractFloat}, Vector{<:AbstractFloat}} where {N}

            ([n.params[1]],[n.params[2]])
        end
    end
end

@testset "Unions with Aliasing" begin
    @test I_.bounded_probs(I_.AAA((1.0,2.0)), Union{}[], ([1.0],)) == ([1.0],[2.0])
end

module K_
    using MultiInterface
    struct X end
    
    @interface myx(x::K_.X)::Vector{<:K_.X}

    @impl begin
        struct MyX_1 end
        function myx(x::X)::Vector{<:K_.X}
            [x]
        end   
    end

    abstract type SFunc{I,O} end
    Base.print_without_params(x::Type{<:SFunc}) = false

    abstract type Dist{T} <: SFunc{Tuple{}, T} end
    abstract type Score{I} <: SFunc{I,Nothing} end
    struct SC <: Score{Float64} end
    struct D <: Dist{Float64} end

    @interface send_lambda(sf::SFunc{I,O},
                        lambda::Score{<:O},
                        range::Union{Vector{Union{}}, Vector{<:O}},
                        parranges::NTuple{N,Vector},
                        incoming_pis::Union{Vector{Union{}}, Vector{<:Dist}},
                        parent_idx::Integer)::Score where {N,I,O}

    @impl begin
        function send_lambda(sf::Dist{T},
                            lambda::Score{<:T},
                            range::Union{Vector{Union{}}, Vector{<:T}},
                            parranges::NTuple{N,Vector},
                            incoming_pis::Union{Vector{Union{}}, Vector{<:Dist}},
                            parent_idx::Integer)::Score where {N,T}
            lambda
        end
    end

@testset "Arguments with module specification" begin
    @test K_.myx(K_.X()) == [K_.X()]
    @test K_.send_lambda(K_.D(), K_.SC(), Float64[], (), Union{}[], 1) == K_.SC()
end
