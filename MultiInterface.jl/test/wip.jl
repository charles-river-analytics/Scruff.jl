module L_
    using MultiInterface
    
    abstract type SFunc{I,O} end
    struct MulVecs{U<:AbstractFloat,V<:AbstractFloat,W<:AbstractFloat} <: SFunc{Tuple{Vector{U}, Vector{V}}, Vector{W}, Nothing} end

    @interface sample(sf::SFunc{I,O}, i::I)::O where {I,O}

    @impl begin
        struct SampleMulVecs end 
        function sample(sf::MulVecs{U,V,W}, x::Tuple{Vector{U}, Vector{V}})::Vector{W} where {U,V,W}
            return x[1] .* x[2]
        end
    end
end
