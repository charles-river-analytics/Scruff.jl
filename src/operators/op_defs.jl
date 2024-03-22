using ..MultiInterface

export
    importance_sample,
    support_quality,
    support_quality_rank,
    support_quality_from_rank,
    __OptVec,
    __Opt

# These are the new operator definitions where the type signature is specified

"""__OptVec{T} = Union{Vector{Union{}}, Vector{T}}"""
__OptVec{T} = Union{Vector{Union{}}, Vector{T}}
"""__Opt{T} = Union{Nothing, T}"""
__Opt{T} = Union{Nothing, T}

# to support 
MultiInterface.get_imp(::Nothing, args...) = nothing

@interface forward(sf::SFunc{I,O}, i::I)::Dist{O} where {I,O}
@interface inverse(sf::SFunc{I,O}, o::O)::Score{I} where {I,O}
@interface is_deterministic(sf::SFunc)::Bool
@interface sample(sf::SFunc{I,O}, i::I)::O where {I,O}
@interface sample_logcpdf(sf::SFunc{I,O}, i::I)::Tuple{O, AbstractFloat} where {I,O}
# @interface invert(sf::SFunc{I,O}, o::O)::I where {I,O}
@interface lambda_msg(sf::SFunc{I,O}, i::SFunc{<:__Opt{Tuple{}}, O})::SFunc{<:__Opt{Tuple{}}, I} where {I,O}
@interface marginalize(sf::SFunc{I,O}, i::SFunc{<:__Opt{Tuple{}}, I})::SFunc{<:__Opt{Tuple{}}, O} where {I,O}
@interface logcpdf(sf::SFunc{I,O}, i::I, o::O)::AbstractFloat where {I,O}
@interface cpdf(sf::SFunc{I,O}, i::I, o::O)::AbstractFloat where {I,O}
@interface log_cond_prob_plus_c(sf::SFunc{I,O}, i::I, o::O)::AbstractFloat where {I,O}
@interface f_expectation(sf::SFunc{I,O}, i::I, fn::Function) where {I,O}
@interface expectation(sf::SFunc{I,O}, i::I)::O where {I,O}
@interface variance(sf::SFunc{I,O}, i::I)::O where {I,O}
@interface get_score(sf::SFunc{Tuple{I},O}, i::I)::AbstractFloat where {I,O}
@interface get_log_score(sf::SFunc{Tuple{I},O}, i::I)::AbstractFloat where {I,O}

@impl begin
    struct SFuncExpectation end

    function expectation(sf::SFunc{I,O}, i::I) where {I,O}
        return f_expectation(sf, i, x -> x)
    end
end

@interface support(sf::SFunc{I,O}, 
                    parranges::NTuple{N,Vector}, 
                    size::Integer, 
                    curr::Vector{<:O}) where {I,O,N}

function importance_sample end

"""
    support_quality_rank(sq::Symbol)

Convert the support quality symbol into an integer for comparison.
"""
function support_quality_rank(sq::Symbol)
    if sq == :CompleteSupport return 3
    elseif sq == :IncrementalSupport return 2
    else return 1 end
end

"""
    support_quality_from_rank(rank::Int)

Convert the rank back into the support quality.
"""
function support_quality_from_rank(rank::Int)
    if rank == 3 return :CompleteSupport
    elseif rank == 2 return :IncrementalSupport
    else return :BestEffortSupport() end
end

@interface support_quality(sf::SFunc, parranges)

@impl begin
    struct SFuncSupportQuality end

    function support_quality(s::SFunc, parranges)
            :BestEffortSupport
    end
end

@interface bounded_probs(sf::SFunc{I,O}, 
                         range::__OptVec{<:O}, 
                         parranges::NTuple{N,Vector})::Tuple{Vector{<:AbstractFloat}, Vector{<:AbstractFloat}} where {I,O,N}

@interface make_factors(sf::SFunc{I,O},
                        range::__OptVec{<:O}, 
                        parranges::NTuple{N,Vector}, 
                        id, 
                        parids::Tuple)::Tuple{Vector{<:Scruff.Utils.Factor}, Vector{<:Scruff.Utils.Factor}} where {I,O,N}

#= Statistics computation not included in the release
@interface initial_stats(sf::SFunc)

# TODO create an abstract type Stats{I,O}
# (range, parranges, pi's, lambda's)  
@interface expected_stats(sf::SFunc{I,O},
                          range::__OptVec{<:O}, 
                          parranges::NTuple{N,Vector},
                          pis::NTuple{M,Dist},
                          child_lambda::Score{<:O}) where {I,O,N,M}

@interface accumulate_stats(sf::SFunc, existing_stats, new_stats)
@interface maximize_stats(sf::SFunc, stats)
=#
@interface compute_bel(sf::SFunc{I,O},
                      range::__OptVec{<:O}, 
                      pi::Dist{<:O}, 
                      lambda::Score)::Dist{<:O} where {I,O}

@interface compute_lambda(sf::SFunc,
                          range::__OptVec, 
                          lambda_msgs::Vector{<:Score})::Score

@interface send_pi(sf::SFunc{I,O},
                   range::__OptVec{O}, 
                   bel::Dist{O}, 
                   lambda_msg::Score)::Dist{<:O} where {I,O}

@interface outgoing_pis(sf::SFunc,
                        range::__OptVec, 
                        bel::Dist, 
                        incoming_lambdas::__OptVec{<:Score})::Vector{<:Dist}

@interface outgoing_lambdas(sf::SFunc{I,O},
                  lambda::Score,
                  range::__OptVec,
                  parranges::NTuple{N,Vector},
                  incoming_pis::Tuple)::Vector{<:Score} where {N,I,O}

@interface compute_pi(sf::SFunc{I,O},
                     range::__OptVec{O}, 
                     parranges::NTuple{N,Vector}, 
                     incoming_pis::Tuple)::Dist{<:O} where {N,I,O}


@interface send_lambda(sf::SFunc{I,O},
                       lambda::Score,
                       range::__OptVec,
                       parranges::NTuple{N,Vector},
                       incoming_pis::Tuple,
                       parent_idx::Integer)::Score where {N,I,O}

