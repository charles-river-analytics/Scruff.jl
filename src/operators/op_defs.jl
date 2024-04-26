using ..MultiInterface

# to support 
MultiInterface.get_imp(::Nothing, args...) = nothing

# Being specific here has big perf implact due to type-stability
const FloatType = Float64

@interface forward(sf::SFunc{I,O}, i::I)::Dist{O} where {I,O}
@interface inverse(sf::SFunc{I,O}, o::O)::Score{I} where {I,O}
@interface is_deterministic(sf::SFunc)::Bool
@interface sample(sf::SFunc{I,O}, i::I)::O where {I,O}
@interface sample_logcpdf(sf::SFunc{I,O}, i::I)::Tuple{O, <:AbstractFloat} where {I,O}
# @interface invert(sf::SFunc{I,O}, o::O)::I where {I,O}
@interface lambda_msg(sf::SFunc{I,O}, i::SFunc{<:Option{Tuple{}}, O})::SFunc{<:Option{Tuple{}}, I} where {I,O}
@interface marginalize(sfb::SFunc{X, Y}, sfa::SFunc{Y, Z})::SFunc{X, Z} where {X, Y, Z}
@interface logcpdf(sf::SFunc{I,O}, i::I, o::O)::FloatType where {I,O}
@interface cpdf(sf::SFunc{I,O}, i::I, o::O)::FloatType where {I,O}
@interface log_cond_prob_plus_c(sf::SFunc{I,O}, i::I, o::O)::AbstractFloat where {I,O}
@interface f_expectation(sf::SFunc{I,O}, i::I, fn::Function) where {I,O}
# Expectation (and others) should either return some continuous relaxation of O (e.g. Ints -> Float) or there should be another op that does
@interface expectation(sf::SFunc{I,O}, i::I) where {I,O} 
@interface variance(sf::SFunc{I,O}, i::I)::O where {I,O}
@interface get_score(sf::SFunc{Tuple{I},O}, i::I)::AbstractFloat where {I,O}
@interface get_log_score(sf::SFunc{Tuple{I},O}, i::I)::AbstractFloat where {I,O}
# Return a new SFunc that is the result of summing samples from each constituent SFunc
@interface sumsfs(fs::NTuple{N, <:SFunc{I, O}})::SFunc{I, O} where {N, I, O}
@interface fit_mle(t::Type{S}, dat::SFunc{I, O})::S where {I, O, S <: SFunc{I, O}}
@interface support_minimum(sf::SFunc{I, O}, i::I)::O where {I, O}
@interface support_maximum(sf::SFunc{I, O}, i::I)::O where {I, O}

@interface support(sf::SFunc{I,O}, 
                   parranges::NTuple{N,Vector}, 
                   size::Integer, 
                   curr::Vector{<:O}) where {I,O,N}

@interface support_quality(sf::SFunc, parranges)

@interface bounded_probs(sf::SFunc{I,O}, 
                         range::VectorOption{<:O}, 
                         parranges::NTuple{N,Vector})::Tuple{Vector{<:AbstractFloat}, Vector{<:AbstractFloat}} where {I,O,N}

@interface make_factors(sf::SFunc{I,O},
                        range::VectorOption{<:O}, 
                        parranges::NTuple{N,Vector}, 
                        id, 
                        parids::Tuple)::Tuple{Vector{<:Scruff.Utils.Factor}, Vector{<:Scruff.Utils.Factor}} where {I,O,N}

@interface get_params(sf::SFunc)

@interface set_params!(sf :: SFunc, params)

# TODO vvvvvv Statistics computation not finished - not using anymore, defined for ConfigurableModel
@interface initial_stats(sf::SFunc)

# TODO create an abstract type Stats{I,O}
# (range, parranges, pi's, lambda's)  
@interface expected_stats(sf::SFunc{I,O},
                          range::VectorOption{<:O}, 
                          parranges::NTuple{N,Vector},
                          pis::NTuple{M,Dist},
                          child_lambda::Score{<:O}) where {I,O,N,M}

@interface accumulate_stats(sf::SFunc, existing_stats, new_stats) 
@interface maximize_stats(sf::SFunc, stats) 

@interface configure(sf::SFunc, config_spec) :: SFunc
# ^^^^ Not finished

@interface compute_bel(sf::SFunc{I,O},
                      range::VectorOption{<:O}, 
                      pi::Dist{<:O}, 
                      lambda::Score)::Dist{<:O} where {I,O}

@interface compute_lambda(sf::SFunc,
                          range::VectorOption, 
                          lambda_msgs::Vector{<:Score})::Score

@interface send_pi(sf::SFunc{I,O},
                   range::VectorOption{O}, 
                   bel::Dist{O}, 
                   lambda_msg::Score)::Dist{<:O} where {I,O}

@interface outgoing_pis(sf::SFunc,
                        range::VectorOption, 
                        bel::Dist, 
                        incoming_lambdas::VectorOption{<:Score})::Vector{<:Dist}

@interface outgoing_lambdas(sf::SFunc{I,O},
                  lambda::Score,
                  range::VectorOption,
                  parranges::NTuple{N,Vector},
                  incoming_pis::Tuple)::Vector{<:Score} where {N,I,O}

@interface compute_pi(sf::SFunc{I,O},
                     range::VectorOption{O}, 
                     parranges::NTuple{N,Vector}, 
                     incoming_pis::Tuple)::Dist{<:O} where {N,I,O}


@interface send_lambda(sf::SFunc{I,O},
                       lambda::Score,
                       range::VectorOption,
                       parranges::NTuple{N,Vector},
                       incoming_pis::Tuple,
                       parent_idx::Integer)::Score where {N,I,O}
