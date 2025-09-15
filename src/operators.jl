"""
The `Operators` module defines the following interfaces for the following operators:

- `is_deterministic(sf::SFunc)::Bool`
- `sample(sf::SFunc{I,O}, i::I)::O where {I,O}`
- `sample_logcpdf(sf::SFunc{I,O}, i::I)::Tuple{O, AbstractFloat} where {I,O}`
- `invert(sf::SFunc{I,O}, o::O)::I where {I,O}`
- `lambda_msg(sf::SFunc{I,O}, i::SFunc{<:Option{Tuple{}}, O})::SFunc{<:Option{Tuple{}}, I} where {I,O}`
- `marginalize(sf::SFunc{I,O}, i::SFunc{<:Option{Tuple{}}, I})::SFunc{<:Option{Tuple{}}, O} where {I,O}`
- `logcpdf(sf::SFunc{I,O}, i::I, o::O)::AbstractFloat where {I,O}`
- `cpdf(sf::SFunc{I,O}, i::I, o::O)::AbstractFloat where {I,O}`
- `log_cond_prob_plus_c(sf::SFunc{I,O}, i::I, o::O)::AbstractFloat where {I,O}`
- `f_expectation(sf::SFunc{I,O}, i::I, fn::Function) where {I,O}`
- `expectation(sf::SFunc{I,O}, i::I)::O where {I,O}`
- `variance(sf::SFunc{I,O}, i::I)::O where {I,O}`
- `get_config_spec(sf::SFunc{I,O,P})::P where {I,O,P}`
- `set_config_spec!(sf::SFunc{I,O,P}, p::P)::SFunc{I,O,P} where {I,O,P}`
- `get_score(sf::SFunc{Tuple{I},O}, i::I)::AbstractFloat where {I,O}`
- `get_log_score(sf::SFunc{Tuple{I},O}, i::I)::AbstractFloat where {I,O}`
- ```support(sf::SFunc{I,O}, 
                        parranges::NTuple{N,Vector}, 
                        size::Integer, 
                        curr::Vector{<:O}) where {I,O,N}```
- `support_quality(sf::SFunc, parranges)`
- ```bounded_probs(sf::SFunc{I,O}, 
                             range::VectorOption{<:O}, 
                             parranges::NTuple{N,Vector})::Tuple{Vector{<:AbstractFloat}, Vector{<:AbstractFloat}} where {I,O,N}```
- ```make_factors(sf::SFunc{I,O},
                            range::VectorOption{<:O}, 
                            parranges::NTuple{N,Vector}, 
                            id, 
                            parids::Tuple)::Tuple{Vector{<:Scruff.Utils.Factor}, Vector{<:Scruff.Utils.Factor}} where {I,O,N}```
- `initial_stats(sf::SFunc)`
- ```expected_stats(sf::SFunc{I,O},
                              range::VectorOption{<:O}, 
                              parranges::NTuple{N,Vector},
                              pis::NTuple{M,Dist},
                              child_lambda::Score{<:O}) where {I,O,N,M}```
- `accumulate_stats(sf::SFunc, existing_stats, new_stats)`
- `maximize_stats(sf::SFunc, stats)`
- ```compute_bel(sf::SFunc{I,O},
                          range::VectorOption{<:O}, 
                          pi::Dist{<:O}, 
                          lambda::Score{<:O})::Dist{<:O} where {I,O}```
- `compute_lambda(sf::SFunc, range::VectorOption, lambda_msgs::Vector{<:Score})::Score`
- ```send_pi(sf::SFunc{I,O},
                       range::VectorOption{O}, 
                       bel::Dist{O}, 
                       lambda_msg::Score{O})::Dist{<:O} where {I,O}```
- ```outgoing_pis(sf::SFunc,
                            range::VectorOption, 
                            bel::Dist, 
                            incoming_lambdas::VectorOption{<:Score})::Vector{<:Dist}```
- ```outgoing_lambdas(sf::SFunc{I,O},
                      lambda::Score{O},
                      range::VectorOption{O},
                      parranges::NTuple{N,Vector},
                      incoming_pis::Tuple)::Vector{<:Score} where {N,I,O}```
- ```compute_pi(sf::SFunc{I,O},
                         range::VectorOption{O}, 
                         parranges::NTuple{N,Vector}, 
                         incoming_pis::Tuple)::Dist{<:O} where {N,I,O}```
- ```send_lambda(sf::SFunc{I,O},
                           lambda::Score{O},
                           range::VectorOption{O},
                           parranges::NTuple{N,Vector},
                           incoming_pis::Tuple,
                           parent_idx::Integer)::Score where {N,I,O}```
"""
module Operators

using ..Scruff

export
    VectorOption,
    Option

"""VectorOption{T} = Union{Vector{Union{}}, Vector{T}}"""
VectorOption{T} = Union{Vector{Union{}}, Vector{T}}
"""Option{T} = Union{Nothing, T}"""
Option{T} = Union{Nothing, T}

include("operators/op_performance.jl")
include("operators/op_defs.jl")
include("operators/op_impls.jl")

end
