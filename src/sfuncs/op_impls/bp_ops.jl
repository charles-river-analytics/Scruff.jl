#=
    bp_ops: operators used in BP that apply to all sfuncs
=#

using ..Utils
using Folds
    
@impl begin
    struct SFuncComputeLambda end

    function compute_lambda(sf::SFunc, range::VectorOption, lambda_msgs::Vector{<:Score})::Score
    
        if isempty(lambda_msgs)
            ps = zeros(Float64, length(range))
        else
            lams = [[get_log_score(l, o) for l in lambda_msgs] for o in range]
            ps = [sum(lams[i]) for i in 1:length(range)]
        end
        # avoid underflow
        m = maximum(ps)
        qs = ps .- m
        return LogScore(range, qs)
    end
end

@impl begin
    struct SFuncComputeBel end
    
    function compute_bel(sf::SFunc{I,O}, range::VectorOption{O},  pi::Dist{O}, lambda::Score)::Dist{<:O} where {I,O}
        ps = [cpdf(pi, (), x) * get_score(lambda, x) for x in range]
        return Cat(range, normalize(ps))
    end
end

@impl begin
    struct SFuncSendPi end
    
    function send_pi(sf::SFunc{I,O}, range::VectorOption{O}, bel::Dist{O}, lambda_msg::Score)::Dist{<:O} where {I,O}
        # pi_msg = [get_score(bel, x) / max.(1e-8, get_score(lambda_msg, x)) for x in range]
        f(x,y) = y == -Inf ? -Inf : x - y
        ps = [f(logcpdf(bel, (), x), get_log_score(lambda_msg, x)) for x in range]
        # delay exponentiation until after avoiding underflow
        m = maximum(ps)
        qs = ps .- m
        exped = exp.(qs)
        return Cat(range, normalize(exped))
    end
end

@impl begin
    struct SFuncOutgoingPis end

    function outgoing_pis(sf::SFunc, range::VectorOption, bel::Dist, 
            incoming_lambdas::VectorOption{<:Score})::Vector{<:Dist}
        if length(incoming_lambdas) == 0
            return Vector{Dist}()
        else
            return [send_pi(sf, range, bel, l) for l in incoming_lambdas]
        end
    end
end

@impl begin
    struct SFuncOutgoingLambdas end

    function outgoing_lambdas(sf::SFunc{I,O},
        lambda::Score,
        range::VectorOption,
        parranges::NTuple{N,Vector},
        incoming_pis::Tuple)::Vector{<:Score} where {N,I,O}
    
        lambdas = Score[]
        for i = 1:length(incoming_pis)
            msg = send_lambda(sf, lambda, range, parranges, incoming_pis, i)
            push!(lambdas, msg)
        end
        return lambdas
    end

end

