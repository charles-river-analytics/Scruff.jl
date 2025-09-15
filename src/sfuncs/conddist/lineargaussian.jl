export LinearGaussian

"""
    mutable struct LinearGaussian{I <: Tuple{Vararg{Float64}}} <: 
        Conditional{I, Tuple{}, I, Float64, Normal}

`LinearGaussian` defines an sfunc whose mean is a linear function of its parents.  A 
`LinearGaussian`'s output type is a `Float`, its parameter type is 
`Tuple{I, Float64, Float64}`, and it's contained *sfunc* is a `Normal` mean `0.0`.

# Type parameters
- `I`: the input type(s) of the `LinearGaussian`

See also: [`Conditional`](@ref), [`Normal`](@ref)
"""
mutable struct LinearGaussian{I <: Tuple{Vararg{Float64}}} <: Conditional{I, Tuple{}, I, Float64, Normal{Float64}}
    sf :: Normal{Float64}
    params :: Tuple{Tuple{Vararg{Float64}}, Float64, Float64}
    """
        function LinearGaussian(weights :: Tuple{Vararg{Float64}}, bias :: Float64, sd :: Float64)

    `LinearGaussian` constructor

    # Arguments
    - `weights::Tuple{Vararg{Float64}}`: the weights of each parent
    - `bias::Float64`: the bias of the mean of the internal `Normal` *sfunc*
    - `sd::Float64`: the standard deviation of the internal `Normal` *sfunc* 
    """
    function LinearGaussian(weights :: Tuple{Vararg{Float64}}, bias :: Float64, sd :: Float64)
        params = (weights, bias, sd)
        sf = Normal(0.0, sd)
        N = length(weights)
        new{NTuple{N, Float64}}(sf, params)
    end
end

# STATS
@impl begin
    struct LinearGaussianInitialStats end
    function initial_stats(sf::LinearGaussian)
        (weights, _, _) = sf.params
        initweights = Tuple(zeros(length(weights)))
        Dict(initweights=>(0.0, 0.0))
    end
end
# END STATS

function gensf(lg::LinearGaussian, inputs::Tuple{Vararg{Float64}})::Normal{Float64}
    (weights, bias, sd) = lg.params
    to_sum = inputs .* weights
    mean = isempty(to_sum) ? bias : sum(to_sum) + bias
    return Normal(mean, sd)
end
