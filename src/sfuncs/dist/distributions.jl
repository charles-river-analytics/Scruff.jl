import Distributions

export DistributionsSF

struct DistributionsSF{D <: Distributions.Distribution, O} <: Dist{O}
    dist::D
    function DistributionsSF(dist::D) where {D <: Distributions.Distribution}
        #O = eltype(D)
        # Why?
        # E.g. https://github.com/JuliaStats/Distributions.jl/issues/1402
        O = typeof(rand(dist))
        return new{D, O}(dist)
    end
    function DistributionsSF{D}(params...) where {D <: Distributions.Distribution}
        d = D(params...)
        return DistributionsSF(d)
    end
    function DistributionsSF{D, O}(params...) where {D <: Distributions.Distribution, O}
        d = D(params...)
        return DistributionsSF(d)
    end
end

@impl begin
  function expectation(sf::DistributionsSF, i::Tuple{})
      return Distributions.mean(sf.dist)
  end
end

@impl begin
    function sample(sf::DistributionsSF, i::Tuple{})
        return Distributions.rand(sf.dist)
    end
end

@impl begin
    function variance(sf::DistributionsSF{D}, i::Tuple{}) where {D <: Distributions.ContinuousDistribution}
        return Distributions.std(sf.dist)^2
    end
end

@impl begin
    function logcpdf(sf::DistributionsSF{D, T}, i::Tuple{}, o::T) where {D, T}
        return Distributions.logpdf(sf.dist, o)
    end
end

@impl begin
    function support_minimum(sf::DistributionsSF, i::Tuple{})
        return Distributions.minimum(sf.dist)
    end
end

@impl begin
    function support_maximum(sf::DistributionsSF, i::Tuple{})
        return Distributions.maximum(sf.dist)
    end
end

@impl begin
    function support(sf::DistributionsSF{<:Distributions.DiscreteDistribution, O}, 
                     parranges::NTuple{N, Vector}, 
                     size::Integer, 
                     curr::Vector{<:O}) where {O, N}
        return Distributions.support(sf.dist)
    end
end

@impl begin
    function support_quality(::DistributionsSF{<:Distributions.DiscreteNonParametric}, parranges)
        :CompleteSupport
    end
end

# See https://juliastats.org/Distributions.jl/stable/convolution/
ConvSupported = Union{Distributions.Bernoulli,
                      Distributions.Binomial,
                      Distributions.NegativeBinomial,
                      Distributions.Geometric,
                      Distributions.Poisson,
                      Distributions.Normal,
                      Distributions.Cauchy,
                      Distributions.Chisq,
                      Distributions.Exponential,
                      Distributions.Gamma,
                      Distributions.MvNormal}

@impl begin
    function sumsfs(fs::NTuple{N, DistributionsSF{SubSF}}) where {N, SubSF <: ConvSupported}
        # Return an SFunc representing g(x) = f1(x) + f2(x) + ...
        # I.e. convolution of the respective densities
        dists = tuple((f.dist for f in fs)...)
        return DistributionsSF(reduce(Distributions.convolve, dists))
    end
end

# This won't work for all Distributions
@impl begin
    struct GeneralFitDistributions end
    function fit_mle(::Type{DistributionsSF{T, O}}, ref::Dist{O}) where {O, T}
        return DistributionsSF(Distributions.fit_mle(T, weighted_values(ref)...))
    end
end

# Some implementations are iterative and you can control iters - lets expose that via hyperparams
IterFitDists = Union{Distributions.Beta,
                     Distributions.Dirichlet,
                     Distributions.DirichletMultinomial,
                     Distributions.Gamma}
@impl begin
    # These defaults are a fair bit looser than Distributions.jl
    struct IterFitDistributions
        maxiter::Int = 100
        tol::Float64 = 1e-6
    end

    function fit_mle(::Type{DistributionsSF{T, O}}, ref::Dist{O}) where {O, T <: IterFitDists}
        return DistributionsSF(Distributions.fit_mle(T, weighted_values(ref)...;
                                                     maxiter=maxiter, tol=tol))
    end
end

