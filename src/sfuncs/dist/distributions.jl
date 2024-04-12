import Distributions

export DistributionsSF

struct DistributionsSF{D <: Distributions.Distribution, O} <: Dist{O}
    dist::D
    function DistributionsSF(dist::D) where {D <: Distributions.Distribution}
        O = eltype(D)
        return new{D, O}(dist)
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
