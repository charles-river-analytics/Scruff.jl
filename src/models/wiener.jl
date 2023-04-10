import Distributions

export WienerProcess

struct WienerProcess
    # A continuous limit of Gaussian random walk
    k::Float64 # "Rate" of random walk. Units of var/time
end

struct CondMuNormal <: SFunc{Float64, Float64}
    # A conditional distribution for a Normal conditioned on mu with fixed var.
    # CondMuNormal_var(mu) = N(mu,var), essentially.
    var::Float64
end

@impl begin
	  struct CondMuNormalSample end
	  function sample(sf::CondMuNormal, x)
	      return rand(Distributions.Normal(x[1], sqrt(sf.var)))
	  end
end

@impl begin
	  struct CondMuNormalExpectation end
	  function expectation(sf::CondMuNormal, x)
	      return x
	  end
end

@impl begin
	  struct CondMuNormalVariance end
	  function variance(sf::CondMuNormal, x)
	      return sf.var
	  end
end

@impl begin
	  struct CondMuNormalMarginalize end
	  function marginalize(sf::CondMuNormal, x)
        mu = expectation(x, tuple())
    	  var = variance(x, tuple()) + sf.var
    	  return Normal(mu, var^0.5)
	  end
end

@impl begin
	  struct CondMuNormalLogcpdf end
	  function logcpdf(sf::CondMuNormal, x, y)
        return Distributions.logpdf(Distributions.Normal(x, sqrt(sf.var)), y)
	  end
end

function make_transition(m::WienerProcess, t0, t1)
	  var = m.k*abs(t1 - t0)
	  return CondMuNormal(var)
end

function create_initial(m::WienerProcess)
	  return Normal(0, 1)
end
