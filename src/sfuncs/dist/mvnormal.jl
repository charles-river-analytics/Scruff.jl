export MvNormal

import Distributions

function MvNormal(mean :: Vector{T}, cov :: Matrix{T}) where T <: Real 
    mvn = Distributions.MvNormal(mean, cov)
    DistributionsSF(mvn, Vector{T})
end
