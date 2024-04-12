struct SumSF{I, O, SFs <: NTuple{<:Number, <: SFunc{I, O}}} <: SFunc{I, O}
    sfs::SFs
end

@impl begin
  function sumsfs(fs::NTuple{N, <:SFunc}) where {N}
        # Return an SFunc representing g(x) = f1(x) + f2(x) + ...
        # I.e. convolution of the respective densities
        return SumSF(fs)
    end
end

@impl begin
    function sample(sf::SumSF, x)
        return sum(sample(sub_sf, x) for sub_sf in sf.sfs)
    end
end
