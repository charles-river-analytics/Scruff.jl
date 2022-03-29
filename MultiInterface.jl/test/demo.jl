import Statistics: mean

global policy = Nothing()

# y_n ~ Pr_P(O|I)
@interface sample_n(sf::SFunc{I,O,P}, x::I, n::Int)::Vector{O}
# Becomes:
abstract type SampleN <: Interface end
interface_signature(s::SampleN) = signature
function sample_n(sf::SFunc{I,O,P}, x::I, n::Int)::Vector{O}
    global policy
    return sample_n(policy, sf, x, n)
end
function sample_n(policy::Policy, sf, x, n)
    imp = get_imp(policy, SampleN, sf, x, n)
    return sample_n(imp, sf, x, n)
end

@implement sample_n(sf::AddPlusDiagGaussVecs, x::Tuple{Vector, Vector}, n::Int)
    return x[1] + x[2] +
           rand(Normal(), (size(x[1]), n)).*exp(logstd)
end
#### ^ CONVERTED BY MACRO INTO >
## Struct (e.g. to hold hypers):
struct SampleNABC123 <: SampleN end
impl_signature(o::SampleNABC123) = signature
# And methods:
function sample_n(imp::SampleNABC123, sf::AddPlusDiagGaussVecs, x::Tuple{Vector, Vector}, n::Int)
    #... as above declaration
end
function sample_n(policy::Nothing, sf::AddPlusDiagGaussVecs, x::Tuple{Vector, Vector}, n::Int) = sample_n(SampleNABC123(), sf, x, n)

# When user then calls
# sample_n(sf, x)
# 1) Calls the @op_def, validating the input signature, which calls sample_n_imp,
# 2) sample_n_imp exists for each imp, so it multiple dispatches to get precedence, calling the appropriate sample_n(imp, ...)
# 3) sample_n(imp, ...) contains the actual op_imp definition - you can always call this directly if you know which op_imp you want

### Policies
function with_policy(f, new_policy)
    global policy
    last_policy, policy = policy, new_policy
    result = f() # TODO Exceptions
    policy = last_policy
    return result
end

# Use like:
with_policy(policy) do
    sample_n(sf, x, 10)
end


