export NetworkInput,
       NetworkSFunc
       
struct NetworkInput{T} end

tupler(x::Array) = length(x) == 1 ? x[1] : tuple(x...)
tupler(x::Tuple) = length(x) == 1 ? x[1] : x
tupler(x) = x

get_place_type(p::NetworkInput{T}) where T = T

"""
    struct NetworkSFunc{I,O} <: SFunc{I,O}

An sfunc that combines multiple sfuncs in a network structure.

# Arguments
    input_placeholders A vector of placeholders indicating the types of network inputs. The type parameter `I` is computed from these.
    sfuncs The sfuncs to combine.
    parents A `Dict` that maps sfuncs to their parent sfuncs. Note that this parallels networks, 
         except that we are mapping sfuncs to lists of sfuncs directly rather than variables to vectors of variables.
    output A vector of output sfuncs, determining the `O` type parameter.

# Additional supported operators
- `sample`
- `sample_logcpdf`
- `logcpdf`
"""
struct NetworkSFunc{I,O} <: SFunc{I,O}
    input_placeholders::NTuple{N,NetworkInput} where N
    sfuncs::NTuple{N,SFunc} where N
    parents::Dict{SFunc,Vector} where M
    outputs::Tuple
    """
        function NetworkSFunc(input_placeholders, sfuncs, parents, outputs)

    TODO
    """
    function NetworkSFunc(input_placeholders,
                          sfuncs,
                          parents,
                          outputs)
        in_types = [get_place_type(placeholder) for placeholder in input_placeholders]
        # I = length(in_types) == 1 ? in_types[1] : Tuple{in_types...}
        I = Tuple{in_types...}

        out_types = [output_type(sf) for sf in outputs]
        O = length(out_types) == 1 ? out_types[1] : Tuple{out_types...}

        # TODO (MRH): Validate types of parents
        # TODO (MRH): Propagate param types of constituent SFuncs to type parameter P
        return new{I,O}(input_placeholders,
                                sfuncs,
                                parents,
                                outputs)
    end
end

@impl begin
    struct NetworkSFuncSample end
    function sample(sf::NetworkSFunc{I,O}, input::I)::O where {I,O}
        network = sf
        sample_cache = Dict{Union{SFunc,NetworkInput},Any}(s_inp => walk_inp for (s_inp, walk_inp) in zip(network.input_placeholders, input))

        # Assume network.funcs is topologically sorted. TODO?
        for sfunc in network.sfuncs
            sample_cache[sfunc] = sample(sfunc, Tuple([sample_cache[sinp] for sinp in network.parents[sfunc]]))
        end
        
        s = [sample_cache[o] for o in network.outputs]
        return tupler(s)
    end
end

@impl begin
    struct NetworkSFuncSampleLogcpdf end
    function sample_logcpdf(sf::NetworkSFunc{I,O}, input::I)::Tuple{O, AbstractFloat} where {I,O}
        network = sf
        sample_cache = Dict{Union{SFunc,NetworkInput},Any}(s_inp => (walk_inp, 0.0) for (s_inp, walk_inp) in zip(network.input_placeholders, input))

        # Assume network.funcs is topologically sorted. TODO?
        for sfunc in network.sfuncs
            par_samples = [sample_cache[sinp][1] for sinp in network.parents[sfunc]]
            this_sample, logcpdf = sample_logcpdf(sfunc, Tuple(par_samples))
            logcpdf = logcpdf + sum([sample_cache[sinp][2] for sinp in network.parents[sfunc]])
            sample_cache[sfunc] = (this_sample, logcpdf)
        end
        joint_sample = tuple([sample_cache[o][1] for o in network.outputs]...)
        joint_logcpdf = sum([sample_cache[o][2] for o in network.outputs])

        return (tupler(joint_sample), joint_logcpdf)
    end
end

@impl begin
    struct NetworkSFuncLogcpdf end
    function logcpdf(sf::NetworkSFunc{I,O}, input::I, output::O)::AbstractFloat where {I,O}
        # I think this returns a sample x of a distribution s.t. log(Expectation(exp(x))) gives the logcpdf
        # In the special case where all NetworkSFunc nodes are in outputs then the calculation is deterministic

        sample_cache = merge(Dict{Union{SFunc,NetworkInput},Any}(s_inp => (walk_inp, 0.0) for (s_inp, walk_inp) in zip(sf.input_placeholders, input)),
                        Dict{Union{SFunc,NetworkInput},Any}(s_out => (walk_out, nothing) for (s_out, walk_out) in zip(sf.outputs, output)))

        # Assume network.funcs is topologically sorted. TODO?

        # Is this even right? Seems neat
        for sfunc in sf.sfuncs
            if haskey(sample_cache, sfunc)
                par_samples = [sample_cache[sinp][1] for sinp in sf.parents[sfunc]]
                cumlogcpdf = logcpdf(sfunc, Tuple(par_samples), sample_cache[sfunc][1]) + 
                        sum([sample_cache[sinp][2] for sinp in sf.parents[sfunc]])
                sample_cache[sfunc] = (sample_cache[sfunc][1], cumlogcpdf)
            else
                par_samples = [sample_cache[sinp][1] for sinp in sf.parents[sfunc]]
                this_sample = sample(sfunc, Tuple(par_samples))
                cumlogcpdf = sum([sample_cache[sinp][2] for sinp in sf.parents[sfunc]])
                sample_cache[sfunc] = (this_sample, cumlogcpdf)
            end
        end

        return sum([sample_cache[o][2] for o in sf.outputs])
    end

end
