export
    Expander,
    apply

"""
    mutable struct Expander{I,O} <: SFunc{I,O}

An Expander represents a model defined by a function that returns a
network. For a given value of an input, the conditional probability
distribution is provided by the network produced by the function
on that input.

For each such network, the expander manages a runtime to reason about it.
Expanders are lazy and do not evaluate the function until they have to.

As a result, there is state associated with Expanders. This is analysis
state rather than world state, i.e., it is the state of Scruff's
reasoning about the Expander. In keeping with Scruff design, Expanders
are immutable and all state associated with reasoning is stored in the
runtime that contains the expander. To support this, a runtime has three
fields of global state:

- `:subnets`: the expansions of all Expanders managed by the runtime
- `:subruntimes`: all the subruntimes recursively managed by this
  runtime through Expanders, keyed by the networks
- `:depth`: the depth to which Expanders in this runtime should be expanded

# Type parameters
- `I`: the input type(s) of the `Expander`
- `O`: the output type(s) of the `Expander`
"""
mutable struct Expander{I,O} <: SFunc{I,O}
    # TODO (MRH): SFunc params type parameter
    fn :: Function
    Expander(fn, I, O) = new{I,O}(memo(fn))
end

apply(expander::Expander, args...) = expander.fn(args...) 
