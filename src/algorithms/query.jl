export
    Query,
    Queryable,
    Marginal,
    Joint,
    ProbValue,
    ProbFunction,
    ProbabilityBounds,
    Expectation,
    Mean,
    Variance,
    answer,
    marginal,
    joint,
    probability,
    probability_bounds,
    expectation,
    mean

"""
    abstract type Query end

General type of query that can be answered after running an algorithm.
"""
abstract type Query end

"""
A query target is either a variable instance or a variable.
Allowing queries to be defined in terms of instances rather than variables makes it possible
to ask queries across multiple instances of a variable at different times.
However, in many cases the current instance of the variable(s) is required and then it is easier
to use variables.
"""
Queryable{O} = Union{VariableInstance{O}, Variable{I,J,O}} where {I,J}

function _resolve(r::Runtime, q::Queryable{O})::VariableInstance{O} where O
    if q isa Instance
        return q
    else
        return current_instance(r, q)
    end
end

"""
    answer(::Query, ::Algorithm, ::Runtime, ::VariableInstance)
    answer(::Query, ::Algorithm, ::Runtime, ::Vector{VariableInstance})
    answer(::Query, ::Algorithm, ::Runtime, ::Queryable)
    answer(::Query, ::Algorithm, ::Runtime, ::Vector{Queryable})

    Answer the query.

    An implementation of an algorithm should implement an `answer` method for any queries
    it can handle. The type hierarchies of `Query` and `Algorithm` will enable
    query answering methods to be used wherever appropriate with the right specialization.
    The implementations of `answer` are differentiated along two dimensions:
    - single or multiple items
    - queryable items in general or specifically instances

    It is expected that an algorithm will implement one of the first two methods for queries it
    can handle. I.e., an algorithm is expected to handle a single instance or a vector of instances.
    If it can handle multiple instances, it should implement a second method and a single instance implementation
    is derived by default using a singleton vector. An algorithm can still override this default
    method if it handles single instances differently from multiple.

    Algorithms will generally not implement the latter two methods, which are provide for convenience. 
    Default implementations are provided that delegate to the instance-specific methods.

    Defining a very high-level default implementation that throws an error enables implementations
    to go through sequences of preferences.
"""
function answer(q::Query, a::Algorithm, r::Runtime, i::VariableInstance)
    is = VariableInstance[i]
    answer(q, a, r, is)
end

function answer(q::Query, a::Algorithm, r::Runtime, item::Queryable)
    answer(q, a, r, _resolve(r,item))
end

function answer(q::Query, a::Algorithm, r::Runtime, items::Vector{Queryable})
    insts = VariableInstance[]
    for i in items
        inst::VariableInstance = _resolve(r,i)
        push!(insts, inst)
    end
    answer(q, a, r, insts)
end

answer(::Any, ::Any, ::Any, ::Any) = error("_answer_")

""

struct Marginal <: Query end

"""
    marginal(alg::Algorithm, runtime::Runtime, item::Queryable{O})::Union{Dist{O}, Tuple{Dist{O}, Dist{O}}} where O

Return the marginal distribution over `item`, or lower and upper marginals,
depending on the algorithm.

The returned `Score` assigns a score to each value of `item`.
"""
function marginal(alg::Algorithm, run::Runtime, item::Queryable{O})::Union{Dist{O}, Tuple{Dist{O}, Dist{O}}} where O
    mg = answer(Marginal(), alg, run, item)
    return mg
end

struct Joint <: Query end

"""
    joint(alg::Algorithm, run::Runtime, items::Vector{Queryable})::Union{Score{O}, Tuple{Score{O}, Score{O}}}

Return the joint distribution over `items`, or lower and upper distributions,
depending on the algorithm.

The returned `Score` assigns a score for each Vector of values of the items.
"""
function joint(alg::Algorithm, run::Runtime, items::Vector{Queryable})::Union{Score, Tuple{Score, Score}}
    answer(Joint(), alg, run, items)
end

struct ProbValue{O} <: Query
    value :: O
end

struct ProbFunction <: Query
    fn :: Function
end

"""
    probability(alg::Algorithm, run::Runtime, items::Vector{Queryable}, predicate::Function)::Union{Float64, Tuple{Float64, Float64}}

Return the probability that `items` satisfy `query` or lower and upper probabilities.

`predicate` is a function from Vector{Any} to `Bool`.
"""
function probability(alg::Algorithm, run::Runtime, items::Vector{Queryable}, predicate::Function)::Union{Float64, Tuple{Float64, Float64}}
    answer(ProbFunction(predicate), alg, run, items)
end

function probability(alg::Algorithm, run::Runtime, item::Queryable, predicate::Function)::Union{Float64, Tuple{Float64, Float64}}
    f(vec) = predicate(vec[1])
    insts = VariableInstance[_resolve(run,item)]
    answer(ProbFunction(f), alg, run, insts)
end

"""
    probability(alg::Algorithm, run::Runtime, item::Queryable{O}, value::O)::Union{Float64, Tuple{Float64, Float64}} where O    

Return the probability that `item` has `value` or lower and upper probabilities.

The default implementation tries to use the more general probability of a query.
If that fails, it uses the `cpdf` operation on the marginal of `item`.
"""
function probability(alg::Algorithm, run::Runtime, item::Queryable{O}, value::O)::Union{Float64, Tuple{Float64, Float64}} where O
    inst::Instance = _resolve(run, item)
    try
        answer(ProbValue(value), alg, run, inst)
    catch e
        if e == ErrorException("_answer_")
            try
                probability(alg, run, item, x -> x == value)
            catch e
                if e == ErrorException("_answer_")
                    try
                        m = marginal(alg, run, inst)
                        return cpdf(m, (), value)
                    catch e
                        if e == ErrorException("_answer_")
                            error("None of the methods to compute the probability of a value are implemented")
                        else
                            rethrow(e)
                        end
                    end
                else
                    rethrow(e)
                end
            end
        else
            rethrow(e)
        end
    end
end

struct ProbabilityBounds{O} <: Query
    range :: Vector{O}
end

"""
    probability_bounds(alg::Algorithm, run::Runtime, item::Queryable, range::Vector)::Tuple{Vector{Float64}, Vector{Float64}}

    For an algorithm that produces lower and upper bounds, return vectors of lower and upper bounds on probabilities for values in the range.

    The range is important for computing the bounds, because it is assumed that values outside the range have probability zero.
"""
function probability_bounds(alg::Algorithm, run::Runtime, item::Queryable, range::Vector)::Tuple{Vector{Float64}, Vector{Float64}}
    return answer(ProbabilityBounds(range), alg, run, item)
end

struct Expectation <: Query 
    fn :: Function
end

struct Mean <: Query end

"""
    expectation(alg::Algorithm, runtime::Runtime, item::Queryable, f::Function)::Float64

Return the expectation of the function `f` over the marginal distribution of `item`.

The default implementation uses the expectation operation on the SFunc representing the
marginal distribution.
"""
function expectation(alg::Algorithm, run::Runtime, item::Queryable, fn::Function)::Float64
    try
        return answer(Expectation(fn), alg, run, item)
    catch e
        if e == ErrorException("_answer_")
            m = marginal(alg, run, item)
            if m isa Tuple 
                # marginal produced bounds; support is same for lower and upper bounds
                m = m[1]
            end
            tot = 0.0
            O = output_type(item)
            sup = support(m, (), 1000, O[])
            for x in unique(sup)
                tot += cpdf(m, (), x) * fn(x)
            end
            return tot
            # return Operators.f_expectation(marginal(alg, run, item), (), fn)
        else
            rethrow(e)
        end
    end
end

"""
    mean(alg::Algorithm, runtime::Runtime, item::Queryable)

Return the mean of `item`.
"""
function mean(alg::Algorithm, runtime::Runtime, item::Queryable{O})::Float64 where {O <: Number}
    try 
        answer(Mean(), alg, runtime, item)
    catch e
        if e == ErrorException("_answer_")
            return expectation(alg, runtime, item, x -> x)
        else
            rethrow(e)
        end
    end
        return 
end

struct Variance <: Query end

"""
    variance(alg::Algorithm, runtime::Runtime, item::Queryable)::Float64

Return the variance of `item`.
"""
function variance(alg::Algorithm, run::Runtime, item::Queryable)::Float64
    #answer(Variance(), alg, run, item)
    try
        return answer(Variance(), alg, run, item)
    catch e
        if e == ErrorException("_answer_")
            exp_x = expectation(alg, run, item, x->x)
            exp_xsq = expectation(alg, run, item, x->x^2)
            variance_x = exp_xsq - exp_x^2
            #return Operators.f_expectation(marginal(alg, run, item), (), fn)
            return variance_x
        else
            rethrow(e)
        end
    end
end

