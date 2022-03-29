# The Scruff Tutorial

## What Scruff is all about

Scruff is a flexible framework for building AI systems. Although its roots are in probabilistic programming, it is not strictly speaking a probabilistic programming language. Instead, it is a framework for combining models of different kinds and reasoning with them. Scruff provides three main features:

1. The ability to combine different kinds of models and reason with them using an algorithm in an integrated way. Scruff decomposes the representation of models from algorithms that work with them using operators. Any representation (the scruff word is sfunc (stochastic function, pronounced "essfunk")) that implements the operators can appear in algorithms. Using this approach enables us to generalize algorithms like belief propagation and importance sampling that have traditionally been applied to probabilistic models. A given sfunc does not have to support all operators and algorithms can use sfuncs in the appropriate way. For example, it is legal to have an sfunc that you can't sample from, which would not be possible in a typical probabilistic programming language. 

2. A flexible framework for inference using these representations. Scruff distinguishes between the notion of a variable, which represents a value that can vary over time, and an instance of that variable, which represents its value at a particular time. In Scruff, variables are associated with models, which determine which sfunc to use for specific instances. There is no requirement that instances follow a regular time pattern; if the model supports it, instances can appear at any time interval. It is also possible to combine instances appearing at different time intervals, for example slowly changing and rapidly changing variables. Scruff also provides the ability to perform iterative inference, where beliefs about instances are refined through repeated computation.

3. Composition, reuse, and experimentation with different models, sfuncs, and algorithms. Scruff comes with an extensible and structured library of models, sfuncs, operators, and algorithms, making it easy to mix and match or extend with your own. For example, it is possible to implement alternative versions of an operators for an sfunc side by side and choose between them manually, or even automatically based on the characteristics of the specific instance. Another example is to compare accuracy and runtime between different time granularities on a variable by variable basis. Finally, as sfunc composition is highly structured, it is possible to experiment with specific sfunc choices in a systematic way.

The name Scruff derives from the old debates in AI between the neats and the scruffies. Neats believed that unless systems were developed in a coherent framework, it would be impossible to scale development of AI systems to complex real-world problems. Scruffies believed that real-world problems require a variety of techniques that must be combined as best as possible, and forcing everything into a neat framework would hinder progress.
We believe that both camps have an element of the truth, and Scruff is an attempt to provide the best of both worlds.
Scruff's philosophy is to allow a variety of representations and implementations to coexist side by side, and not every algorithm can be applied to every representation. However, they all coexist in a clean, well-defined and organized framework that enables scalable development of models and systems.

## Some opening examples

We start this tutorial with three examples illustrating idiomatic use of Scruff and some of its capabilities. These examples can be found in the `docs/examples` folder.

### Instant reasoning

Our first example, found in `novelty_example.jl` is about detecting and characterizing novel behaviors. In this example, a behavior is simply something that generates a real number, but the example extends to more interesting kinds of behavior. The example shows how to create sfuncs, models, variables, and networks, and how to reason with them. We call this an instant reasoning example because there is no temporal element.

We begin with some imports:

    using Scruff
    using Scruff.Models
    using Scruff.SFuncs
    using Scruff.Algorithms

Since we're going to run experiments with different setups, we define a NoveltySetup data structure.

    struct NoveltySetup
        known_sfs::Vector{Dist{Float64}}
        known_probs::Vector{Float64}
        novelty_prob::Float64
        novelty_prior_mean::Float64
        novelty_prior_sd::Float64
        novelty_sd::Float64
    end

Here, `known_sfs` is a vector of known behaviors, each one represented by a sfunc. In particular, each behavior is a `Dist{Float64}`, meaning it is an unconditional distribution over `Float64`. `known_probs` is the probabilities of the known behaviors, assuming the behavior is not novel, while `novelty_prob` is the probability that the behavior is novel. A novel behavior has a mean and standard deviation. The mean is drawn from a normal distrbution with mean `novelty_prior_mean` and standard deviation `novelty_prior_sd`. The novel behavior's own standard deviation is given by `novelty_sd`.

We now define a function that takes a setup and returns a network. Since observations are also part of the network, this function also takes the number of observations as an argument.

    function novelty_network(setup::NoveltySetup, numobs::Int)::InstantNetwork

This function begins by defining some variables. For the first variable, we'll go through the steps in detail. For the remaining variables, we'll use some syntactic sugar. The first variable represents the known behavior. Defining it takes three steps: creating the sfunc, defining the model, and associating it with a variable. Much of Scruff's power comes from separating these three concepts. However, for the common case where we want to do all three of these things together, we provide syntactic sugar.

First we create the sfunc:

    known_sf = Cat(setup.known_sfs, setup.known_probs)

This defines `known_sf` to be a categorical distribution, where the choices are provided by `setup.known_sfs` and the probabilities are specified by `setup.known_probs`. The important idea is that this distribution is an entity of its own, irrespective of specific variables that are related using it. An sfunc is similar to the mathematical concept of a function. It describes a relationship between variables that is not necessarily determinisitic. In mathematics, we can define concepts like function composition, which operate on the functions directly and don't require the notion of variables. Similarly in Scruff, there are a variety of ways to compose and combine sfuncs. Furthermore, we can have sfuncs be values in models as well, which enables higher-order probabilistic programming. In fact, in this example, `known_sf` represents a categorical distribution over sfuncs.

After creating the sfunc, we create a model using the sfunc:

    known_model = SimpleModel(known_sf)

A model describes which sfunc to generate for a variable in different situations. In general, the sfunc representing a variable's distribution can change depending on the situation, such as the time of instantiation of the variable and times of related instances. Here, we just have a `SimpleModel` that always returns the same sfunc, but later we will have more interesting models.

The third step is to create a variable and associate it with the model. This is achieved by calling the model with the variable name (a symbol) as argument:

    known = known_model(:known)

We now have the Julia variable `known` whose value is a Scruff variable with the name `:known` associated with `known_model`. If you just want to create a variable with a `SimpleModel` for a specific sfunc, you can use syntactic sugar as follows:

    known = Cat(setup.known_sfs, setup.known_probs)()(:known)

When we call the sfunc with zero arguments, we create a `SimpleModel` with the sfunc; then, when we apply that model to the variable name, we create the variable. In the rest of this example, we will use this form. Let's create some more variables:

    is_novel = Flip(setup.novelty_prob)()(:is_novel)
    novelty_mean = Normal(setup.novelty_prior_mean, setup.novelty_prior_sd)()(:novelty_mean)
    novelty = Det(Tuple{Float64}, Dist{Float64}, m -> Normal(m[1], setup.novelty_sd))()(:novelty)
    behavior = If{Dist{Float64}}()()(:behavior)

`is_novel` represents whether the behavior is novel or known. This variable will be true with probability `setup.novelty_prob`.
`novelty_mean` represents the mean of the novel behavior using a normal distribuiton whose mean and standard deviation are given by the setup.
`novelty` uses an sfunc called `Det`, which stands for "deterministic". It describes a determinstic relationship between one or more arguments and a result.
When you define a `Det`, you have to give the Julia compiler hints about the input and output types of the function. The input type of an sfunc in Scruff is always a tuple of arguments, so in this case it is a tuple of a single `Float64` argument. Our intent is for this input to represent the mean of the novel behavior; however, as we have discussed, sfuncs exist independently of the variables to which they will be applied. The connection to the novelty mean will be made later.
The output of the `Det` is an unconditional distribution of type `Dist{Float64}`. This is another example of an sfunc outputting an sfunc representing a behavior.
We now have two such sfuncs: `known` and `novelty`. We are ready to choose the actual behavior, using the `sf_choice` variable.
The sfunc for `sf_choice` is defined by `If{Dist{Float64}}()`. Unlike most probabilistic programming languages, which almost always provide an `if` control flow concept that choose between specific alternatives based on a test, Scruff's `If` describes the general process of choosing between two alternatives using a Boolean test. In this example, the intent is to choose between `novelty` and `known` based on `is_novel`. These connections will be made later.
Note that the type of value produced by the `If` is a type parameter, which in this case is again a `Dist{Float64}`, representing the actual behavior that gets chosen.

Now that we have these variables, we are ready to start building the connections described in the previous paragraph. We will build the ingredients to an `InstantNetwork`, which are a list of variables, and a `VariableGraph`, representing a dictionary from variables to their parents.

    variables = [known, is_novel, novelty_mean, novelty, behavior]
    graph = VariableGraph(novelty => [novelty_mean], behavior => [is_novel, novelty, known])

Finally, we need to add observations, which is done in a flexible way depending on the number of observations.

    for i in 1:numobs
        obs = Generate{Float64}()()(obsname(i))
        push!(variables, obs)
        graph[obs] = [behavior]
    end

For each observation, we create a variable whose name is given by the utility function
`obsname(i)`. The sfunc is `Generate{Float64}`. `Generate{Float64}` is a second-order sfunc 
that takes as input a `Dist{Float64}` and generates a `Float64` from it. 
Thus, each observation is an independent sample from the behavior. 
We add the observation to the `variables` vector and make its parents the `behavior` variable.

Finally, we create the instant network and return it.

        return InstantNetwork(variables, graph)

    end

Now that we've built the network, we're ready to run some experiments. 
Here's the code to run an experiment. It takes as arguments the setup, the vector of
observations, and the `InstantAlgorithm` to use (an `InstantAlgorithm` is an algorithm
run on an `InstantNetwork`; it does not handle dynamics).

    function do_experiment(setup::NoveltySetup, obs::Vector{Float64}, alg::InstantAlgorithm)
        net = novelty_network(setup, length(obs))
        evidence = Dict{Symbol, Score}()
        for (i,x) in enumerate(obs)
            evidence[obsname(i)] = HardScore(x)
        end
        runtime = Runtime(net)
        infer(alg, runtime, evidence)

        is_novel = get_node(net, :is_novel)
        novelty_mean = get_node(net, :novelty_mean)
        println("Probability of novel = ", probability(alg, runtime, is_novel, true))
        println("Posterior mean of novel behavior = ", mean(alg, runtime, novelty_mean))
    end

`do_experiment` first creates the network and then builds up the `evidence` data structure,
which is a dictionary from variable names to scores. In Scruff, a `Score` is an sfunc 
with no outputs that specifies a number for each value of its input. A `HardScore` is a
score that assigns the value 1 to its argument and 0 to everything else.
The next step is to create a runtime using the network.
The runtime holds all the information needed by the inference algorithm to perform
its computations and answer queries.
We then call `infer`, which does the actual work.
Once `infer` completes, we can answer some queries.
To answer a query, we need handles to the variables we want to use,
which is done using the `get_node` method.
Finally, the `probability` and `mean` methods give us the answers we want.

The examples next defines some setups and an observation list.

    function setup(generation_sd::Float64, prob_novel::Float64)::NoveltySetup
        known = [Normal(0.0, generation_sd), Normal(generation_sd, generation_sd)]
        return NoveltySetup(known, [0.75, 0.25], prob_novel, 0.0, 10.0, generation_sd)
    end
    setup1 = setup(1.0, 0.1)
    setup2 = setup(4.0, 0.1)
    obs = [5.0, 6.0, 7.0, 8.0, 9.0]

In `setup1`, behaviors have a smaller standard deviation, while in `setup2`, 
the standard deviation is larger.
We would expect the posterior probability of `is_novel` to be higher for `setup1`
than `setup2` because it is harder to explain the observations with known behaviors
when they have a small standard deviation.

Finally, we run some experiments. 

    println("Importance sampling")
    println("Narrow generation standard deviation")
    do_experiment(setup1, obs, LW(1000))
    println("Broad generation evidence")
    do_experiment(setup2, obs, LW(1000))

    println("\nBelief propagation")
    println("Narrow generation standard deviation")
    do_experiment(setup1, obs, ThreePassBP())
    println("Broad generation evidence")
    do_experiment(setup2, obs, ThreePassBP())

    println("\nBelief propagation with larger ranges")
    println("Narrow generation standard deviation")
    do_experiment(setup1, obs, ThreePassBP(25))
    println("Broad generation evidence")
    do_experiment(setup2, obs, ThreePassBP(25))

`LW(1000)` creates a likelihood weighting algorithm
that uses 1000 particles, while `ThreePassBP()` creates a non-loopy belief propagation 
algorithm. In this example, the network has no loops so using a non-loopy BP algorithm is
good. However, BP needs to discretize the continuous variables, which most of the
variables in this example are. With no arguments, it uses the default number of bins
(currently 10). `ThreePassBP(25)` creates a BP algorithm that uses 25 bins. 

The first time you run this example, it might take a
while. Julia uses just in time (JIT) compilation, so the first run can involve a lot of
compilation overhead. But subsequent runs are very fast. When you run this example, it produces output like this:

    julia> include("docs/examples/novelty_example.jl")
    Importance sampling
    Narrow generation standard deviation
    Probability of novel = 1.0
    Posterior mean of novel behavior = 7.334211013744095
    Broad generation evidence
    Probability of novel = 0.1988404327033635
    Posterior mean of novel behavior = 0.631562661691411

    Belief propagation
    Narrow generation standard deviation
    Probability of novel = 1.0
    Posterior mean of novel behavior = 7.71606183538526
    Broad generation evidence
    Probability of novel = 0.2534439250343668
    Posterior mean of novel behavior = 1.7131189737655137

    Belief propagation with larger ranges
    Narrow generation standard deviation
    Probability of novel = 1.0
    Posterior mean of novel behavior = 6.979068103646596
    Broad generation evidence
    Probability of novel = 0.2591460898562207
    Posterior mean of novel behavior = 1.7363865329521413

We see that as expected, the probability of novel is much higher with narrow generation
standard deviation than with broad. All three algorithms have similar qualitative results.
Running the experiment a few times shows that the importance sampling method has relatively
high variance. We also see that the estimate of the posterior mean changes significantly
as we add more values to the ranges of variables for the BP method.

### Incremental reasoning

Building on the last point, our next example, found in `novelty_lazy.jl`, uses Scruff's 
incremental inference capabilities to gradually increase the range sizes of the 
variables to improve the estimates. We're going to use an algorithm called Lazy Structured
Factored Inference (LSFI). LSFI repeatedly calls an `InstantAlgorithm` (in this case
variable elimination) on more and more refined versions of the network.
Refinement generally takes two forms: Expanding recursive networks to a greater depth, 
and enlarging the ranges of continuous variables.
Our example only has the latter refinement.
When expanding recursive networks, LSFI can produce lower and upper bounds to query
answers at each iteration. This capability is less useful for range refinement,
but our code needs to handle the bounds.

The network and setups are just as in `novelty_example.jl`. The code for running an
experiment is similar in structure but has some new features.

    function do_experiment(setup, obs)
        net = novelty_network(setup, length(obs))
        is_novel = get_node(net, :is_novel)
        novelty_mean = get_node(net, :novelty_mean)
        evidence = Dict{Symbol, Score}()
        for (i,x) in enumerate(obs)
            evidence[obsname(i)] = HardScore(x)
        end

        alg = LSFI([is_novel, novelty_mean]; start_size = 5, increment = 5)
        runtime = Runtime(net)
        prepare(alg, runtime, evidence)

        for i = 1:10
            println("Range size: ", alg.state.next_size)
            refine(alg, runtime)
            is_novel_lb = probability_bounds(alg, runtime, is_novel, [false, true])[1]
            println("Probability of novel = ", is_novel_lb[2])
            println("Posterior mean of novel behavior = ", mean(alg, runtime, novelty_mean))
        end
    end

As before, the code creates the network, gets handles of some variables, and fills
the evidence data structure.
In this case, we use `LSFI`. When creating an `LSFI` algorithm, we need to tell it which
variables we want to query, which are `is_novel` and `novelty_mean`. `LSFI` also has
some optional arguments. In this example, we configure it to have a starting range size
of 5 and increment the range size by 5 on each refinement.
Before running inference, we need to call `prepare(alg, runtime, evidence)`. Then we go
through ten steps of refinement. We can get the range size of the next refinement
using `alg.state.next_size` (we only use this for printing). 
Refinement is done through a call to `refine(alg, runtime)`.
We then need to do a little more work than before to get the answers to queries because
of the probabilities bounds.
`probability_bounds(alg, runtime, is_novel, [false, true])` returns lower and upper bounds 
as 2-element vectors of probabilities of `false` and `true`. As discussed earlier,
these bounds are not true bounds in the case of range refinement, so we just pick
the first one, and then pick the second value, corresponding to `true`, out of that vector.
The `mean` method already arbitrarily uses the lower bounds so we don't have to do any work there.

Running this example produces a result like:

    Lazy Inference
    Narrow generation standard deviation
    Range size: 5
    Probability of novel = 1.0
    Posterior mean of novel behavior = 5.0
    Range size: 10
    Probability of novel = 1.0
    Posterior mean of novel behavior = 5.000012747722526
    Range size: 15
    Probability of novel = 1.0
    Posterior mean of novel behavior = 5.000012747722526
    Range size: 20
    Probability of novel = 1.0
    Posterior mean of novel behavior = 5.000012747722526
    Range size: 25
    Probability of novel = 1.0
    Posterior mean of novel behavior = 5.000012747722526
    Range size: 30
    Probability of novel = 1.0
    Posterior mean of novel behavior = 5.000012747722526
    Range size: 35
    Probability of novel = 1.0
    Posterior mean of novel behavior = 5.000012747722526
    Range size: 40
    Probability of novel = 1.0
    Posterior mean of novel behavior = 5.000012747722526
    Range size: 45
    Probability of novel = 1.0
    Posterior mean of novel behavior = 5.000012747722526
    Range size: 50
    Probability of novel = 1.0
    Posterior mean of novel behavior = 5.000012747722526

    Broad generation evidence
    Range size: 5
    Probability of novel = 0.23525574698998955
    Posterior mean of novel behavior = 1.1750941076530532
    Range size: 10
    Probability of novel = 0.19825748797545847
    Posterior mean of novel behavior = -0.11214142944113853
    Range size: 15
    Probability of novel = 0.19745646974840933
    Posterior mean of novel behavior = -0.11168834527313051
    Range size: 20
    Probability of novel = 0.19283490006948978
    Posterior mean of novel behavior = 0.3602757973718763
    Range size: 25
    Probability of novel = 0.1926826680899825
    Posterior mean of novel behavior = 0.35995765581210176
    Range size: 30
    Probability of novel = 0.1825284089501074
    Posterior mean of novel behavior = 1.1318032244818
    Range size: 35
    Probability of novel = 0.18251757269528399
    Posterior mean of novel behavior = 1.1294239567980586
    Range size: 40
    Probability of novel = 0.18251757269528404
    Posterior mean of novel behavior = 1.1294239567980597
    Range size: 45
    Probability of novel = 0.18251757269528404
    Posterior mean of novel behavior = 1.1294239567980597
    Range size: 50
    Probability of novel = 0.18251757269528404
    Posterior mean of novel behavior = 1.1294239567980597

Looking at this output, we see that the narrow generation standard deviation case is easy and the algorithm 
quickly converges. However, in the broad generation standard deviation case, we see that there is a big
change in the posterior mean of novel behavior between range size 25 and 30. This is to
do with the way values in the range are generated. As the range size is increased,
values further and further away from the prior mean are created.
At range size 30, a value is introduced that has low prior but fits the data well,
which increases the posterior mean.

### Dynamic reasoning

Our final example riffs on the novelty theme to use dynamic reasoning.
Now, observations are received over time at irregular intervals.
A behavior now represents the velocity of an object moving in one dimension,
starting at point 0.0.
This example moves away from the higher-order sfuncs but introduces some new kinds of
models.

The setup is similar but slightly different:

    struct NoveltySetup
        known_velocities::Vector{Float64}
        known_probs::Vector{Float64}
        novelty_prob::Float64
        novelty_prior_mean::Float64
        novelty_prior_sd::Float64
        transition_sd::Float64
        observation_sd::Float64
    end

We have known velocities and their probabilities, the probability of novelty, and the mean and standard deviation of the novel velocity.
We also have the standard deviation of the transition and observation models.

Because the observations appear irregularly and not at fixed time steps, we are going to
use a `VariableTimeModel` to represent the position of the object. 
To create a `VariableTimeModel`, we need to create a new type that inherits from 
`VariableTimeModel` and implement the methods `make_initial`, which creates the sfunc
for the initial time step, and `make_transition`, which creates the sfunc at each time
step at which we instantiate the variable. 

    struct PositionModel <: VariableTimeModel{Tuple{}, Tuple{Float64, Float64}, Float64} 
        setup::NoveltySetup
    end
    function make_initial(::PositionModel, ::Float64)::Dist{Float64}
        return Constant(0.0)
    end
    function make_transition(posmod::PositionModel, parenttimes::Tuple{Float64, Float64}, time::Float64)::SFunc{Tuple{Float64, Float64}, Float64}
        function f(pair)  
            (prevval, velocity) = pair
            Normal(prevval + t * velocity, t * posmod.setup.transition_sd)
        end
        t = time - parenttimes[1]
        return Chain(Tuple{Float64, Float64}, Float64, f)
end

`make_initial` simply returns `Constant(0.0)`, meaning that the object always starts at
position 0.0 with no uncertainty.
Because the amount of time between instantiations
is variable, `make_transition` takes as argument a vector of times of the previous
instantiation of its parents, as well as the current time. 
It uses these times to determine exactly what the transition model should be.
Here, it computes the time `t` between the current time and the previous instantiation
of the first parent, which we will later connect to the position variable.
So `t` represents the time since the last instantiation of the position variable.
`make_transition` uses the `Chain` sfunc, which takes parent values and applies 
a Julia function to produce the sfunc used to generate the value of the `Chain`.
In this case, once we make the connections, the `Chain` will take the previous value of 
the position and the velocity and create a `Normal` sfunc whose mean and standard deviation
depend on `t`, as well as the standard deviation of the transition model in the setup.
This Normal is then used to generate the current position.
This code is a little sophisticated, but the ability to create variable time models and perform 
asynchronous dynamic reasoning is a powerful feature of Scruff.

The rest of the example is simpler and we won't go over it in full detail.
We do introduce the `StaticModel`, which represents a variable whose value is generated
at the beginning of a run and never changes. 
`StaticModel` is implemented as a `VariableTimeModel` where the transition function is 
the identify function.
Also, the `observation` variable uses a `SimpleModel`, because it is generated afresh 
instantaneously every time it is instantiated. It is defined to be a normal whose mean
is the position and whose standard deviation is given by the setup. This is implemented
using the `LinearGaussian` sfunc.

A `DynamicNetwork` uses two variable graphs for the initial and transition steps. In this
example, all the logic of choosing the behavior happens in the initial graph, while the
position logic and its dependence on previous position and velocity is in the transition
graph. The transition graph also contains copy edges for the static variables.

    variables = [known_velocity, is_novel, novel_velocity, velocity, position, observation]
    initial_graph = VariableGraph(velocity => [is_novel, novel_velocity, known_velocity], observation => [position])
    transition_graph = VariableGraph(known_velocity => [known_velocity], is_novel => [is_novel], novel_velocity => [novel_velocity], 
                                     velocity => [velocity], position => [position, velocity], observation => [position])


We'll show the `do_experiment` implementation in detail because it illustrates how 
asynchronous inference is performed.

    function do_experiment(setup::NoveltySetup, obs::Vector{Tuple{Float64, Float64}}, alg::Filter)
        net = novelty_network(setup, length(obs))
        runtime = Runtime(net, 0.0) # Set the time type to Float64 and initial time to 0
        init_filter(alg, runtime)

        is_novel = get_node(net, :is_novel)
        velocity = get_node(net, :velocity)
        observation = get_node(net, :observation)

        for (time, x) in obs
            evidence = Dict{Symbol, Score}(:observation => HardScore(x))
            println("Observing ", x, " at time ", time)
            # At a minimum, we need to include query and evidence variables in the filter step
            filter_step(alg, runtime, Variable[is_novel, velocity, observation], time, evidence)

            println("Probability of novel = ", probability(alg, runtime, is_novel, true))
            println("Posterior mean of velocity = ", mean(alg, runtime, velocity))
        end
    end

After creating the network, we create a runtime. The call to `Runtime` takes a second
argument that not only sets the initial time but also established the type used to
represent time, which is `Float64`. We first need to initialize the filter with `init_filter` which runs the
initial time step, and get handles 
to the variables we care about. Our observation sequence is a vector (sorted by increasing
time) of (time, value) pairs.
For each such pair, we create the evidence at that time point.
Then we run a `filter_step`.
Besides the algorithm and runtime, the filter step takes a vector of variables to instantiate,
the current time, and the evidence.
There is no need to instantiate all the variables at every filter step.
At a minimum, we need to instantiate evidence variables as well as any variables
we want to query.
Since we're going to query `is_novel` and `velocity`, we'll have to instantiate those
using their copy transition model.
However, we never need to instantiate the `known_velocity` and `novel_velocity` variables
after the initial time step.
Finally, we can answer queries about the current state in a similar way to the other
examples.

For the experiments, we create a setup and two sequences of observations, the second of
which is harder to explain with known behaviors.

    # Known velocities are 0 and 1, novelty has mean 0 and standard deviation 10
    setup = NoveltySetup([0.0, 1.0], [0.7, 0.3], 0.1, 0.0, 10.0, 1.0, 1.0)
    obs1 = [(1.0, 2.1), (3.0, 5.8), (3.5, 7.5)] # consistent with velocity 2
    obs2 = [(1.0, 4.9), (3.0, 17.8), (3.5, 20.5)] # consistent with velocity 6

We then use `CoherentPF(1000)` as the filtering algorithm. Current filtering algorithms in 
Scruff combine an instantiation method that creates a window with an underlying 
`InstantAlgorithm` to infer with the window. Available window creation methods include
synchronous, asynchronous, and coherent. Coherent is similar to asynchronous except that
it adds variables to the instantiation to maintain coherence of parent-child relationships.
In this example, it ensures that the position variable is also instantiated, not just the query and
evidence variables. `CoherentPF(1000)` describes a particle filter that uses a coherent window
creator and an importance sampling algorithm with 1000 particles. The example also shows
how you can similarly create a coherent BP algorithm. However, BP does not work well in
models with static variables because dependencies between the static variables are lost
between filtering steps.

Running this example produces output like the following for the particle filter:

    Particle filter
    Smaller velocity
    Observing 2.1 at time 1.0
    Probability of novel = 0.0351642575352557
    Posterior mean of velocity = 0.5411884423148781
    Observing 5.8 at time 3.0
    Probability of novel = 0.057222570825582145
    Posterior mean of velocity = 0.8705507592898075
    Observing 7.5 at time 3.5
    Probability of novel = 0.08166159149240186
    Posterior mean of velocity = 1.007810909419299

    Larger velocity
    Observing 4.9 at time 1.0
    Probability of novel = 0.6741688102988623
    Posterior mean of velocity = 3.6150131656907174
    Observing 17.8 at time 3.0
    Probability of novel = 1.0
    Posterior mean of velocity = 5.898986723263269
    Observing 20.5 at time 3.5
    Probability of novel = 1.0
    Posterior mean of velocity = 5.86994402484129

## Scruff concepts

The central concepts of Scruff are:

- Sfuncs, or stochastic functions, which represent mathematical relationships between variables
- Operators, which define and implement computations on sfuncs
- Models, which specify how to create sfuncs in different situations
- Variables, which represent domain entities that may take on different values at different times
- Networks, which consist of variables and the dependencies between them
- Instances, which represent a specific instantiation of a variable at a point in time
- Algorithms, which use operations to perform computations on networks
- Runtimes, which manage instances as well as information used by algorithms

## Sfuncs

An `SFunc` has an input type, which is a tuple, and an output type.
Although the name implies probabilistic relationships, in principle sfuncs
can be used to represent any kind of information.
The representation of an sfunc is often quite minimal, with most of the detail contained
in operators. The general type is `SFunc{I <: Tuple, O}`.

### Dists

A `Dist{O}` is an `SFunc{Tuple{}, O}`.
In other words, a `Dist` represents an unconditional distribution with no parents.
Examples of `Dist` include `Constant`, `Cat`, `Flip`, and `Normal`.

### Scores

A `Score{I}` is an `SFunc{Tuple{I}, Nothing}`. In other words, it takes a single value of 
type `I`, and rather than produce an output, it just associates information (typically a likelihood)
with its input. A `Score` is often used to represent evidence.
Examples of `Score` include `HardScore` (only a single value allowed), 
`SoftScore` (allows multiple values), `LogScore` (similar to `SoftScore` but represented
in log form), `FunctionalScore` (score is computed by applying a function to the input),
`NormalScore` (representing a normal distribution around a value), 
and `Parzen` (mixture of normal scores).

### Conditional Sfuncs

Scruff provides a range of ways to construct sfuncs representing conditional distributions.
These are organized in a type hierarchy:

— `Invertible`: deterministic functions with a deterministic inverse, enabling efficient operator implementations\
— `Det`: deterministic functions without an inverse\
   └ `Switch`: chooses between multiple incoming choices based on first argument\
      └ `LinearSwitch`: first argument is an integer and switch chooses corresponding result\
      └ `If`: first argument is a Boolean and switch chooses appropriate other argument \
— `Conditional`: abstract representation of sfuncs that use first arguments to create sfunc to apply to other arguments\
   └ `LinearGaussian`: sfunc representing normal distribution whose mean is a linear function of the parents\
   └ `Table`: abstract representation of sfuncs that use first arguments to choose sfunc to apply from a table\
      └ `DiscreteCPT`: discrete conditional probability table\
      └ `CLG`: conditional linear Gaussian model: table of linear Gaussians depending on discrete parents\
— `Separable`: Mixture of `DiscreteCPT` to decompose dependency on many parents, enabling efficient operator implementations\

### Compound Sfuncs

Compound sfuncs can be though of as a construction kit to compose more complex sfuncs out of
simpler ones. These also include some higher-order sfuncs.

- `Generate`: generate a value from its sfunc argument
- `Apply`: similar to generate, but the sfunc argument is applied to another argument
- `Chain`: apply a function to the arguments to produce an sfunc, then generate a value from the sfunc
- `Mixture`: choose which sfunc to use to generate values according to a probability distribution
- `Serial`: connect any number of sfuncs in series
- `NetworkSFunc`: connect any number of sfuncs according to a graph
- `Expander`: apply a function to the arguments to produce a network that can be used recursively

## Operators

An operator represents a computation that can be performed on an sfunc.
An operator is not just a function or a method. 
It is an object that can contain information (such as configuration instructions)
and can be reasoned about, for example to specify policies to choose between alternative
implementations.
Operators consist of definitions, created using `@op_def`, which specify type information,
and implementation, created using `@impl`.

Here are some of the most commonly used operators:

- `cpdf(sf, parent_values, x)` returns the conditional probability of `x` given `parent_values`
- `logcpdf(sf, parent_values, x)`
- `sample(sf, parent_values)`
- `get_score(sf, x)` returns the score associated with `x`
- `get_log_score(sf, x)`
- `support(sf, parent_ranges, target_size, current)` computes a range of values for the sfunc given that the parents have values in `parent_ranges`. `target_size` is guidance as to the size of support to produce, which does not need to be obeyed precisely. `current` is a list of values that should be included in the support, which is useful for iterative refinement.

The above operators will be implemented specifically for a given sfunc. In general, an sfunc does not need to support all operators. For example, typically only a `Score` will support `get_score` and `get_log_score`. 
Some sfuncs will not be able to support sampling or density computation, and that's okay.
For example, if an sfunc doesn't support `sample`, but it does support `cpdf`, and that sfunc is always observed,
it can be used in likelihood weighting.
If it is not always observed, it won't be usable in importance sampling but it might be usable in BP.
The goal is to enable representations to be used as much as possible, rather than require everything to work uniformly.
This is where the scruffiness of Scruff comes in.

There are a variety of operators useful in BP and related algorithms.
Most of these have default implementations that work for sfuncs in general and you don't
need to worry about implementing them for a new sfunc. 
The two that need to be implemented specifically are:

- `compute_pi(sf, range, parent_ranges, parent_pi_messages)`, which integrates over the parents to produce a distribution over the value of the instance associated with the sfunc. The `parent_pi_messages`, as well as the computed distribution, are represented as `Dist`s, rather than vectors or anything specific, which enables great flexibility in implementation.
- `send_lambda(sf, lambda, range, parent_ranges, parent_pi_messages, parent_index)` computes the lambda message to be sent to the parent specified by `parent_index`.
  
Once these two operators are implemented for an sfunc, the sfunc can participate in any BP
algorithm. Furthermore, sfuncs at the leaves of a network do not need to implement `compute_pi`.
For example, `send_lambda` can be implemented for a feedforward neural network, enabling it
to be included in a general BP inference process.

## Models

One of Scruff's key features is the ability to reason flexibly about variables that vary over time, and, in future, space. This is accomplished using models, which specify how to make the sfunc to use for a particular instance of a variable. Currently, Scruff's `models` library is relatively small. We plan to expand it in future, for example with learning models that improve their sfuncs based on experience.

Here is the current type hierarchy of models

—`InstantModel`: for a variable with no dependencies on previous time points\
  └ `TimelessInstantModel`: an `InstantModel` where the sfunc also does not depend on the current time\
  └ `SimpleModel`: a `TimelessInstantModel` in which the sfunc to use is passed in the definition of the model\
— `FixedTimeModel`: a model for a variable that depends on its own state at the previous time point and other variables at the current time point. The delta between the current and previous time point must be a fixed `dt`.\
  └ `TimelessFixedTimeModel`: a `FixedTimeModel` in which the sfunc does not depend on the current time\
  └ `HomogenousModel`: a `TimelessFixedTimeModel` in which the initial and transition sfuncs are passed in the definition of the model\
— `VariableTimeModel`: a model for a variable whose transition sfunc depends on the time intervals since the instantiations of its parents (which may be at different times)\
  └ `StaticModel`: a model for a variable whose value is set in the initial time point and never changes afterward\

## Networks

Networks contains nodes, which are either variables or placeholders. Unlike variables, placeholders are not associated with models. Rather, they are intended to indicate values that should be received from outside the network. They are particularly useful for recursive reasoning, as well as dynamic inference.

An `InstantNetwork` is created with two to four arguments:

- A vector of variables
- A variable graph, associating variables with their parents. If a variable has no parents, it can be omitted from the graph.
- (Optional) A vector of placeholders, which defaults to empty
- (Optional) A vector of outputs, which should be a subset of the variables, again defaults to empty. This is intended to support providing an interface to networks that enables internal nodes and embedded networks to be eliminated, but this feature is not used yet.

A `DynamicNetwork` is created with three to six arguments

- A vector of variables
- An initial variable graph
- A transition variable graph
- (Optional) A vector of initial placeholders, defaults to empty
- (Optional) A vector of transition placeholders, defaults to empty
- (Optional) A vector of outputs, defaults to empty

## Algorithms

Scruff's algorithms library is structured so that more complex algorithms can be built out of simpler algorithms. The basic algorithms are instances of `InstantAlgorithm` and run on an `InstantNetwork`.
Scruff currently provides the following hierarchy of instant algorithms. We intend to expand this list over time:

— `InstantAlgorithm`: abstract type for which implementations must implement the `infer` method\
  └ `Importance`: general importance sampling framework\
    └ `Rejection`: rejection sampling\
    └ `LW`: likelihood weighting\
    └ Custom proposal: An importance sampling algorithm can be made from a proposal scheme using `make_custom_proposal`. A proposal scheme specifies alternative sfuncs to use as alternatives to the prior distribution for specific sfuncs\
  └ `BP`: general belief propagation framework\
    └ `ThreePassBP`: non-loopy belief propagation\
    └ `LoopyBP`\
  └ `VE`: variable elimination

Scruff provides iterative algorithms that gradually improve their answers over time. These follow the following hierarchy:

— `IterativeAlgorithm`: abstract type for which implementations must implement the `prepare` and `refine` methods\
  └ `IterativeSampler`: iterative algorithm that uses a sampler to increase the number of samples each refinement. For example, you can use `IterativeSampler(LW(1000))` to use a likelihood weighting algorithm that adds 1,000 more samples on each call to `refine`.\
  └ `LazyInference`: an algorithm that expands the recursion depth and ranges of variables on each call to `refine` and then invokes an `InstantAlgorithm`\
    └ `LSFI`: a `LazyInference` algorithm that uses variable elimination as its instant algorithm

For filtering, Scruff provides the general `Filter` class, for which implementations must implement the `init_filter` and `filter_step` methods. All current filter implementations in Scruff derive from `WindowFilter`, where, on each call to `filter_step`, the algorithm first creates an `InstantNetwork` representing a window and then invokes an `InstantAlgorithm`. To create a `WindowFilter`, you choose a windowing method from `SyncWindow`, `AsyncWindow`, and `CoherentWindow`, and specify the instant algorithm to use. For example, Scruff provides the following constructor for asynchronous BP:

    AsyncBP(range_size = 10, T = Float64) = WindowFilter(AsyncWindow{T}(), ThreePassBP(range_size))

Once algorithms have been run, queries can be answered using a uniform interface. This includes methods like
`marginal`, `joint`, `probability` (which could be the probability of a specific value or the probability of a predicate), `expectation`, `mean`, and `variance`. As usual, not all algorithms need implement all queries.

When you implement a new algorithm, you can specify how to answer queries using a standard `answer` method. Take a look at algorithm implementations to see how this works.

## The runtime

Unless you are implementing a new algorithm, you can largely ignore details of the runtime after you have created it, as everything happens under the hood.
In general, the responsibilities of the runtime are to:

- Instantiate variables and associate them with the correct instance parents and sfunc
- Identify the appropriate instances of variables at different point in time
- Store and retrieve values associated with instances
- Store and retrieve algorithm state across multiple invocations (e.g., using `refine`)
- Manage passing of messages between variables

## Future work

Future work in Scruff will follow five main lines: 
developing more extensive libraries, including integration of other frameworks;
developing a larger suite of algorithms using compositional methods;
developing a more flexible framework of networks and recursive models; 
creating spatial and spatiotemporal models with the same flexibility as current temporal models;
and operators for performance characterization and optimization.
We welcome contributions from the user community.
If any of these items catches your interest, let us know and we will be happy to help
with design and development.

### Larger libraries and integration of other frameworks

Scruff's current library, particularly of SFuncs, is fairly minimal, and needs to be extended to provide a fully functional probabilistic programming framework.
Our intent is not to write sfuncs ourselves, but rather to wrap existing implementations wherever possible.
An immediate goal is to wrap `Distributions.jl`, while will provide a wide range of `Dist` sfuncs.
We also want to integrate with other probabilistic programming frameworks in Julia, such as Gen.

In addition, the ability to use data-driven models that don't support sampling but do support inference is central to Scruff.
We want to develop a library of such models, again by integrating with existing frameworks and wrapping with appropriate observations.
Algorithms also need to be modified to take advantage of such models.

### More algorithms

It is important that algorithms in Scruff are well-structured and compositional.
The algorithms developed so far are a starter set that have been carefully designed with this philosophy.
Noticable by its absence is MCMC, which is common in many probabilistic programming frameworks.
Gibbs sampling can be implemented as a message passing algorithm and fits well with the current framework.
Metropolis-Hastings and reversible jump algorithms will take more thought, but experience with other 
probabilistic programming languages should show how to implement them in a consistent, compositional way.

A very natural next step is to generalize our algorithms to use other semirings besides aum-product. Again, this should happen in a compositional way.
It should be possible to say something like `with_semiring(semiring, algorithm)` and have all computations in operators invoked by the algorithm
drawn from the appropriate semiring. If we do this, it will be natural to write learning algorithms like EM and decision-making algorithms using maximum 
expected utility using our instant algorithms. This will lead to powerful combinations. Would anyone like asynchronous online EM using BP?

Similarly, BP is just one example of a variational method. We want to expand BP into a more general compositional
variational inference library.
Finally, we want to generalize our elimination methods to employ conditioning as well as elimination.

### More flexible networks and recursion

The ability for networks to contain other networks is critical to structured, modular, representations
as well as efficient inference through encapsulation and conditional compilation.
In addition, the ability to generate contained networks stochastically supports open universe modeling. 
Scruff currently supports these capabilities through Expanders.
However, Expanders were an early addition to Scruff and are not integrated all that well in the most recent Scruff development.
NetworkSFuncs are better integrated, but do not currently support containment and recursion.
We want to align Expanders and NetworkSFuncs to provide more general structured and recursive networks.

### Spatially flexible models

Scruff currently has a flexible representation of variables that vary over time, but not of variables that vary over space, or space and time together.
We want to provide spatiotemporal networks with the same flexibility as current DynamicNetworks.
Moving beyond spatial models, we also want to create a framework for reasoning about variables that vary across graphs, such as social networks.

### Performance Characterization and Optimization

Scruff's design is intended to enable reasoning about performance characteristics of operators and to support algorithms making decisions
about which operators to use. Multiple operator implementations can exist side by side for given sfuncs and algorithms can use policies
to decide which ones to use. This capability is currently only exercised in very rudimentary ways.
We want to take advantage of this capability to provide a wide set of performance characteristics and intelligent algorithms that use them.