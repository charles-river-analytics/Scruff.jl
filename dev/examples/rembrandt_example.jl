module RembrandtExample
using Scruff
using Scruff.SFuncs
using Scruff.Operators
using Scruff.Algorithms
using Scruff.RTUtils
using Scruff.Utils

# define parameters
isAuthentic_range = [:yes, :no]
isAuthentic_prior = [0.1, 0.9]

size_range = [:small, :medium, :large]
size_CPT = Dict((:yes, :people) => [0.3, 0.2, 0.5], (:no, :people) => [1/3, 1/3, 1/3], (:yes, :landscape) => [0.1, 0.2, 0.7], (:no, :landscape) => [1/3, 1/3, 1/3])

subject_range = [:people, :landscape]
subject_CPT = Dict((:yes, ) => [0.8, 0.2], (:no, ) => [0.5, 0.5])

brightness_range = [:bright, :dark]
brightness_CPT = Dict((:yes, :people) => [0.2, 0.8], (:no, :people) => [0.5, 0.5], (:yes, :landscape) => [0.3, 0.7], (:no, :landscape) => [0.8, 0.2])

function create_network()
    isAuthentic = Cat(isAuthentic_range, isAuthentic_prior)()(:isAuthentic)
    subject = DiscreteCPT(subject_range, subject_CPT)()(:subject)
    size = DiscreteCPT(size_range, size_CPT)()(:size)
    brightness = DiscreteCPT(brightness_range, brightness_CPT)()(:brightness)
    vars = Variable[isAuthentic, subject, size, brightness]
    network = InstantNetwork(vars, VariableGraph(subject=>[isAuthentic],size=>[isAuthentic, subject],brightness => [isAuthentic, subject]))
    return network
end

function run_inference(network, alg::InstantAlgorithm)
    runtime = Runtime(network) # create runtime
    # default_initializer(runtime) # create instances

    # Set soft evidence.
    evid = SoftScore([:bright, :dark], [0.2, 0.8]) 
    # Set hard evidence.
    #evid = HardScore(:dark)
    # post_evidence!(runtime, inst_brightness, evid)
    # println("Evidence $(evid) applied to $(:brightness)")

    # Perform inference
    infer(alg, runtime, Dict{Symbol, Score}(:brightness => evid))

    # Get current instances.
    inst_isAuthentic = current_instance(runtime, get_node(network, :isAuthentic))
    inst_subject = current_instance(runtime, get_node(network, :subject))
    inst_size = current_instance(runtime, get_node(network, :size))
    inst_brightness = current_instance(runtime, get_node(network, :brightness))
    
    
    # Get updated beliefs 
    println("Probability that it is authenthic: $(probability(alg, runtime, inst_isAuthentic, :yes))")
    println("Probability that it is small: $(probability(alg, runtime, inst_size, :small))")
    println("Probability that it is medium: $(probability(alg, runtime, inst_size, :medium))")
    println("Probability that it is large: $(probability(alg, runtime, inst_size, :large))")
    println("Probability that it is dark: $(probability(alg, runtime, inst_brightness, :dark))")
    println("Probability that it is bright: $(probability(alg, runtime, inst_brightness, :bright))")
    println("Probability that it has people: $(probability(alg, runtime, inst_subject, :people))")
    println("Probability that it has landscape: $(probability(alg, runtime, inst_subject, :landscape))")
end

# create network
network = create_network() 

# run inference
println("##################################")
println("Running Three Pass Belief Propagation")
println("##################################")
run_inference(network, ThreePassBP())
println("Running Loopy Belief Propagation")
println("##################################")
run_inference(network, LoopyBP())
println("##################################")
println("Running Variable Elimination")
println("##################################")
run_inference(network, VE(get_variables(network)))
println("##################################")
println("Running Importance Sampling")
println("##################################")
prop = convert(Dict{Symbol, SFunc}, Dict(:subject => DiscreteCPT(subject_range, subject_CPT)))
run_inference(network,  Importance(make_custom_proposal(prop), 1000))
end

