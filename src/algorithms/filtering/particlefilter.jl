
export 
    ParticleFilter,
    SyncPF,
    AsyncPF,
    CoherentPF

function ParticleFilter(window_creator::WindowCreator, num_particles::Int, resampling_size::Int = num_particles)
    function pp!(run)
        particles = get_state(run, :particles)
        newparticles = resample(particles, resampling_size)
        set_state!(run, :particles, newparticles)
    end
    return WindowFilter(window_creator, LW(num_particles), pp!)
end

"""
    SyncPF(num_particles::Int, resampling_size::Int = num_particles)    

A particle filter that uses a synchronous window with the given number of particles and resampling buffer size.
"""
SyncPF(num_particles::Int, resampling_size::Int = num_particles) = ParticleFilter(SyncWindow(), num_particles, resampling_size)

"""
    AsyncPF(num_particles::Int, resampling_size::Int = num_particles, T = Float64)    

A particle filter that uses an asynchronous window with the given number of particles and resampling buffer size.
`T` represents the time type and must be the same as used in creation of the runtime.
"""
AsyncPF(num_particles::Int, resampling_size::Int = num_particles, T = Float64) = ParticleFilter(AsyncWindow{T}(), num_particles, resampling_size)

"""
    CoherentPF(num_particles::Int, resampling_size::Int = num_particles, T = Float64)    

A particle filter that uses a coherent window with the given number of particles and resampling buffer size.
`T` represents the time type and must be the same as used in creation of the runtime.
"""
CoherentPF(num_particles::Int, resampling_size::Int = num_particles, T = Float64) = ParticleFilter(CoherentWindow{T}(), num_particles, resampling_size)

#=
"""
General type of particle filters. Implementations must have fields runtime, num_particles, and resampling_size.
Some implementations will also provide APIs that organize the filtering in a specific way.
"""
abstract type ParticleFilter end

"""
current_particles(pf::ParticleFilter)

Returns a Particles data structure consisting of the current view of the most recent samples
of all variables in the network, along with the current log weights.
"""
# Samples are stored in a distributed manner across variables, possibly at different time points.
# Coherence of global samples is maintained by consistent indexing into arrays of variable samples.
function current_particles(pf::ParticleFilter)
    vars = get_variables(get_network(pf.runtime))
    # Need to make sure samples is initialized with distinct empty dictionaries
    samples = Sample[]
    for i in 1:pf.num_particles
        push!(samples, Sample())
    end
    for var in vars
        inst = current_instance(pf.runtime, var)
        varsample = get_value(pf.runtime, inst, :samples)
        for i in 1:pf.num_particles
            samples[i][var.name] = varsample[i]
        end
    end
    lws = get_state(pf.runtime, :log_weights)
    return Particles(samples, lws)
end

"""
store_particles!(pf::ParticleFilter, ps::Particles)

Stores the information in ps in a form pf can use.
"""
function store_particles!(pf::ParticleFilter, ps::Particles)
    vars = get_variables(get_network(pf.runtime))
    # Reset the samples and log weights in the runtime
    for var in vars
        inst = current_instance(pf.runtime, var)
        varsample = Vector{output_type(get_sfunc(inst))}()
        for i in 1:pf.num_particles
            # correction for 1-indexing
            push!(varsample, ps.samples[(i-1) % length(ps.samples) + 1][var.name])
        end
        set_value!(pf.runtime, inst, :samples, varsample)
    end
    set_state!(pf.runtime, :log_weights, ps.log_weights)
end

function resample!(pf::ParticleFilter)
    current_ps = current_particles(pf)
    new_ps = resample(current_ps, pf.resampling_size)
    store_particles!(pf, new_ps)
end

function init_filter(pf::ParticleFilter)
    ensure_all!(pf.runtime, 0)
    likelihood_weighting(pf.runtime, pf.num_particles)
    ps = get_state(pf.runtime, :particles)
    store_particles!(pf, ps)
end

# TODO: Handle evidence at different points in time
function filter_step(pf::ParticleFilter, variables_to_sample::Vector{<:Variable}, time::Int, evidence::Dict{Symbol, Score})
    @assert time > current_time(pf.runtime)
    set_time!(pf.runtime, time)

    # The log weights of the current evidence are added to the existing log weights for each sample.
    scores = get_state(pf.runtime, :log_weights)

    for var in variables_to_sample
        varsamples = Vector{output_type(var)}()
        inst = instantiate!(pf.runtime, var, time)
        sf = get_sfunc(inst)
        varev = var.name in keys(evidence) ? evidence[var.name] : nothing

        # Get the sample sets of each of the parents of the current variable in the transition model.
        # These parents may have been previously instantiated at any time.
        # instantiate! will make sure that the sfunc correctly takes into account the time lags,
        # as long as the model of the variable is defined appropriately.
        parents = get_transition_parents(get_network(pf.runtime), var)
        n = length(parents)
        parsamples = map(parents) do p
            if p == var # self edge from previous
                parinst = previous_instance(pf.runtime, var)
            else
                parinst = current_instance(pf.runtime, p)
            end
            get_value(pf.runtime, parinst, :samples)
        end

        for i in 1:pf.num_particles
            parvals = tuple([parsamples[j][i] for j in 1:n]...)
            if isa(varev, HardScore)
                # Since the value is known, force it and score it instead of sampling, LW style
                evval = varev.value
                push!(varsamples, evval)
                scores[i] += logcpdf(sf, parvals, evval)
            else
                # Otherwise, we sample it, but if there is evidence we still score it
                sampval = sample(sf, parvals)
                push!(varsamples, sampval)
                if !isnothing(varev)
                    scores[i] += get_log_score(varev, sampval)
                end
            end
        end

        set_value!(pf.runtime, inst, :samples, varsamples)
    end

    set_state!(pf.runtime, :log_weights, scores)
end



=#