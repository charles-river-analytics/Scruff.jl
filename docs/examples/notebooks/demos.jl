### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 2f130e30-dbdf-11ec-12c6-0d0cb74dcfbe
begin
	import Pkg
	Pkg.activate("../../..")
	using Revise
	using Scruff
	using Scruff.Operators
	using Scruff.SFuncs
	using Scruff.Models
	using Plots
	import Distributions
end

# ╔═╡ 6131b416-6e38-4cad-839b-a668df2c1477
begin
	function est_mu_from_samples(sf, num_samples)
	    return sum([sample(sf, tuple()) for i in 1:num_samples])/num_samples
	end
	function est_var_from_samples(sf, num_samples)
		samples = [sample(sf, tuple()) for i in 1:num_samples]
		mu_est = sum(samples)/num_samples
		return sum([(samp - mu_est)^2 for samp in samples])/(num_samples - 1)
	end
end

# ╔═╡ e597edc8-635b-4d6d-beb9-455eeb603eab
begin
	# Some basic operator examples
	norm = Normal(0., 1.)

    samples = [2^i for i in 1:24]
	
	#= CALL OPERATOR =#
	# Estimate basic statistics. Available in closed form for gaussians.
	(mu_true, var_true) = (expectation(norm, tuple()),
		                   variance(norm, tuple()))
end

# ╔═╡ 97e6059b-b7f0-4b17-9340-7aef21464349
begin
    plot(samples,
		 [abs(mu_true - est_mu_from_samples(norm, num_samples)) for num_samples in samples],
	     scale=:log10,
	     xlabel="Num Samples",
		 ylabel="Error",
 		 legend=false,
	     title="Mean Estimate Absolute Error")
end

# ╔═╡ 73c7c02b-99ad-495d-a507-7c2302fd5a73
begin
	plot(samples,
		 [abs(var_true - est_var_from_samples(norm, num_samples)) for num_samples in samples],
	     scale=:log10,
	     xlabel="Num Samples",
		 ylabel="Error",
		 legend=false,
	     title="Variance Estimate Absolute Error")
end

# ╔═╡ 56773c97-6b4c-4dd8-8f24-d528eac54b70
begin
	function push_plot_data!(data, expectation, variance)
        push!(data[1], expectation - sqrt(variance))
        push!(data[2], expectation)
        push!(data[3], expectation + sqrt(variance))
    end
    function plot_async_random_walk(data, observations)
		N = size(data[1])[1] - 1
		data_min = min(minimum(data[1]), minimum((expectation(obs[2], tuple()) - 0.5*(variance(obs[2], tuple())^0.5) for obs in observations)))
		data_max = max(maximum(data[1]), maximum((expectation(obs[2], tuple()) + 0.5*(variance(obs[2], tuple())^0.5) for obs in observations)))
        plt = plot(0:N, data[[1,3]], 
			       xlim=(0, N),
			       ylim=(data_min, data_max), 		 				   
			       title="Asynchronous Random Walk",
			       marker=0, 
			       legend=false)
		scatter!([obs[1] for obs in observations],
				 [expectation(obs[2], tuple()) for obs in observations]; 
				 yerror=[0.5*variance(obs[2], tuple())^0.5 for obs in observations])
    end
    function observation_occurred()
		p_observe = 0.01
        return rand(Distributions.Bernoulli(p_observe))
    end
end

# ╔═╡ 660f3c4a-740c-4388-80d5-9bd02412071b
begin
    g = WienerProcess(0.0001)

    NSteps = 5000
	start = 3.

    function pick_obs_noise()
		return rand(1:10)/20.
	end
	
	_last_observation = 0
    _plot_data = [[start], [start], [start]]
    beliefs = [Normal(start, 0.)]
    observations = []
    for i in 1:NSteps
		global _last_observation
        # Update belief using some random previous time step (after last obs).
        # In this case the result doesn't matter because 
        # the Wiener process is defined exactly for different time deltas.
        # This just demonstrates that reasoning continues 
		# to be correct when updating beliefs conditioned on arbitrary deltas.
        prev_step = rand((_last_observation + 1):i)
		
        belief_transition::SFunc = make_transition(g, prev_step - 1, i)

		#= CALL OPERATOR =#
		# Get SFunc given by p(y) = \int p(y|x)q(x)dx
		# Can be done in closed form for gaussians
		new_belief = marginalize(belief_transition, beliefs[prev_step])
        
		push!(beliefs, new_belief)

        if observation_occurred() && i - _last_observation > 3
			# Take a sample from our predicted distribution as an "observation"
			true_val = sample(beliefs[end], tuple())
			# Add some noise to the observation
			noise_amt = pick_obs_noise()
			obs_model = Normal(true_val, noise_amt)
			noisy_obs = sample(obs_model, tuple())
			# Model the noise when using it to update state estimate
			noisy_obs_model = Normal(noisy_obs, noise_amt)

			#= CALL OPERATOR =#
			# Create an SFunc describing p(x) \propto u(x)v(x)
			# This can be done in closed form for gaussians
			# The Kalman filter is based upon this
            beliefs[end] = productnorm(noisy_obs_model, beliefs[end])
				
            _last_observation = i
			push!(observations, (_last_observation, noisy_obs_model))
        end

		#= CALL OPERATOR =#
		# Estimate stats. Again, fast for gaussians
        _expectation = expectation(beliefs[end], tuple())
        _variance = variance(beliefs[end], tuple())

		# Update plots
        push_plot_data!(_plot_data, _expectation, _variance)
    end
    plot_async_random_walk(_plot_data, observations)
end

# ╔═╡ Cell order:
# ╠═2f130e30-dbdf-11ec-12c6-0d0cb74dcfbe
# ╟─6131b416-6e38-4cad-839b-a668df2c1477
# ╠═e597edc8-635b-4d6d-beb9-455eeb603eab
# ╠═97e6059b-b7f0-4b17-9340-7aef21464349
# ╠═73c7c02b-99ad-495d-a507-7c2302fd5a73
# ╠═56773c97-6b4c-4dd8-8f24-d528eac54b70
# ╠═660f3c4a-740c-4388-80d5-9bd02412071b
