function dynamic_pf(n1, n2, num_particles, num_steps)
    run_filter(n1, n2, SyncPF(num_particles), num_steps)
end