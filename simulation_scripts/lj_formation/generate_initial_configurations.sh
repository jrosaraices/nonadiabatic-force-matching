#!/bin/bash

python3 lj_formation_init.py \
    -work_dir './β_1.0/ρ_1.28/κ_100.0' -output_dir_suffix 'λ_0.0' \
    -β '1.0' -ρ '1.28' -κ '100.0' -λ '0.0' -dt '1e-5' -N_timesteps '1000000' -N_burnin '100000' -N_dumpevery '10000' -N_trajectories '100' \
    -first_traj_index '0' -gpu_id '0'
