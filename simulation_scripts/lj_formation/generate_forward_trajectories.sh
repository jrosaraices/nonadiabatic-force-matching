#!/bin/bash

python3 lj_formation_run.py \
    -work_dir './β_1.0/ρ_1.28/κ_100.0/λ_0.0' -input_name 'equilibrium' -output_dir_suffix 'τ_10.0' \
    -β '1.0' -ρ '1.28' -κ '100.0' -λ '0.0' -dt '1e-5' -N_timesteps '1000000' -N_dumpevery '10000' -N_trajectories '100' \
    -first_traj_index '0' -gpu_id '0'

python3 lj_formation_run.py \
    -work_dir './β_1.0/ρ_1.28/κ_100.0/λ_0.0' -input_name 'equilibrium' -output_dir_suffix 'τ_1.00' \
    -β '1.0' -ρ '1.28' -κ '100.0' -λ '0.0' -dt '1e-5' -N_timesteps '100000' -N_dumpevery '1000' -N_trajectories '100' \
    -first_traj_index '0' -gpu_id '0'

python3 lj_formation_run.py \
    -work_dir './β_1.0/ρ_1.28/κ_100.0/λ_0.0' -input_name 'equilibrium' -output_dir_suffix 'τ_0.50' \
    -β '1.0' -ρ '1.28' -κ '100.0' -λ '0.0' -dt '1e-5' -N_timesteps '50000' -N_dumpevery '500' -N_trajectories '100' \
    -first_traj_index '0' -gpu_id '0'

python3 lj_formation_run.py \
    -work_dir './β_1.0/ρ_1.28/κ_100.0/λ_0.0' -input_name 'equilibrium' -output_dir_suffix 'τ_0.20' \
    -β '1.0' -ρ '1.28' -κ '100.0' -λ '0.0' -dt '1e-5' -N_timesteps '20000' -N_dumpevery '200' -N_trajectories '100' \
    -first_traj_index '0' -gpu_id '0'

python3 lj_formation_run.py \
    -work_dir './β_1.0/ρ_1.28/κ_100.0/λ_0.0' -input_name 'equilibrium' -output_dir_suffix 'τ_0.10' \
    -β '1.0' -ρ '1.28' -κ '100.0' -λ '0.0' -dt '1e-5' -N_timesteps '10000' -N_dumpevery '100' -N_trajectories '100' \
    -first_traj_index '0' -gpu_id '0'

python3 lj_formation_run.py \
    -work_dir './β_1.0/ρ_1.28/κ_100.0/λ_0.0' -input_name 'equilibrium' -output_dir_suffix 'τ_0.05' \
    -β '1.0' -ρ '1.28' -κ '100.0' -λ '0.0' -dt '1e-5' -N_timesteps '5000' -N_dumpevery '50' -N_trajectories '100' \
    -first_traj_index '0' -gpu_id '0'

python3 lj_formation_run.py \
    -work_dir './β_1.0/ρ_1.28/κ_100.0/λ_0.0' -input_name 'equilibrium' -output_dir_suffix 'τ_0.02' \
    -β '1.0' -ρ '1.28' -κ '100.0' -λ '0.0' -dt '1e-5' -N_timesteps '2000' -N_dumpevery '20' -N_trajectories '100' \
    -first_traj_index '0' -gpu_id '0'

python3 lj_formation_run.py \
    -work_dir './β_1.0/ρ_1.28/κ_100.0/λ_0.0' -input_name 'equilibrium' -output_dir_suffix 'τ_0.01' \
    -β '1.0' -ρ '1.28' -κ '100.0' -λ '0.0' -dt '1e-5' -N_timesteps '1000' -N_dumpevery '10' -N_trajectories '100' \
    -first_traj_index '0' -gpu_id '0'
