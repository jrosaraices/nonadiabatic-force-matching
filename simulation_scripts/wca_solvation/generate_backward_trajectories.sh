#!/bin/bash

python3 wca_monomer_backward.py \
    -work_dir "./β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_0.50" -output_dir_suffix "backward" \
    -β "0.824" -ρ "0.96" -φ "2.0" -λ "0.0" -dt "5e-5" -model_basename "pairwise_model_24_neighs.pt" \
    -N_timesteps "10000" -N_dumpevery "200" -N_trajectories "100" \
    -first_traj_index "0" -gpu_id "0"

python3 wca_monomer_backward.py \
    -work_dir "./β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_1.00" -output_dir_suffix "backward" \
    -β "0.824" -ρ "0.96" -φ "2.0" -λ "0.0" -dt "5e-5" -model_basename "pairwise_model_24_neighs.pt" \
    -N_timesteps "20000" -N_dumpevery "200" -N_trajectories "100" \
    -first_traj_index "0" -gpu_id "0"

python3 wca_monomer_backward.py \
    -work_dir "./β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_2.50" -output_dir_suffix "backward" \
    -β "0.824" -ρ "0.96" -φ "2.0" -λ "0.0" -dt "5e-5" -model_basename "pairwise_model_24_neighs.pt" \
    -N_timesteps "50000" -N_dumpevery "200" -N_trajectories "100" \
    -first_traj_index "0" -gpu_id "0"

python3 wca_monomer_backward.py \
    -work_dir "./β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_5.00" -output_dir_suffix "backward" \
    -β "0.824" -ρ "0.96" -φ "2.0" -λ "0.0" -dt "5e-5" -model_basename "pairwise_model_24_neighs.pt" \
    -N_timesteps "100000" -N_dumpevery "200" -N_trajectories "100" \
    -first_traj_index "0" -gpu_id "0"
