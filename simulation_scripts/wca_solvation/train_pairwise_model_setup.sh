#!/bin/bash

python3 train_pairwise_model_setup.py \
    -gsd_dirname "./β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_0.50" -gsd_basename_glob "forward_*.gsd" \
    -trajectory_duration "0.5" -frame_step "10" -i_type "A" -j_type "B" \
    -num_neighs "24" -output_basename "pairwise_data_24_neighs.npz"

python3 train_pairwise_model_setup.py \
    -gsd_dirname "./β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_1.00" -gsd_basename_glob "forward_*.gsd" \
    -trajectory_duration "1.0" -frame_step "10" -i_type "A" -j_type "B" \
    -num_neighs "24" -output_basename "pairwise_data_24_neighs.npz"

python3 train_pairwise_model_setup.py \
    -gsd_dirname "./β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_2.50" -gsd_basename_glob "forward_*.gsd" \
    -trajectory_duration "2.5" -frame_step "10" -i_type "A" -j_type "B" \
    -num_neighs "24" -output_basename "pairwise_data_24_neighs.npz"

python3 train_pairwise_model_setup.py \
    -gsd_dirname "./β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_5.00" -gsd_basename_glob "forward_*.gsd" \
    -trajectory_duration "5.0" -frame_step "10" -i_type "A" -j_type "B" \
    -num_neighs "24" -output_basename "pairwise_data_24_neighs.npz"
