#!/bin/bash

python3 wca_monomer_init.py \
    -work_dir "./β_0.824/ρ_0.96/φ_2.0" -output_dir_suffix "λ_0.0" \
    -β "0.824" -ρ "0.96" -φ "2.0" -λ "0.0" -dt "5e-5" \
    -N_timesteps "500000" -N_burnin "500000" -N_dumpevery "2000" -N_trajectories "100" \
    -first_traj_index "0" -gpu_id "0"
