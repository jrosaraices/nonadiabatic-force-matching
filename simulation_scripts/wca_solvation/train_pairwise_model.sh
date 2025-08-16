#!/bin/bash

python3 train_pairwise_model.py \
    -npz_dirname "./β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_0.50" -npz_basename "pairwise_data_24_neighs.npz" \
    -model_basename "pairwise_model_24_neighs.pt" -num_inputs "24" -num_neurons "96" -num_layers "1" \
    -num_epochs "100000" -batch_size "2000" -learning_rate "0.05"

python3 train_pairwise_model.py \
    -npz_dirname "./β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_1.00" -npz_basename "pairwise_data_24_neighs.npz" \
    -model_basename "pairwise_model_24_neighs.pt" -num_inputs "24" -num_neurons "96" -num_layers "1" \
    -num_epochs "100000" -batch_size "4000" -learning_rate "0.05"

python3 train_pairwise_model.py \
    -npz_dirname "./β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_2.50" -npz_basename "pairwise_data_24_neighs.npz" \
    -model_basename "pairwise_model_24_neighs.pt" -num_inputs "24" -num_neurons "96" -num_layers "1" \
    -num_epochs "100000" -batch_size "10000" -learning_rate "0.05"

python3 train_pairwise_model.py \
    -npz_dirname "./β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_5.00" -npz_basename "pairwise_data_24_neighs.npz" \
    -model_basename "pairwise_model_24_neighs.pt" -num_inputs "24" -num_neurons "96" -num_layers "1" \
    -num_epochs "100000" -batch_size "20000" -learning_rate "0.05"
