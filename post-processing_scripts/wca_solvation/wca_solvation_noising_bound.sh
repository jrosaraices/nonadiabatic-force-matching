#!/bin/bash

mkdir "./noising_bound"

python3 wca_solvation_noising_bound.py \
    -data_dirname "../../simulation_scripts/wca_solvation/β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_50.0" \
    -data_baseglob "noising_*.txt" > "./noising_bound/data_τ_50.0.log"
python3 wca_solvation_noising_bound.py \
    -data_dirname "../../simulation_scripts/wca_solvation/β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_5.00" \
    -data_baseglob "noising_*.txt" > "./noising_bound/data_τ_5.00.log"
python3 wca_solvation_noising_bound.py \
    -data_dirname "../../simulation_scripts/wca_solvation/β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_2.50" \
    -data_baseglob "noising_*.txt" > "./noising_bound/data_τ_2.50.log"
python3 wca_solvation_noising_bound.py \
    -data_dirname "../../simulation_scripts/wca_solvation/β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_1.00" \
    -data_baseglob "noising_*.txt" > "./noising_bound/data_τ_1.00.log"
python3 wca_solvation_noising_bound.py \
    -data_dirname "../../simulation_scripts/wca_solvation/β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_0.50" \
    -data_baseglob "noising_*.txt" > "./noising_bound/data_τ_0.50.log"
python3 wca_solvation_noising_bound.py \
    -data_dirname "../../simulation_scripts/wca_solvation/β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_0.25" \
    -data_baseglob "noising_*.txt" > "./noising_bound/data_τ_0.25.log"
python3 wca_solvation_noising_bound.py \
    -data_dirname "../../simulation_scripts/wca_solvation/β_0.824/ρ_0.96/φ_2.0/λ_0.0/τ_0.10" \
    -data_baseglob "noising_*.txt" > "./noising_bound/data_τ_0.10.log"
