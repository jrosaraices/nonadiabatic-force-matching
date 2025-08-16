#!/bin/bash

mkdir "./noising_bound"

python3 lj_formation_noising_bound.py \
    -data_dirname "../../simulation_scripts/lj_formation/β_1.0/ρ_1.28/κ_100.0/λ_0.0/τ_10.0" \
    -data_baseglob "noising_*.txt" > "./noising_bound/data_τ_10.0.log"
python3 lj_formation_noising_bound.py \
    -data_dirname "../../simulation_scripts/lj_formation/β_1.0/ρ_1.28/κ_100.0/λ_0.0/τ_1.00" \
    -data_baseglob "noising_*.txt" > "./noising_bound/data_τ_1.00.log"
python3 lj_formation_noising_bound.py \
    -data_dirname "../../simulation_scripts/lj_formation/β_1.0/ρ_1.28/κ_100.0/λ_0.0/τ_0.50" \
    -data_baseglob "noising_*.txt" > "./noising_bound/data_τ_0.50.log"
python3 lj_formation_noising_bound.py \
    -data_dirname "../../simulation_scripts/lj_formation/β_1.0/ρ_1.28/κ_100.0/λ_0.0/τ_0.20" \
    -data_baseglob "noising_*.txt" > "./noising_bound/data_τ_0.20.log"
python3 lj_formation_noising_bound.py \
    -data_dirname "../../simulation_scripts/lj_formation/β_1.0/ρ_1.28/κ_100.0/λ_0.0/τ_0.10" \
    -data_baseglob "noising_*.txt" > "./noising_bound/data_τ_0.10.log"
python3 lj_formation_noising_bound.py \
    -data_dirname "../../simulation_scripts/lj_formation/β_1.0/ρ_1.28/κ_100.0/λ_0.0/τ_0.05" \
    -data_baseglob "noising_*.txt" > "./noising_bound/data_τ_0.05.log"
python3 lj_formation_noising_bound.py \
    -data_dirname "../../simulation_scripts/lj_formation/β_1.0/ρ_1.28/κ_100.0/λ_0.0/τ_0.02" \
    -data_baseglob "noising_*.txt" > "./noising_bound/data_τ_0.02.log"
python3 lj_formation_noising_bound.py \
    -data_dirname "../../simulation_scripts/lj_formation/β_1.0/ρ_1.28/κ_100.0/λ_0.0/τ_0.01" \
    -data_baseglob "noising_*.txt" > "./noising_bound/data_τ_0.01.log"
