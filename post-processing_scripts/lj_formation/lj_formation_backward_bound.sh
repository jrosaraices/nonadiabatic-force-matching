#!/bin/bash

mkdir "./backward_bound"

python3 lj_formation_backward_bound.py \
    -data_dirname "../../simulation_scripts/lj_formation/β_1.0/ρ_1.28/κ_100.0/λ_0.0/τ_10.0/backward" \
    -data_baseglob "backward_*.txt" > "./backward_bound/data_τ_10.0.log"
python3 lj_formation_backward_bound.py \
    -data_dirname "../../simulation_scripts/lj_formation/β_1.0/ρ_1.28/κ_100.0/λ_0.0/τ_1.00/backward" \
    -data_baseglob "backward_*.txt" > "./backward_bound/data_τ_1.00.log"
python3 lj_formation_backward_bound.py \
    -data_dirname "../../simulation_scripts/lj_formation/β_1.0/ρ_1.28/κ_100.0/λ_0.0/τ_0.50/backward" \
    -data_baseglob "backward_*.txt" > "./backward_bound/data_τ_0.50.log"
python3 lj_formation_backward_bound.py \
    -data_dirname "../../simulation_scripts/lj_formation/β_1.0/ρ_1.28/κ_100.0/λ_0.0/τ_0.20/backward" \
    -data_baseglob "backward_*.txt" > "./backward_bound/data_τ_0.20.log"
python3 lj_formation_backward_bound.py \
    -data_dirname "../../simulation_scripts/lj_formation/β_1.0/ρ_1.28/κ_100.0/λ_0.0/τ_0.10/backward" \
    -data_baseglob "backward_*.txt" > "./backward_bound/data_τ_0.10.log"
python3 lj_formation_backward_bound.py \
    -data_dirname "../../simulation_scripts/lj_formation/β_1.0/ρ_1.28/κ_100.0/λ_0.0/τ_0.05/backward" \
    -data_baseglob "backward_*.txt" > "./backward_bound/data_τ_0.05.log"
python3 lj_formation_backward_bound.py \
    -data_dirname "../../simulation_scripts/lj_formation/β_1.0/ρ_1.28/κ_100.0/λ_0.0/τ_0.02/backward" \
    -data_baseglob "backward_*.txt" > "./backward_bound/data_τ_0.02.log"
python3 lj_formation_backward_bound.py \
    -data_dirname "../../simulation_scripts/lj_formation/β_1.0/ρ_1.28/κ_100.0/λ_0.0/τ_0.01/backward" \
    -data_baseglob "backward_*.txt" > "./backward_bound/data_τ_0.01.log"
