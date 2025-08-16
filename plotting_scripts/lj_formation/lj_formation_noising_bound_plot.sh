#!/bin/bash

python3 lj_formation_noising_bound_plot.py \
    -data_files \
        "../../post-processing_scripts/lj_formation/noising_bound/data_τ_0.01.log" \
        "../../post-processing_scripts/lj_formation/noising_bound/data_τ_0.02.log" \
        "../../post-processing_scripts/lj_formation/noising_bound/data_τ_0.05.log" \
        "../../post-processing_scripts/lj_formation/noising_bound/data_τ_0.10.log" \
        "../../post-processing_scripts/lj_formation/noising_bound/data_τ_0.20.log" \
        "../../post-processing_scripts/lj_formation/noising_bound/data_τ_0.50.log" \
        "../../post-processing_scripts/lj_formation/noising_bound/data_τ_1.00.log" \
        "../../post-processing_scripts/lj_formation/noising_bound/data_τ_10.0.log" \
    -rescaling_factor "500" -ylim "-4.0" "3.0"
