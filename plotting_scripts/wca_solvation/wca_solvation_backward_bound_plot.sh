#!/bin/bash

python3 wca_solvation_backward_bound_plot.py \
    -data_files \
        "../../post-processing_scripts/wca_solvation/backward_bound/data_τ_0.50.log" \
        "../../post-processing_scripts/wca_solvation/backward_bound/data_τ_1.00.log" \
        "../../post-processing_scripts/wca_solvation/backward_bound/data_τ_2.50.log" \
        "../../post-processing_scripts/wca_solvation/backward_bound/data_τ_5.00.log" \
        "../../post-processing_scripts/wca_solvation/backward_bound/data_τ_50.0.log" \
    -ylim "-26.0" "1.0"
