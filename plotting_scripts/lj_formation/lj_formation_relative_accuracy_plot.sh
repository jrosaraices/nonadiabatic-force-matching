#!/bin/bash

python3 lj_formation_relative_accuracy_plot.py \
    -forward_filenames \
         "../../post-processing_scripts/lj_formation/forward_bound/data_τ_0.01.log" \
         "../../post-processing_scripts/lj_formation/forward_bound/data_τ_0.02.log" \
         "../../post-processing_scripts/lj_formation/forward_bound/data_τ_0.05.log" \
         "../../post-processing_scripts/lj_formation/forward_bound/data_τ_0.10.log" \
         "../../post-processing_scripts/lj_formation/forward_bound/data_τ_1.00.log" \
    -forward_key "wt" \
    -backward_filenames \
         "../../post-processing_scripts/lj_formation/backward_bound/data_τ_0.01.log" \
         "../../post-processing_scripts/lj_formation/backward_bound/data_τ_0.02.log" \
         "../../post-processing_scripts/lj_formation/backward_bound/data_τ_0.05.log" \
         "../../post-processing_scripts/lj_formation/backward_bound/data_τ_0.10.log" \
         "../../post-processing_scripts/lj_formation/backward_bound/data_τ_1.00.log" \
    -backward_key "wt" \
    -reference_filename \
        "../../post-processing_scripts/lj_formation/forward_bound/data_τ_10.0.log" \
    -reference_key "wt" \
    -rescaling_factor "500" -ylim "-4.0" "3.0"
