#!/bin/bash

python3 wca_solvation_relative_accuracy_plot.py \
    -noising_filenames \
        "../../post-processing_scripts/wca_solvation/noising_bound/data_τ_0.50.log" \
        "../../post-processing_scripts/wca_solvation/noising_bound/data_τ_1.00.log" \
        "../../post-processing_scripts/wca_solvation/noising_bound/data_τ_2.50.log" \
        "../../post-processing_scripts/wca_solvation/noising_bound/data_τ_5.00.log" \
    -noising_key "wt" \
    -denoising_filenames \
        "../../post-processing_scripts/wca_solvation/denoising_bound/data_τ_0.50.log" \
        "../../post-processing_scripts/wca_solvation/denoising_bound/data_τ_1.00.log" \
        "../../post-processing_scripts/wca_solvation/denoising_bound/data_τ_2.50.log" \
        "../../post-processing_scripts/wca_solvation/denoising_bound/data_τ_5.00.log" \
    -denoising_key "ft" \
    -reference_filename \
        "../../post-processing_scripts/wca_solvation/noising_bound/data_τ_50.0.log" \
    -reference_key "wt" \
    -ylim "-8.0" "8.0"
