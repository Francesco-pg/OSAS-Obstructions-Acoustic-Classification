#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Analysis
Module:      run01_dataset.py
Purpose:     Script to prepare the dataset for the OSAS classification project. This includes:
             1. Extracting single channel from stereo recordings;
             2. Normalizing audio files using RMS normalization;
             3. Resampling audio files to a consistent sampling rate;
             4. Generating a classes report for downstream processing.
Author:      Francesco Pietrogiacomi
Created:     2026-02-25
"""

# Custom imports
import config
from utils import files_handle as fh
from processing import resampling as res

from processing import normalization as norm
from processing import extract_singl_chn 

def main():
    """
    Main pipeline for preparing the acoustic dataset.
    """
    # Paths
    datadir   = config.datadir             # Raw data folder
    stereodir = config.stereo_dataset_dir  # Raw stereo files

    print("--- 1. EXTRACT SINGLE CHANNEL ---")
    extract_singl_chn.extract_single_channel(
        target_path = stereodir, 
        output_path = datadir, 
        outdir_name = config.mono_dataset_name
    )
    
    print("\n--- 2. RMS NORMALIZATION (Global) ---")
    # Takes Mono Dataset -> Outputs Normalized Dataset
    # Outputs normalized data and returns nan_ratios for each file (used in feature extraction)
    norm.normalize_rms_wav_librosa(
        root_folder        = config.mono_dataset_dir,
        output_folder_name = config.normalized_mono_dataset_name,
        target             = config.rmsTarget
    )

    print("\n--- 3. RESAMPLING ---")
    # Takes Normalized Dataset -> Outputs Resampled Dataset
    res.nuovo_campionamento(  # TODO: Change the function name in the final version
        target_path     = config.normalized_mono_dataset_dir, 
        new_folder_path = config.resampled_normalized_mono_dataset_dir, 
        freq_normal     = config.CURRENT_SR, 
        freq_resampled  = config.TARGET_SR
    )

    print("\n--- 4. GENERATING CLASSES REPORT ---")
    # The data source for the rest of the pipeline is now the Resampled-Normalized folder
    target  = config.resampled_normalized_mono_dataset_dir
    outpath = config.classes_report_path
    fh.scan_files_for_classes(target, [".wav"], outpath)
    
    print("✅ Dataset Preparation Complete.")

if __name__ == "__main__":
    main()