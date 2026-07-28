#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Features
Module:      config.py
Purpose:     Configuration file to centralize all paths and general settings.
Author:      Francesco Pietrogiacomi
Created:     2026-02-25
"""

import os
from pathlib import Path

### --- PATHS AND NAMES --- ###

# Dynamically resolve paths so outputs are created next to the repository folder
src_dir = Path(__file__).resolve().parent
projdir = src_dir.parent
outputsdir = projdir.parent / 'outputs'  # General outputs directory next to the repo

# USER CONFIGURATION: Change this path to where your raw dataset is located.
# (e.g., after manual cropping of silence before and after the event of interest)
datadir = Path("D://Datasets//AI_healthcare//osas-project//osas_data_cleaned//subjects")
stereo_dataset_dir = datadir / "full_stereo" # This is the folder where the raw stereo .wav files are located (Reaper's output)

currentOutName = 'output_NCV5X5_01'  # Name of output folder
ModelOutdir = outputsdir / currentOutName / 'model'
AnalysisOutdir = outputsdir / currentOutName / 'analysis'

full_dataset_name = "full_dataset"
full_dataset_path = outputsdir / f"{full_dataset_name}.xlsx"

# Create model output directory if it doesn't exist
if not os.path.exists(ModelOutdir): 
    os.makedirs(ModelOutdir)

# Processed data directories 
mono_dataset_name = "Dataset_OSAS_clean_mono"
mono_dataset_dir  = datadir / mono_dataset_name

normalized_mono_dataset_name = "Dataset_OSAS_clean_mono_norm"
normalized_mono_dataset_dir  = datadir / normalized_mono_dataset_name

# Final Dataset used for Splitting: mono, normalized, and resampled
resampled_normalized_mono_dataset_name = "Dataset_OSAS_clean_mono_norm_res"
resampled_normalized_mono_dataset_dir  = datadir / resampled_normalized_mono_dataset_name

# Point source to the final prepared dataset
source_dataset_dir      = resampled_normalized_mono_dataset_dir 
classes_report_filename = "classes_report.xlsx"
classes_report_path     = outputsdir / classes_report_filename


def get_fold_paths(fold_idx):
    """
    Generate and return fold-specific input/output paths.
    
    Args:
        fold_idx (int): The index of the current fold.
        
    Returns:
        dict: A dictionary containing paths for models, features, and results.
    """
    fold_name = f"fold_{fold_idx:02d}"

    ModFoldOuDir = ModelOutdir / f"out_{fold_name}"
    AnaFoldOuDir = AnalysisOutdir / f"out_{fold_name}"
    
    if not os.path.exists(ModFoldOuDir): 
        os.makedirs(ModFoldOuDir)
    if not os.path.exists(AnalysisOutdir): 
        os.makedirs(AnalysisOutdir)

    train_feat_file = "train_features.csv"
    test_feat_file  = "test_features.csv"

    return {
        "mod_output_dir": ModFoldOuDir,
        "ana_output_dir": AnaFoldOuDir,
        
        "train_features": ModFoldOuDir / train_feat_file,
        "test_features" : ModFoldOuDir / test_feat_file, 
        
        "model_output": ModFoldOuDir / f"best_model_{model_n:02d}.pkl",
        "results_txt": ModFoldOuDir / f"results_fold_{fold_idx:02d}_model_{model_n:02d}.txt",
        "pruning_report": ModFoldOuDir / f"feature_pruning_report_model_{model_n:02d}.txt"
    }


### --- GENERAL SETTINGS --- ###

split_random_state = 1  # Used only for train-test splitting

# --- MODEL ID --- 
model_n = 1  # ID of the NEW model you want to SAVE (e.g., 2 for 'best_model_02.pkl')

# --- NESTED CROSS VAL: TRAIN TEST SPLIT CONFIGURATION ---
N_FOLDS = 5  # Number of folds for train/test split (outer loop of NESTED CV)

# Set to False to disable, or use strings like '_mean', '_median', '_std' to remove specific features.
REMOVE_FTS = False 

# Formant extraction method 
FORMANT_METHOD = 'lpc'

# Classes names
classes = {'palato': 0, 'epiglottide': 1}

# Resampling configuration
CURRENT_SR = 44100
TARGET_SR  = 22050

# RMS normalization methods
rmsTarget = 'min'  # Options: 'min', 'max', 'mean', 'median'