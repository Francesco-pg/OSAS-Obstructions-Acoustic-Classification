#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Analysis
Module:      Run02_FeatureExtraction.py
Purpose:     Extract acoustic features from all the wav files outputted from Run01_Dataset.
Author:      Francesco Pietrogiacomi
Created:     2026-02-25
"""

# Third-party imports
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Custom imports
import config
from processing.f0 import extract_f0_with_dicts
from processing import spectral_feat
from utils import files_handle as fh
from utils.ful_df_from_dict import build_osas_dataframe, extract_f0_stats


def extract_features(input_dir, output_path):
    """
    Extracts F0, spectral, and advanced acoustic features from the audio dataset
    and merges them into a single comprehensive dataset.
    """
    print("  > Extracting features")
    
    # 1. Read Classes Report 
    classes_report = config.classes_report_path
    fh.scan_files_for_classes(input_dir, [".wav"], classes_report)
    iddf = pd.read_excel(classes_report)

    # 2. F0 Extraction
    # We capture nan_ratios (the first return value) instead of ignoring it
    nan_ratios, f0_arrays = extract_f0_with_dicts(
        datapath  = input_dir, 
        outname   = "dummy_f0", 
        outpath   = input_dir.parent, 
        ext       = "wav", 
        savetable = False
    )
    
    # Clean keys (remove path, keep stem)
    f0_dict = {Path(k).stem: v for k, v in f0_arrays.items()}
    
    # Create the nan_dict needed for build_osas_dataframe
    # The keys in nan_ratios usually mimic the file path or name, we ensure they are stems
    nan_dict = {Path(k).stem: v for k, v in nan_ratios.items()}
    path_map = {f.stem: f for f in list(input_dir.rglob("*.wav"))}
    
    # 3. Build F0 DF
    df_f0 = build_osas_dataframe(nan_dict, f0_dict, path_map) 
    
    f0_stats   = df_f0["f0_array"].apply(extract_f0_stats)
    df_f0_full = pd.concat([df_f0, f0_stats], axis=1).drop(
        columns=['class', 'subject_id', 'file_path'], errors='ignore'
    )

    # 4. Extract All Features
    print("    - Extracting Features...")
    features_list = []
    for _, row in tqdm(iddf.iterrows(), total=len(iddf)):
        f0_val = f0_dict.get(row['id'])
        feats  = spectral_feat.extract_all_features(Path(row['file_path']), f0=f0_val)
        features_list.append(feats)
    
    df_features = pd.DataFrame(features_list)
    df_features['id'] = iddf['id']

    # 5. Merge
    print("    - Merging...")
    df_full = fh.concatenate_dataframes([iddf, df_f0_full, df_features], id_columns=["id"])
    
    # --- Dataset Summary Statistics ---
    print("\n--- Dataset Summary Statistics ---")
    total_samples = len(df_full)
    total_subjects = df_full['subject_id'].nunique()
    samples_per_subject = df_full.groupby('subject_id').size()
    
    print(f"Total Samples (Rows):           {total_samples}")
    print(f"Total Unique Subjects:          {total_subjects}")
    print(f"Mean Samples per Subject:       {samples_per_subject.mean():.2f}")
    print(f"Std Dev of Samples per Subject: {samples_per_subject.std():.2f}")
    print("\nSamples per Class:")
    print(df_full['class'].value_counts().to_string())

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_excel(output_path, index=False)


def main():
    print("--- FEATURE EXTRACTION ---")
    path = config.resampled_normalized_mono_dataset_dir 
    print(f"\n>>> Data Source {path}")
    
    extract_features(
        input_dir   = path,
        output_path = config.full_dataset_path
    )
    

if __name__ == "__main__":
    main()