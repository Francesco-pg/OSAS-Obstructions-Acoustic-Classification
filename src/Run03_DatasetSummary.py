#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Analysis
Module:      Run02b_DatasetSummary.py
Purpose:     Provides summary statistics for the extracted feature dataset.
Author:      Francesco Pietrogiacomi
Created:     2026-03-02
"""

import pandas as pd
import config
import model_config as m_cfg

def main():
    print(f"--- DATASET SUMMARY: {config.full_dataset_path.name} ---")
    
    if not config.full_dataset_path.exists():
        print(f"❌ Error: Full dataset not found at {config.full_dataset_path}")
        print("Please run Run02_FeatureExtraction.py first.")
        return

    # Load the dataset (Run02 outputs an .xlsx file)
    df = pd.read_excel(config.full_dataset_path)

    ## 0. Filter for specific classes
    valid_classes = ['epiglottide', 'palato']
    df = df[df['class'].isin(valid_classes)].copy()

    ## 0.1 Calculate Feature Count (Excluding metadata/IDs)
    feature_cols = [c for c in df.columns if c not in m_cfg.EXCLUDE_COLS]
    total_features = len(feature_cols)

    ## 1. Total Number of Samples (Rows)
    total_samples = len(df)

    ## 2. Total Number of Unique Subjects
    total_subjects = df['subject_id'].nunique()

    ## 3. Samples per Subject (Mean and Standard Deviation)
    samples_per_subject = df.groupby('subject_id').size()
    mean_samples = samples_per_subject.mean()
    std_samples = samples_per_subject.std()

    ## 4. Number of Samples per Class
    class_counts = df['class'].value_counts()

    # --- Output Results ---
    print("\n--- Dataset Summary Statistics ---")
    print(f"Total Samples (Rows):           {total_samples}")
    print(f"Total Features (Filtered):      {total_features}")
    print(f"Total Unique Subjects:          {total_subjects}")
    print(f"Mean Samples per Subject:       {mean_samples:.2f}")
    print(f"Std Dev of Samples per Subject: {std_samples:.2f}")
    print("\nSamples per Class:")
    print(class_counts.to_string())

if __name__ == "__main__":
    main()