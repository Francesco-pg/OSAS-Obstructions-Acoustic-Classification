# -*- coding: utf-8 -*-
"""
Module:      splitting.py
Purpose:     Handles Stratified Group K-Fold splitting with leakage checks.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold


def create_N_folds(table, cfg, n_folds, classes, plot_sgkf=False):
    """
    Generates N folds for cross-validation using StratifiedGroupKFold. 
    Includes a reroll logic to ensure both classes are represented in every 
    test set, and performs strict leakage checks.
    """
    valid_classes = list(classes.keys())

    # Filter dataframe to only include valid classes
    df = table[table['class'].isin(valid_classes)].reset_index(drop=True)

    X = df['file_path'].values
    y = df['class'].values
    groups = df['subject_id'].values

    # --- THE REROLL LOGIC ---
    max_attempts = 200
    winning_seed = None
    valid_folds = []

    print(f"Searching for a stable {n_folds}-fold split (both classes in all test sets)...")
    
    for seed in range(cfg.split_random_state, cfg.split_random_state + max_attempts):
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        current_attempt_folds = []
        is_balanced = True
        
        for train_idx, test_idx in sgkf.split(X, y, groups):
            test_df_check = df.iloc[test_idx]
            
            # Verify both classes exist in this test set
            if test_df_check['class'].nunique() < 2:
                is_balanced = False
                break
                
            current_attempt_folds.append((train_idx, test_idx))
            
        # Fix: Use the passed n_folds argument rather than the direct config call
        if is_balanced and len(current_attempt_folds) == n_folds:
            winning_seed = seed
            valid_folds = current_attempt_folds
            break

    if winning_seed is None:
        raise ValueError(
            f"CRITICAL: Could not find a valid split after {max_attempts} seeds. "
            f"You may have too few subjects in one class to support {n_folds} folds."
        )

    print(f"✅ Success! Valid split found using seed: {winning_seed}")

    # --- SPLITTING PHASE ---
    for fold_idx, (train_idx, test_idx) in enumerate(valid_folds, start=1):
        paths = cfg.get_fold_paths(fold_idx)
        print(f"\n--- Processing Fold {fold_idx} ---")
        
        train_df = df.iloc[train_idx]
        test_df  = df.iloc[test_idx]
        
        # Leakage Check (Double Verification)
        train_sub = set(train_df['subject_id'].unique())
        test_sub  = set(test_df['subject_id'].unique())
        
        if train_sub.intersection(test_sub):
            raise ValueError(f"CRITICAL: Data Leakage Detected in Fold {fold_idx}")
        
        print(f"Train subjects: {len(train_sub)} | Test subjects: {len(test_sub)}")

        # Save Fold Data
        train_df.to_csv(paths['train_features'], index=False)
        test_df.to_csv(paths['test_features'], index=False)
        
        # Save Fold Statistics Report
        _save_fold_report(fold_idx, train_df, test_df, paths['mod_output_dir'], winning_seed)


def _save_fold_report(fold_idx, train_df, test_df, output_dir, winning_seed):
    """
    Saves a detailed statistical summary for the given fold to a text file.
    """
    # Calculate Statistics
    total_samples = len(train_df) + len(test_df)
    train_pct     = (len(train_df) / total_samples) * 100
    test_pct      = (len(test_df) / total_samples) * 100

    train_subs = train_df['subject_id'].nunique()
    test_subs  = test_df['subject_id'].nunique()
    
    # Class Distribution
    train_counts = train_df['class'].value_counts(normalize=True) * 100
    test_counts  = test_df['class'].value_counts(normalize=True) * 100

    # Write the report
    report_path = output_dir / f"fold_{fold_idx:02d}_stats.txt"
    with open(report_path, 'w') as f:
        f.write(f"--- FOLD {fold_idx:02d} SPLIT STATISTICS ---\n")
        f.write(f"Winning Random Seed (Reroll): {winning_seed}\n")
        f.write(f"Total Samples in Fold: {total_samples}\n")
        f.write("-" * 40 + "\n")
        
        f.write(f"TRAIN SET:\n")
        f.write(f"  Samples:  {len(train_df)} ({train_pct:.1f}%)\n")
        f.write(f"  Subjects: {train_subs}\n")
        for cls, pct in train_counts.items():
            f.write(f"  Class '{cls}': {pct:.1f}%\n")
            
        f.write("\nTEST SET:\n")
        f.write(f"  Samples:  {len(test_df)} ({test_pct:.1f}%)\n")
        f.write(f"  Subjects: {test_subs}\n")
        for cls, pct in test_counts.items():
            f.write(f"  Class '{cls}': {pct:.1f}%\n")
        
        f.write("-" * 40 + "\n")
        f.write("LEAKAGE CHECK:\n")
        intersection = set(train_df['subject_id']).intersection(set(test_df['subject_id']))
        f.write(f"  Overlapping Subjects: {len(intersection)}\n")
        if len(intersection) == 0:
            f.write("  Status: PASSED (No leakage)\n")
        else:
            f.write("  Status: FAILED (Data Leakage Detected!)\n")

    print(f"  -> Split stats saved to: {report_path.name}")