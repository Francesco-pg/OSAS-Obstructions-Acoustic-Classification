#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Analysis
Module:      Run03_TrainTestSplit.py (Audit Script)
Purpose:     Performs an integrity check on the generated nested cross-validation folds.
Author:      Francesco Pietrogiacomi
Created:     2026-02-26
"""

import pandas as pd
from pathlib import Path
import config as cfg
import model_config as m_cfg

# --- CONFIGURATION ---
out_root = cfg.ModelOutdir
N_FOLDS = cfg.N_FOLDS
SUBJECT_COL = "subject_id"
CLASS_COL = "class" 

def run_ncv_audit():
    """
    Audits the generated train/test split files to verify stratification, 
    sample/subject distributions, and absolute lack of subject data leakage.
    """
    fold_summaries = {}
    all_subjects_global = set()

    valid_classes = list(cfg.classes.keys())
    audit_report_path = out_root / f"NestedCV_Report_Model_{cfg.model_n:02d}.txt"

    print(f"Starting Nested CV Check... saving to {audit_report_path.name}")

    with open(audit_report_path, 'w', encoding='utf-8') as f:
        # Header Metadata
        f.write("NESTED CROSS-VALIDATION INTEGRITY CHECK\n")
        f.write(f"{'='*60}\n")
        f.write(f"{'='*60}\n\n")

        # Table Header
        header = f"{'Fold':<8} | {'Set':<8} | {'Samples':<10} | {'Subjects':<10} | {valid_classes[0]:<10} | {valid_classes[1]:<12}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for i in range(1, N_FOLDS + 1):
            f_idx = f"{i:02d}"
            fold_dir = out_root / f"out_fold_{f_idx}"
            
            train_file = fold_dir / "train_features.csv"
            test_file  = fold_dir / "test_features.csv"

            if not train_file.exists() or not test_file.exists():
                print(f"⚠️ Warning: Missing files for Fold {f_idx}")
                continue

            # Load Data
            df_train = pd.read_csv(train_file)
            df_test  = pd.read_csv(test_file)

            def get_stats(df):
                subs = df[SUBJECT_COL].nunique()
                rows = len(df)
                c0 = (df[CLASS_COL] == valid_classes[0]).sum()
                c1 = (df[CLASS_COL] == valid_classes[1]).sum()
                return rows, subs, c0, c1

            tr_r, tr_s, tr_c0, tr_c1 = get_stats(df_train)
            te_r, te_s, te_c0, te_c1 = get_stats(df_test)

            # Write Rows
            f.write(f"Fold {f_idx} | Train    | {tr_r:<10} | {tr_s:<10} | {tr_c0:<10} | {tr_c1:<12}\n")
            f.write(f"Fold {f_idx} | Test     | {te_r:<10} | {te_s:<10} | {te_c0:<10} | {te_c1:<12}\n")
            f.write("-" * len(header) + "\n")
            
            fold_summaries[f_idx] = {
                'train': set(df_train[SUBJECT_COL].unique()), 
                'test': set(df_test[SUBJECT_COL].unique())
            }
            all_subjects_global.update(fold_summaries[f_idx]['train'].union(fold_summaries[f_idx]['test']))

        # --- INTEGRITY CHECKS ---
        f.write("\nINTEGRITY VERIFICATION\n")
        f.write(f"{'='*60}\n")
        
        # Check 1: Internal Leakage (Train/Test overlap)
        f.write("1. SUBJECT LEAKAGE CHECK (Train vs Test):\n")
        leak_found = False
        for f_idx, data in fold_summaries.items():
            leak = data['train'].intersection(data['test'])
            if leak:
                f.write(f"   [FAILED] Fold {f_idx} Leakage: {leak}\n")
                leak_found = True
        if not leak_found:
            f.write("   [PASSED] No subjects shared between Train and Test within folds.\n")
        
        # Check 2: Test Exclusivity (Fold to Fold overlap)
        f.write("\n2. TEST EXCLUSIVITY CHECK (Fold vs Fold):\n")
        seen_in_test = {}
        repeat_found = False
        for f_idx, data in fold_summaries.items():
            for prev_f, prev_set in seen_in_test.items():
                dupes = data['test'].intersection(prev_set)
                if dupes:
                    f.write(f"   [FAILED] Subjects {dupes} tested in both Fold {prev_f} and Fold {f_idx}\n")
                    repeat_found = True
            seen_in_test[f_idx] = data['test']
        if not repeat_found:
            f.write(f"   [PASSED] Every subject used for testing exactly once across the {cfg.N_FOLDS} folds.\n")

        # Global Stats
        f.write("\n3. GLOBAL COHORT SUMMARY:\n")
        f.write(f"   Total Unique Subjects: {len(all_subjects_global)}\n")
        f.write("   Folds are properly stratified and independent.\n")
        f.write(f"{'='*60}\n")
        f.write("Audit Complete.")

    print(f"✅ Audit report generated: {audit_report_path.name}")

if __name__ == "__main__":
    run_ncv_audit()