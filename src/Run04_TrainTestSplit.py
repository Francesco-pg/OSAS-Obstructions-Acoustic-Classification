#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Analysis
Module:      Run03_TrainTestSplit.py
Purpose:     Split the full_dataset.xlsx (Run02 output) into n folds.
Author:      Francesco Pietrogiacomi
Created:     2026-02-26
Version:     1.0.0
"""

# Third-party imports
import pandas as pd

# Custom imports
import config
from training import splitting


def main():
    """
    Reads the full feature dataset and splits it into N stratified groups 
    (folds) for cross-validation, ensuring no subject data leakage between 
    train and test sets.
    """
    print(f"--- {config.N_FOLDS}-FOLD SPLITTING ---")

    if not config.full_dataset_path.exists():
        print(f"❌ Error: Full dataset not found at {config.full_dataset_path}")
        return

    df = pd.read_excel(config.full_dataset_path)

    splitting.create_N_folds(
        table   = df,
        cfg     = config,
        n_folds = config.N_FOLDS,
        classes = config.classes
    )
    
    print(f"\n✅ {config.N_FOLDS} Folds created successfully.")


if __name__ == '__main__':
    main()