# -*- coding: utf-8 -*-
"""
Module:      get_features.py
Purpose:     Retrieves ranked feature scores for individual folds.
"""

import pandas as pd


def get_ranked_features_with_scores(fold_idx, paths):
    """
    Loads saved RFE feature scores for a given fold and returns a list of tuples: (FeatureName, Rank, Score).
    """
    score_path = paths['mod_output_dir'] / f"rfe_scores_fold_{fold_idx:02d}.csv"
    
    if score_path.exists():
        df = pd.read_csv(score_path)
        return [(row['Feature'], i + 1, row['Score']) for i, row in df.iterrows()]
    else:
        print(f"     ⚠️ Warning: Score CSV not found at {score_path.name}")
        return []