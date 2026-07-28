#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Analysis
Module:      Run07_EvaluationPlots.py
Purpose:     Generates evaluation plots, confusion matrices, and report summaries per fold.
Author:      Francesco Pietrogiacomi
Created:     2026-02-27
"""

# Third-party imports
import pandas as pd
import numpy as np
import joblib
import re

# Custom imports
import config as cfg
import model_config as m_cfg
from training import model_pipeline 
from plotting.eval_plots import (
    save_table_as_png, 
    save_summary_table_as_png, 
    plot_feature_importance_score, 
    plot_confusion_matrix
)
from utils.get_features import get_ranked_features_with_scores


def main():
    """
    Loops through each model and fold to generate summary tables, 
    feature importance plots, confusion matrices, and report snapshots.
    """
    trained_models = ["SVM", "MLP"]
    
    for target_model in trained_models:
        if target_model not in m_cfg.MODELS: 
            continue

        analysis_root = cfg.ModelOutdir
        analysis_out  = cfg.AnalysisOutdir / target_model
        analysis_root.mkdir(parents=True, exist_ok=True)
        analysis_out.mkdir(parents=True, exist_ok=True)
        print(f"\n--- ANALYZING MODEL: {target_model} ---\n📂 Path: {analysis_root}")

        global_cm = np.zeros((2, 2), dtype=int)

        for fold_idx in range(1, cfg.N_FOLDS + 1):
            paths      = cfg.get_fold_paths(fold_idx)
            model_path = paths['mod_output_dir'] / f"best_model_{cfg.model_n:02d}_{target_model}.pkl"
            txt_path   = paths['mod_output_dir'] / f"results_fold_{fold_idx:02d}_model_{cfg.model_n:02d}_{target_model}.txt"
            
            if not model_path.exists(): 
                continue
                
            print(f"  > Fold {fold_idx:02d}")
            fold_dir = analysis_out / f"out_fold_{fold_idx:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            # 1. Load Data & Model
            model = joblib.load(model_path)
            df_tr = model_pipeline.prepare_split(pd.read_csv(paths['train_features']))
            df_ts = model_pipeline.prepare_split(pd.read_csv(paths['test_features']))
            
            # Detect features from model
            try:
                expected_feats = list(model.named_steps['imputer'].feature_names_in_)
            except AttributeError:
                expected_feats = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else \
                                 list(model.named_steps['clf'].feature_names_in_)

            X_ts = df_ts[expected_feats].copy()
            y_ts = df_ts['y'].astype(int).values

            # 2. Dataset Tables
            for df, name, lbl in [(df_tr, "train", "Train"), (df_ts, "test", "Test")]:
                summary_df = pd.DataFrame({ 
                    'Class': m_cfg.TARGET_NAMES,
                    'Subjects': [df[df['y']==0]['subject_id'].nunique(), df[df['y']==1]['subject_id'].nunique()],
                    'Videos': [len(df[df['y']==0]), len(df[df['y']==1])]
                })
                save_summary_table_as_png(summary_df, f"{lbl} - Fold {fold_idx}", fold_dir / f"summary_{name}.png")

            # 3. Importance Plot
            ranked_feats = get_ranked_features_with_scores(fold_idx, paths)
            plot_feature_importance_score(ranked_feats, f"Top Features - Fold {fold_idx}", fold_dir / "importance.png")

            # 4. CM & ROC
            preds = model.predict(X_ts)
            cm    = plot_confusion_matrix(y_ts, preds, m_cfg.TARGET_NAMES, f"CM Fold {fold_idx}", fold_dir / "cm.png")
            global_cm += cm
            
            # 5. PARSE TXT FOR REPORT
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f: 
                    content = f.read()
                
                rows = []
                
                # A. Find Class-Specific Rows
                class_pattern = re.compile(r'(palato|epiglottide)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)')
                for m in class_pattern.findall(content):
                    rows.append([m[0].capitalize(), m[1], m[2], m[3], m[4]])
                
                rows.append(["-"*10] * 5)

                # B. Find Overall Accuracy
                acc_match = re.search(r'accuracy\s+([\d.]+)\s+(\d+)', content)
                if acc_match:
                    rows.append(["Accuracy", "", "", acc_match.group(1), acc_match.group(2)])

                # C. Find Macro & Weighted Averages
                for avg_type in ["macro avg", "weighted avg"]:
                    avg_match = re.search(fr'{avg_type}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)', content)
                    if avg_match:
                        name = "Macro Avg" if "macro" in avg_type else "Weighted Avg"
                        rows.append([name, avg_match.group(1), avg_match.group(2), avg_match.group(3), avg_match.group(4)])

                # D. Add Final Balanced Accuracy
                rows.append(["="*10] * 5)
                bal_match = re.search(r'Balanced Acc:\s+([\d.]+)', content)
                if bal_match:
                    rows.append(["Balanced Acc.", "", "", f"{round(float(bal_match.group(1)), 2)}", ""])

                # Create and Save DataFrame
                rep_df = pd.DataFrame(rows, columns=['', 'Precision', 'Recall', 'F1-Score', 'Support'])
                save_table_as_png(rep_df, f"Classification Report - Fold {fold_idx}", fold_dir / "classification_report.png")

        print(f"✅ Analysis for {target_model} complete.")


if __name__ == "__main__":
    main()