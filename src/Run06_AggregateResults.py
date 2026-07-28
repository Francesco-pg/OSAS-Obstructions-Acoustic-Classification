#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Analysis
Module:      Run06_AggregateResults.py
Purpose:     Aggregates performance metrics from all N folds for each selected algorithm.
Author:      Francesco Pietrogiacomi
Created:     2026-02-27
"""

# Third-party imports
import numpy as np
import joblib
import pandas as pd
from collections import Counter
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, classification_report

# Custom imports
import config as cfg
import model_config as m_cfg
from training import model_pipeline


def main():
    """
    Aggregates metrics across all folds, calculates mean/std performance, 
    and generates a global evaluation report with feature stability statistics.
    """
    print(f"--- GLOBAL EVALUATION: {m_cfg.MODEL_ID} ---")
    print(f"Strategy: {m_cfg.WEIGHTING_STRATEGY} | RFE Judge: {m_cfg.RFE_MODEL}")
    
    target_models = ["SVM", "MLP"]
    
    for model_name in target_models:
        if model_name not in m_cfg.MODELS:
            continue
            
        print(f"\n>>> Analyzing Algorithm: {model_name}")

        fold_metrics = {
            "balanced_acc": [], 
            "roc_auc": [], 
            "confusion_matrices": [],
            "reports": []
        }
        all_selected_features = []
        n_feats_per_fold = []
        
        for fold_idx in range(1, cfg.N_FOLDS + 1):
            paths = cfg.get_fold_paths(fold_idx)
            model_filename = f"best_model_{cfg.model_n:02d}_{model_name}.pkl"
            model_path = paths['mod_output_dir'] / model_filename
            
            if not model_path.exists():
                print(f"  ! Fold {fold_idx}: Model file not found at {model_path}")
                continue

            # 1. Load Model & Data
            model = joblib.load(model_path)
            df_train = pd.read_csv(paths['train_features'])
            df_test  = pd.read_csv(paths['test_features'])

            df_train = model_pipeline.prepare_split(df_train)
            df_test  = model_pipeline.prepare_split(df_test)
            
            # Extract full feature matrices using pipeline helper logic
            X_train_full, _, X_test_full, y_test, _, feature_cols = \
                model_pipeline.get_features_and_target(df_train, df_test, m_cfg)
            
            # Fix: Retrieve the exact features the model was trained on via feature_names_in_
            try:
                # Scikit-learn pipelines store the feature names from the input of the fitted imputer/scaler
                feats = list(model.named_steps['imputer'].feature_names_in_)
            except AttributeError:
                try:
                    feats = list(model.feature_names_in_)
                except AttributeError:
                    # Fallback to loading from saved RFE score sheet if available
                    rfe_score_path = paths['mod_output_dir'] / f"rfe_scores_fold_{fold_idx:02d}.csv"
                    if rfe_score_path.exists():
                        rfe_df = pd.read_csv(rfe_score_path)
                        feats = rfe_df['Feature'].tolist()
                    else:
                        raise RuntimeError(f"Could not resolve feature names for Fold {fold_idx}")

            all_selected_features.extend(feats)
            n_feats_per_fold.append(len(feats))

            X_test = X_test_full[feats].copy()
            
            # 2. Predict & Probabilities
            preds = model.predict(X_test)
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, "decision_function"):
                    proba = model.decision_function(X_test)
                else:
                    proba = np.zeros(len(y_test))
            except Exception:
                proba = np.zeros(len(y_test))

            # 3. Metrics Calculation
            fold_metrics["balanced_acc"].append(balanced_accuracy_score(y_test, preds))
            try:
                fold_metrics["roc_auc"].append(roc_auc_score(y_test, proba))
            except Exception:
                fold_metrics["roc_auc"].append(0.5)
                
            fold_metrics["confusion_matrices"].append(confusion_matrix(y_test, preds, labels=[0, 1]))
            fold_metrics["reports"].append(classification_report(y_test, preds, output_dict=True))

        if not fold_metrics["balanced_acc"]:
            continue

        # --- GENERATE FINAL REPORT ---
        report_name = f"GLOBAL_RESULTS_{m_cfg.MODEL_ID}_{model_name}_{m_cfg.RFE_MODEL}_{m_cfg.WEIGHTING_STRATEGY}.txt"
        output_file = cfg.ModelOutdir / report_name
        
        with open(output_file, "w", encoding='utf-8') as f:
            def rprint(text):
                print(text)
                f.write(text + "\n")

            rprint("GLOBAL EVALUATION REPORT")
            rprint(f"Model ID: {m_cfg.MODEL_ID}_{model_name}")
            rprint("=" * 60)
            rprint("METADATA:")
            rprint(f"  RFE Proxy Judge:      {m_cfg.RFE_MODEL}")
            rprint(f"  RFE Step Size:        {m_cfg.RFE_STEP}")
            rprint(f"  Weighting Strategy:   {m_cfg.WEIGHTING_STRATEGY.upper()}")
            rprint(f"  Correlation Filter:   {m_cfg.CORRELATION_THRESHOLD} (Point Biserial Guided)")
            rprint(f"  Target Features:      {m_cfg.N_FEATURES_TO_SELECT}")
            rprint("=" * 60)

            # Averages
            rprint(f"\nBalanced Accuracy: {np.mean(fold_metrics['balanced_acc']):.4f} ± {np.std(fold_metrics['balanced_acc']):.4f}")
            rprint(f"ROC AUC:           {np.mean(fold_metrics['roc_auc']):.4f} ± {np.std(fold_metrics['roc_auc']):.4f}")

            # Confusion Matrix
            rprint("\nAggregated Confusion Matrix (Total Sum):")
            rprint(str(np.sum(fold_metrics["confusion_matrices"], axis=0)))

            # Classification Table
            rprint("\n--- Detailed Report ---")
            rprint(f"{'Class':<15} | {'Precision':<18} | {'Recall':<18} | {'F1-Score':<18}")
            rprint("-" * 75)
            row_map = {'0': m_cfg.TARGET_NAMES[0], '1': m_cfg.TARGET_NAMES[1], 'macro avg': 'Macro Avg'}
            
            for sk_key, disp_name in row_map.items():
                p = [r[sk_key]['precision'] for r in fold_metrics["reports"]]
                r = [r[sk_key]['recall'] for r in fold_metrics["reports"]]
                f1 = [r[sk_key]['f1-score'] for r in fold_metrics["reports"]]
                rprint(f"{disp_name:<15} | {np.mean(p):.2f} ± {np.std(p):.2f}      | {np.mean(r):.2f} ± {np.std(r):.2f}      | {np.mean(f1):.2f} ± {np.std(f1):.2f}")

            # Stability Report
            rprint("\n" + "=" * 60)
            rprint(f"FEATURE STABILITY (Top {m_cfg.N_FEATURES_TO_SELECT})")
            rprint("=" * 60)
            feat_counts = Counter(all_selected_features)
            for feat, count in feat_counts.most_common(30):
                rprint(f"{feat:<50} | {count}/{len(fold_metrics['balanced_acc'])} | {(count/len(fold_metrics['balanced_acc']))*100:.0f}%")

    print("\n✅ All global reports generated.")


if __name__ == "__main__":
    main()