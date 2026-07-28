#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Analysis
Module:      Run08_PlotGlobalResults.py
Purpose:     Aggregates global performance metrics and generates final cohort-level plots.
Author:      Francesco Pietrogiacomi
Created:     2026-03-02
"""

# Third-party imports
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Custom imports
import config as cfg
import model_config as m_cfg
from training import model_pipeline
from plotting.global_plots import (
    save_global_table_png, 
    plot_roc_by_fold, 
    plot_mean_roc, 
    plot_global_confusion_matrix
)

# ==========================================================
# CONTROL PANEL - ADJUST YOUR PLOTS HERE
# ==========================================================
SHOW_GHOST_LINES = False     # True: shows thin grey lines for individual folds on the mean plot
SHOW_SD_SHADE = True         # True: shows the grey standard deviation shaded area
SAVE_INDIVIDUAL_PLOT = True  # True: creates a separate plot with the 5 colored fold curves
# ==========================================================

def main():
    trained_models = ["SVM", "MLP"]
    
    for target_model in trained_models:
        if target_model not in m_cfg.MODELS: 
            continue

        analysis_out = cfg.AnalysisOutdir / target_model

        print(f"\n--- GLOBAL AGGREGATION: {target_model} ---")
        global_dir = analysis_out / "global_results"
        global_dir.mkdir(parents=True, exist_ok=True)

        all_reports = []
        fold_roc_data = [] 
        tprs_interp = []   
        mean_fpr = np.linspace(0, 1, 100)
        summed_cm = np.zeros((2, 2))

        for fold_idx in range(1, cfg.N_FOLDS + 1):
            paths = cfg.get_fold_paths(fold_idx)
            txt_path = paths['mod_output_dir'] / f"results_fold_{fold_idx:02d}_model_{cfg.model_n:02d}_{target_model}.txt"
            model_path = paths['mod_output_dir'] / f"best_model_{cfg.model_n:02d}_{target_model}.pkl"
            
            if not txt_path.exists() or not model_path.exists(): 
                continue

            # 1. Parse TXT report
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            rep = {}
            for cls in ["palato", "epiglottide"]:
                m = re.search(rf'{cls}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)', content)
                if m: 
                    rep[cls] = [float(m[1]), float(m[2]), float(m[3])]
            
            for m_key in ['Balanced Acc', 'ROC AUC']:
                match = re.search(rf'{m_key}:\s+([\d.]+)', content)
                if match: 
                    rep[m_key] = float(match.group(1))
            all_reports.append(rep)

            # 2. Load Model & Extract Test Set Probabilities
            model = joblib.load(model_path)
            df_ts = pd.read_csv(paths['test_features'])
            
            # Robust preparation mirroring previous pipeline scripts
            df_ts = model_pipeline.prepare_split(df_ts)
            
            try:
                feats = list(model.named_steps['imputer'].feature_names_in_)
            except AttributeError:
                feats = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else \
                        list(model.named_steps['clf'].feature_names_in_)
            
            X_ts = df_ts[feats].copy()
            y_ts = df_ts['y'].astype(int).values
            
            proba = model.predict_proba(X_ts)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_ts)
            preds = model.predict(X_ts)
            summed_cm += confusion_matrix(y_ts, preds, labels=[0, 1])

            # 3. Calculate ROC Metrics
            fpr, tpr, _ = roc_curve(y_ts, proba)
            fold_auc = auc(fpr, tpr)
            fold_roc_data.append((fpr, tpr, fold_auc))

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs_interp.append(interp_tpr)

        if not all_reports:
            print(f"  ! No valid data found for {target_model}. Skipping...")
            continue

        # --- 1. GLOBAL TABLE ---
        final_rows = []
        for cls in ["palato", "epiglottide"]:
            row = [cls.capitalize()]
            for i in range(3):
                vals = [r[cls][i] for r in all_reports if cls in r]
                row.append(f"{np.mean(vals):.3f} ± {np.std(vals):.3f}")
            final_rows.append(row)
            
        final_rows.append(["-"*10] * 4)
        
        for glob in ['Balanced Acc', 'ROC AUC']:
            vals = [r[glob] for r in all_reports if glob in r]
            if vals: 
                final_rows.append([glob, "-", "-", f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"])
                
        save_global_table_png(
            pd.DataFrame(final_rows, columns=['Metric', 'Precision', 'Recall', 'F1-Score']), 
            f"Final Metrics: {target_model}", 
            global_dir / "table_global_metrics.png"
        )

        # --- 2. PLOTS ---
        if SAVE_INDIVIDUAL_PLOT:
            plot_roc_by_fold(fold_roc_data, target_model, global_dir / "roc_individual_folds.png")

        plot_mean_roc(
            mean_fpr, tprs_interp, fold_roc_data, target_model, 
            global_dir / "roc_mean_final.png", 
            show_ghost_lines=SHOW_GHOST_LINES, 
            show_sd_shade=SHOW_SD_SHADE
        )

        plot_global_confusion_matrix(
            summed_cm, m_cfg.TARGET_NAMES, target_model, 
            global_dir / "global_confusion_matrix.png"
        )

    print("\n✅ Global aggregation complete.")

if __name__ == "__main__":
    main()