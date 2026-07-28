#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Analysis
Module:      Run09_PlotHyperparameters.py
Purpose:     Aggregates and summarizes the best hyperparameters selected across all folds for each model.
Author:      Francesco Pietrogiacomi
Created:     2026-03-02
"""

import pandas as pd
import re
import ast

import config as cfg
import model_config as m_cfg
from plotting.hyperparameter_plots import (
    save_hyperparameter_grid_table, 
    save_hyperparameter_summary_table
)


def main():
    """
    Parses output text reports from all folds to extract the chosen hyperparameters 
    and generates both a grid view and a frequency summary table.
    """
    trained_models = ["SVM", "MLP"]
    
    for target_model in trained_models:
        if target_model not in m_cfg.MODELS:
            continue

        analysis_out = cfg.AnalysisOutdir / target_model
        if not analysis_out.exists():
            continue

        print(f"\n--- HYPERPARAMETER ANALYSIS: {target_model} ---")
        
        # --- Setup Directories ---
        hyperparam_dir = analysis_out / "hyperparameter_analysis"
        hyperparam_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Collect Best Params from all Folds ---
        all_best_params = []
        for fold_idx in range(1, cfg.N_FOLDS + 1):
            paths = cfg.get_fold_paths(fold_idx)
            txt_path = paths['mod_output_dir'] / f"results_fold_{fold_idx:02d}_model_{cfg.model_n:02d}_{target_model}.txt"
            
            if not txt_path.exists():
                print(f"  -> Warning: Results file not found for fold {fold_idx}. Skipping.")
                continue

            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            match = re.search(r"Best Params: ({.*?})", content, re.DOTALL)
            if match:
                param_str = match.group(1)
                try:
                    params_dict = ast.literal_eval(param_str)
                    
                    # Remove the class_weight parameter as it is mostly static/structural
                    params_dict.pop("clf__class_weight", None)
                    
                    # Strip 'clf__' prefix for cleaner table display
                    cleaned_params = {k.replace('clf__', ''): v for k, v in params_dict.items()}
                    all_best_params.append(cleaned_params)
                except (ValueError, SyntaxError):
                    print(f"  -> Error parsing params from fold {fold_idx}.")
                    continue
        
        if not all_best_params:
            print(f"  -> No hyperparameter data found for {target_model}. Skipping analysis.")
            continue
            
        # --- 2. Create Grid View (Params as rows, Folds as columns) ---
        df_grid = pd.DataFrame(all_best_params).transpose()
        df_grid.columns = [f"Fold {i+1}" for i in range(len(df_grid.columns))]
        df_grid.index.name = "Hyperparameter"
        
        grid_png_path = hyperparam_dir / f"hyperparams_grid_{target_model}.png"
        save_hyperparameter_grid_table(df_grid, f"Best Hyperparameters per Fold: {target_model}", grid_png_path)
        print(f"  -> Saved hyperparameter grid to {grid_png_path.name}")

        # --- 3. Create Frequency Summary Table ---
        summary_data = []
        param_df = pd.DataFrame(all_best_params)
        
        for param_name in param_df.columns:
            counts = param_df[param_name].value_counts().reset_index()
            counts.columns = ['Value', 'Count']
            counts['Parameter'] = param_name
            summary_data.append(counts)
            
        if summary_data:
            df_summary = pd.concat(summary_data, ignore_index=True)[['Parameter', 'Value', 'Count']]
            df_summary = df_summary.sort_values(by=['Parameter', 'Count'], ascending=[True, False])
            
            summary_png_path = hyperparam_dir / f"hyperparams_summary_{target_model}.png"
            save_hyperparameter_summary_table(df_summary, f"Hyperparameter Selection Frequency: {target_model}", summary_png_path)
            print(f"  -> Saved hyperparameter summary to {summary_png_path.name}")

        print(f"✅ Hyperparameter analysis for {target_model} complete.")


if __name__ == "__main__":
    main()