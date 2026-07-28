#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Analysis
Module:      Run05_TrainTest.py
Purpose:     Train and test models using Nested CV.
Author:      Francesco Pietrogiacomi
Created:     2026-02-27
"""

# Third-party imports
import time
import sys
import os
import joblib
import random
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.feature_selection import RFE
from imblearn.over_sampling import RandomOverSampler

# Custom imports
import config as cfg
import model_config as m_cfg
from training import model_pipeline


class Tee:
    """
    Helper class to duplicate stdout to both console and a log file simultaneously.
    """
    def __init__(self, filename):
        self.filename = filename
        self.file     = None
        self.stdout   = None

    def __enter__(self):
        self.file   = open(self.filename, 'w')
        self.stdout = sys.stdout
        sys.stdout  = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout
        if self.file:
            self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def main():
    """
    Executes the main Nested Cross-Validation training and evaluation loop.
    """
    print(f"--- NESTED CROSS VALIDATION {cfg.N_FOLDS} X {m_cfg.N_SPLITS} PIPELINE ---")
    print(f"RFE Model: {m_cfg.RFE_MODEL} | Strategy: {m_cfg.WEIGHTING_STRATEGY.upper()}")
    
    np.random.seed(m_cfg.RANDOM_STATE)
    random.seed(m_cfg.RANDOM_STATE)
    
    for fold_idx in range(1, cfg.N_FOLDS + 1):
        paths = cfg.get_fold_paths(fold_idx)
        print(f"\n>>> Processing Fold {fold_idx}")
        
        if not os.path.exists(paths['train_features']):
            print(f"❌ Error: {paths['train_features']} not found.")
            continue
            
        # 1. Load Data
        df_train = pd.read_csv(paths['train_features'])
        df_test  = pd.read_csv(paths['test_features'])

        df_train = model_pipeline.prepare_split(df_train)
        df_test  = model_pipeline.prepare_split(df_test)
        
        X_train_full, y_train, X_test_full, y_test, groups, feature_cols = \
            model_pipeline.get_features_and_target(df_train, df_test, m_cfg)
        
        # 2. Compute Sample Weights
        sample_weight = None
        if m_cfg.WEIGHTING_STRATEGY == 'sample':
            sample_weight = model_pipeline.calculate_weights(df_train, y_train, mode='subject_level')
        elif m_cfg.WEIGHTING_STRATEGY == 'class':
            sample_weight = model_pipeline.calculate_weights(df_train, y_train, mode='class_level')

        # --- 3 & 4. DYNAMIC PRUNING (CORRELATION + SINGLE JUDGE) ---
        multicoll_method = getattr(m_cfg, 'MULTICOLLINEARITY_METHOD', 'mutual_info')
        print(f"  -> Resolving Multicollinearity via: {multicoll_method.upper()}")
        
        judge_scores = None
        
        if multicoll_method == 'point_biserial':
            pb_dict = {}
            for col in X_train_full.columns:
                val = X_train_full[col].fillna(X_train_full[col].median())
                corr, _ = pointbiserialr(y_train, val)
                pb_dict[col] = abs(corr)
            judge_scores = pd.Series(pb_dict)
        else:
            judge_scores = model_pipeline.calculate_information_gain(X_train_full, y_train)

        # Pruning Step: Use the selected judge to break ties in correlated pairs
        print(f"  -> Pruning correlated features (> {m_cfg.CORRELATION_THRESHOLD})...")
        X_pruned, dropped = model_pipeline.remove_correlated_features(
            X_train_full, y_train, 
            threshold=m_cfg.CORRELATION_THRESHOLD,
            importance_scores=judge_scores 
        )
        X_test_pruned = X_test_full.drop(columns=dropped)
        print(f"  -> Dropped {len(dropped)} features. {X_pruned.shape[1]} survivors entering RFE.")

        # 5. RFE FEATURE SELECTION
        print(f"  -> Running RFE ({m_cfg.RFE_MODEL}) Step={m_cfg.RFE_STEP}...")
        judge_info = m_cfg.RFE_MODELS[m_cfg.RFE_MODEL]
        
        X_rfe_input = X_pruned.fillna(X_pruned.median())
        if judge_info["needs_scaling"]:
            scaler = StandardScaler()
            X_rfe_input = pd.DataFrame(scaler.fit_transform(X_rfe_input), columns=X_pruned.columns)

        selector = RFE(
            estimator           = judge_info["estimator"], 
            n_features_to_select = m_cfg.N_FEATURES_TO_SELECT, 
            step                 = m_cfg.RFE_STEP
        )
        
        rfe_fit_params = {}
        if m_cfg.WEIGHTING_STRATEGY == 'sample':
            rfe_fit_params["sample_weight"] = sample_weight
        elif m_cfg.WEIGHTING_STRATEGY == 'class':
            rfe_fit_params["class_weight"]  = sample_weight

        selector.fit(X_rfe_input, y_train, **rfe_fit_params)
        top_k_features = X_pruned.columns[selector.support_].tolist()
        print(f"  -> Final {m_cfg.N_FEATURES_TO_SELECT} selected features: {top_k_features}")

        # Save RFE Scores for Analysis
        print("  -> Saving RFE Feature Scores...")
        judge_est = selector.estimator_
        if hasattr(judge_est, "coef_"):
            rfe_scores = np.abs(judge_est.coef_[0])
        else:
            rfe_scores = judge_est.feature_importances_
            
        rfe_results = pd.DataFrame({
            "Feature": X_pruned.columns[selector.support_],
            "Score": rfe_scores
        }).sort_values(by="Score", ascending=False)
 
        rfe_results.to_csv(paths['mod_output_dir'] / f"rfe_scores_fold_{fold_idx:02d}.csv", index=False)
        
        X_train_fold = X_pruned[top_k_features].copy()
        X_test_fold  = X_test_pruned[top_k_features].copy()
        
        # 6. FINAL TRAINING LOOP (SVM, MLP)
        for model_name in ["SVM", "MLP"]:
            print(f"\n  --- Training: {model_name} ---")
            config_dict = m_cfg.MODELS[model_name]
            
            X_train_run = X_train_fold.copy()
            y_train_run = y_train.copy()
            groups_run  = groups.copy()

            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler() if config_dict["scale"] else "passthrough"),
                ("clf", config_dict["estimator"])
            ])
            
            raw_params   = config_dict["params"]
            search_space = [{ (k if k.startswith("clf__") else f"clf__{k}"): v for k, v in grid.items()} 
                            for grid in raw_params] if isinstance(raw_params, list) else \
                           { (k if k.startswith("clf__") else f"clf__{k}"): v for k, v in raw_params.items()}
            
            cv = StratifiedGroupKFold(n_splits=m_cfg.N_SPLITS, shuffle=True, random_state=m_cfg.RANDOM_STATE)
            
            search = GridSearchCV(
                pipe, 
                param_grid=search_space, 
                cv=cv, 
                scoring=m_cfg.SCORING, 
                refit=m_cfg.REFIT, 
                n_jobs=-1, 
                verbose=1
            )
            
            fit_kwargs = {} 
            if m_cfg.WEIGHTING_STRATEGY == 'sample':
                if model_name in ["SVM", "ADA"]:
                    fit_kwargs["clf__sample_weight"] = sample_weight 
                elif model_name == "MLP":
                    X_train_run['temp_groups'] = groups_run
                    ros = RandomOverSampler(random_state=m_cfg.RANDOM_STATE)
                    X_train_run, y_train_run = ros.fit_resample(X_train_run, y_train_run)
                    groups_run = X_train_run.pop('temp_groups').values
            
            elif m_cfg.WEIGHTING_STRATEGY == 'class' and model_name == "MLP":
                X_train_run['temp_groups'] = groups_run
                ros = RandomOverSampler(random_state=m_cfg.RANDOM_STATE)
                X_train_run, y_train_run = ros.fit_resample(X_train_run, y_train_run)
                groups_run = X_train_run.pop('temp_groups').values

            # Fit Model
            start_time = time.time()
            try:
                search.fit(X_train_run, y_train_run, groups=groups_run, **fit_kwargs)
            except Exception as e:
                print(f"    ! Training failed for {model_name}: {e}")
                continue
                
            duration        = time.time() - start_time
            best_model      = search.best_estimator_
            model_filename  = f"best_model_{cfg.model_n:02d}_{model_name}.pkl"
            result_filename = f"results_fold_{fold_idx:02d}_model_{cfg.model_n:02d}_{model_name}.txt"
            joblib.dump(best_model, paths['mod_output_dir'] / model_filename)
            
            with Tee(paths['mod_output_dir'] / result_filename):
                print(f"Model: {model_name} | Strategy: {m_cfg.WEIGHTING_STRATEGY}")
                print(f"RFE Judge: {m_cfg.RFE_MODEL} | Selection Step: {m_cfg.RFE_STEP}")
                print(f"Time: {duration:.2f}s | Best Params: {search.best_params_}")
                print(f"Final Features: {top_k_features}")
                model_pipeline.evaluate_model(best_model, X_test_fold, y_test, m_cfg.TARGET_NAMES)


if __name__ == "__main__":
    main()