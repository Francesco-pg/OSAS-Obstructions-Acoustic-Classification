# -*- coding: utf-8 -*-
"""
Module:      model_pipeline.py
Purpose:     Pipeline utilities for model training, feature selection, and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight


def calculate_information_gain(X, y):
    """
    Calculates mutual information gain for features relative to the target vector.
    """
    print("  -> Calculating Information Gain (Mutual Info)...")
    if X.isnull().values.any():
        imputer = SimpleImputer(strategy='median')
        X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    else:
        X_clean = X
    mi = mutual_info_classif(X_clean, y, random_state=42)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def remove_correlated_features(X, y, threshold=0.80, importance_scores=None, output_dir=None, model_id="00"):
    """
    Removes highly correlated features using an importance score as a tie-breaker.
    """
    corr_matrix = X.corr().abs()
    
    if output_dir:
        filename = f"correlation_matrix_model_{model_id}.csv"
        corr_matrix.to_csv(output_dir / filename)

    if importance_scores is None:
        print("Warning: Compute importance score of each feature.")

    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        col_i = corr_matrix.columns[i]
        if col_i in to_drop: 
            continue
        for j in range(i+1, len(corr_matrix.columns)):
            col_j = corr_matrix.columns[j]
            if col_j in to_drop: 
                continue
            
            if corr_matrix.iloc[i, j] > threshold:
                if importance_scores.get(col_i, 0) < importance_scores.get(col_j, 0):
                    to_drop.add(col_i)
                else:
                    to_drop.add(col_j)

    X_reduced = X.drop(columns=to_drop)
    print(f"  -> Dropped {len(to_drop)} redundant features. Remaining: {X_reduced.shape[1]}")
    return X_reduced, list(to_drop)


def prepare_split(df):
    """
    Standardizes column naming conventions and maps target classes to integers.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    if 'y' in df.columns: 
        pass
    elif 'class' in df.columns: 
        df = df.rename(columns={'class': 'y'})
    elif 'category' in df.columns: 
        df = df.rename(columns={'category': 'y'})
    elif 'target' in df.columns: 
        df = df.rename(columns={'target': 'y'})
    else:
        print(f"❌ ERROR: Available columns: {list(df.columns)}")
        raise KeyError("Target column (class/category/y) not found.")

    if 'subject_id' in df.columns: 
        pass
    elif 'sub' in df.columns: 
        df = df.rename(columns={'sub': 'subject_id'})
    elif 'id' in df.columns: 
        pass

    # Map class names to integers whenever y is not already numeric. Using
    # is_numeric_dtype (rather than == 'object') also covers pandas' string and
    # categorical dtypes, so this holds under pandas 2.x and 3.x alike.
    if not pd.api.types.is_numeric_dtype(df['y']):
        df['y'] = df['y'].astype(str).str.lower().str.strip()
        mapping = {"epiglottide": 1, "non epiglottide": 0, "palato": 0, "nonepiglottide": 0}
        df['y'] = df['y'].map(mapping)
        
    df = df.dropna(subset=['y'])
    return df


def get_features_and_target(df_train, df_test, model_config):
    """
    Filters feature columns based on exclusions, suffixes, and family blacklists.
    """
    all_cols = set(df_train.columns)
    
    drop_families = getattr(model_config, "DROP_FAMILIES", [])
    drop_suffixes = getattr(model_config, "DROP_SUFFIXES", []) 
    
    if isinstance(drop_families, str): 
        drop_families = [drop_families]
    if isinstance(drop_suffixes, str): 
        drop_suffixes = [drop_suffixes]
    
    feature_cols = []
    
    for c in all_cols:
        if c in model_config.EXCLUDE_COLS or c == "y":
            continue
            
        should_drop = False
        col_lower = c.lower()

        for suffix in drop_suffixes:
            if col_lower.endswith(suffix.lower()):
                should_drop = True
                break
        
        if not should_drop:
            for family in drop_families:
                if family.lower() in col_lower:
                    should_drop = True
                    break

        if not should_drop:
            feature_cols.append(c)

    feature_cols = sorted([c for c in feature_cols if c in df_test.columns])
    
    X_train = df_train[feature_cols].copy()
    X_test  = df_test[feature_cols].copy()
    
    y_train = df_train["y"].astype(int).values
    y_test  = df_test["y"].astype(int).values
    groups  = df_train["subject_id"].astype(str).values
    
    return X_train, y_train, X_test, y_test, groups, feature_cols


def calculate_weights(df_train, y_train, mode=None):
    """
    Computes balanced sample weights at the subject or class level.
    """
    print(f"  -> Calculating Sample Weights (Mode: {mode})...")
    y_series = pd.Series(y_train, index=df_train.index)
    
    if mode == 'class_level':
        raw_weights = compute_sample_weight(
            class_weight='balanced', 
            y=df_train['subject_id']
        )
    elif mode == 'subject_level':
        combined_key = df_train['subject_id'].astype(str) + "_" + y_series.astype(str)
        raw_weights = compute_sample_weight(
            class_weight='balanced', 
            y=combined_key
        )
    else:
        print("⚠️ Warning: Please provide a valid weights computation mode.")
        raw_weights = np.ones(len(y_train))

    return raw_weights


def evaluate_model(model, X_test, y_test, target_names):
    """
    Evaluates a trained model on the test set and prints performance metrics.
    """
    print("\n--- TEST EVALUATION ---")
    pred = model.predict(X_test)
    
    if hasattr(model, "decision_function"): 
        proba = model.decision_function(X_test)
    else: 
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_test))
        
    bal_acc = balanced_accuracy_score(y_test, pred)
    try:
        roc_auc = roc_auc_score(y_test, proba)
    except Exception:
        # Undefined AUC (e.g. a single class present) -> chance level,
        # consistent with the aggregation in Run06_AggregateResults.
        roc_auc = 0.5
        
    print(classification_report(y_test, pred, target_names=target_names))
    print(f"Balanced Acc: {bal_acc:.4f}")
    print(f"ROC AUC:      {roc_auc:.4f}")
    print("\n--- CONFUSION MATRIX ---")
    print(confusion_matrix(y_test, pred))