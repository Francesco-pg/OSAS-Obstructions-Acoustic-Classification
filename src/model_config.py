#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Analysis
Module:      model_config.py
Purpose:     Configuration file to centralize all model-related settings, 
             hyperparameters, and definitions.
Author:      Francesco Pietrogiacomi
Created:     2026-02-25
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
       
MODEL_ID     = 1
RANDOM_STATE = 42

# --- Columns to Exclude ---
EXCLUDE_COLS = {
    "Unnamed: 0", "file_path", "id", "subject_id", "class", "class_x",
    "class_y", "f0_array", "_sid", "_sub", "_vid", 
    "nan_ratio.1", "hnr", "nan_ratio",
    "mean_full_f0", "median_full_f0", "std_full_f0", "mean_p15_85_f0",  
    "median_p15_85_f0", "std_p15_85_f0", "slope_full_f0", "intercept_full_f0", 
    "windowed_slope", "windowed_intercept", "slope_p15_85_f0", 
    "intercept_p15_85_f0", "30perc_change_f0", "f0_range_p15_85_f0"
}

DROP_FAMILIES = [] 
DROP_SUFFIXES = []

KEEP_CLASSES  = {"epiglottide", "palato"}
CLASS_MAPPING = {"epiglottide": 1, "palato": 0} 
TARGET_NAMES  = ["palato", "epiglottide"]

# --- FEATURE SELECTION ---
DROP_CORRELATED_FEATURES = True 
CORRELATION_THRESHOLD    = 0.80 
N_FEATURES_TO_SELECT     = 20   
RFE_STEP                 = 0.1              

# --- MULTICOLLINEARITY TIE-BREAKER ---
MULTICOLLINEARITY_METHOD = 'point_biserial'

# --- WEIGHTING STRATEGY ---
WEIGHTING_STRATEGY = 'sample'  # 'class' or 'sample'

# --- RFE MODELS ---
RFE_MODEL  = 'SVM_LINEAR'
RFE_MODELS = {
    "RF": {
        "estimator": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
        "needs_scaling": False
    },
    "SVM_LINEAR": {
        "estimator": SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE),
        "needs_scaling": True
    }
}

# --- INTERNAL CROSS VALIDATION ---
CV_STRATEGY = "standard_kfold" 
SEARCH_TYPE = "grid" 
N_SPLITS    = 5  # Number of CV splits for hyperparameter tuning (Inner loop of Nested CV)
SCORING     = {"Balanced_Accuracy": "balanced_accuracy", "F1_Macro": "f1_macro"}
REFIT       = "Balanced_Accuracy"
N_JOBS_CV   = -1

# --- TRAINING MODELS ---
MODELS = {
    "SVM": {
        "estimator": SVC(probability=True, random_state=RANDOM_STATE),
        "scale": True,
        "params": [
            {
                "clf__kernel": ["rbf"],
                "clf__C": [0.01, 0.1, 1, 10, 50, 100, 150, 200], 
                "clf__gamma": ["scale", 0.01, 0.01, 0.1, 1],
                "clf__class_weight": [None] 
            },
            {
                "clf__kernel": ["linear"],
                "clf__C": [0.1, 1, 10, 35, 50],
                "clf__class_weight": [None]
            },
            {
                "clf__kernel": ["poly"],
                "clf__degree": [2, 3],          
                "clf__C": [0.05, 0.1, 1, 10],
                "clf__gamma": ["scale", "auto"],
                "clf__coef0": [0, 1],            
                "clf__class_weight": [None]
            }
        ]
    },
    "MLP": {
        "estimator": MLPClassifier(
            max_iter=5000, 
            random_state=RANDOM_STATE, 
            early_stopping=True,
            validation_fraction=0.1
        ),
        "scale": True,
        "params": {
            "clf__hidden_layer_sizes": [
                (100,), (150,),
                (100, 50),
                (150, 100, 50),
                (200, 150, 75, 50)
            ],
            "clf__activation": ["tanh", "relu"],
            "clf__alpha": [0.0001, 0.001, 0.01, 0.1],
            "clf__learning_rate_init": [0.0001, 0.001, 0.01],
            "clf__solver": ["adam", "lbfgs"]
        }
    }
}