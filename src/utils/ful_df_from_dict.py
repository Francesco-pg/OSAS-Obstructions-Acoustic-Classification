# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 17:23:21 2025
@author: bianc
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path


def build_osas_dataframe(nan_dict, f0_dict, path_map, known_classes=None):
    """
    Costruisce un DataFrame contenente id, nan_ratio (in %), classe (da quelle disponibili) 
    e array f0, unendo le informazioni dai dizionari e dai path ai file .wav.

    Parameters:
        nan_dict (dict): dizionario con {id: nan_ratio}.
        f0_dict (dict): dizionario con {id: f0_array}.
        path_map (dict): dizionario con {id: Path al file .wav}.
        known_classes (list, optional): Lista di classi conosciute per la classificazione.

    Returns:
        pd.DataFrame: dataframe finale unito con classi e f0_array.
    """
    rows = []

    # Se non vengono passate le classi, estraiamo quelle dai nomi dei file
    if known_classes is None:
        known_classes = set()

    for file_id, nan_ratio in nan_dict.items():
        full_path = path_map.get(file_id)

        if full_path is None:
            print(f"⚠  File {file_id} non trovato nei dati. Saltato.")
            continue

        # Determina la classe in base al nome del file (non l'intero path assoluto)
        file_name = Path(full_path).name.lower()
        found_class = None
        for class_name in known_classes:
            if class_name.lower() in file_name:
                found_class = class_name
                break
        
        if found_class is None:
            # Fix: Safely extract from the file name, handling spaces or underscores
            parts = file_name.split(" ")
            if len(parts) < 2:
                parts = file_name.split("_")
            
            if len(parts) >= 2:
                found_class = parts[1].replace(".wav", "")
            else:
                found_class = "unknown"
                
            known_classes.add(found_class)

        rows.append({
            "id": file_id,
            "nan_ratio": nan_ratio, 
            "class": found_class
        })

    df = pd.DataFrame(rows)

    # Aggiunge f0_array
    f0_df = pd.DataFrame(list(f0_dict.items()), columns=["id", "f0_array"])
    df_full = df.merge(f0_df, on="id", how="left")
    
    return df_full  


def extract_f0_stats(f0_array, winsize=5):
    """
    Extract statistical descriptors and regressions from an F0 array.
    """
    # ---------- helper ----------
    def safe_linregress(x, y):
        """Return slope, intercept or (nan, nan) if regression is invalid"""
        if len(x) < 2 or len(y) < 2:
            return np.nan, np.nan
        if np.all(np.isnan(y)):
            return np.nan, np.nan
        try:
            slope, intercept, _, _, _ = stats.linregress(x, y)
            return slope, intercept
        except Exception:
            return np.nan, np.nan

    # ---------- empty / None ----------
    if f0_array is None or len(f0_array) == 0:
        nan_ratio = 1.0
        return pd.Series({
            "mean_full_f0": np.nan,
            "median_full_f0": np.nan,
            "std_full_f0": np.nan,
            "mean_p15_85_f0": np.nan,
            "median_p15_85_f0": np.nan,
            "std_p15_85_f0": np.nan,
            "slope_full_f0": np.nan,
            "intercept_full_f0": np.nan,
            "windowed_slope": np.nan,
            "windowed_intercept": np.nan,
            "slope_p15_85_f0": np.nan,
            "intercept_p15_85_f0": np.nan,
            "30perc_change_f0": np.nan,
            "nan_ratio": nan_ratio,
            "f0_range_p15_85_f0": np.nan
        })

    f0_array = np.asarray(f0_array, dtype=float)

    # ---------- NaN ratio ----------
    nan_ratio = np.mean(np.isnan(f0_array))

    # ---------- clean ----------
    clean_f0 = f0_array[~np.isnan(f0_array)]

    if clean_f0.size == 0:
        return pd.Series({
            "mean_full_f0": np.nan,
            "median_full_f0": np.nan,
            "std_full_f0": np.nan,
            "mean_p15_85_f0": np.nan,
            "median_p15_85_f0": np.nan,
            "std_p15_85_f0": np.nan,
            "slope_full_f0": np.nan,
            "intercept_full_f0": np.nan,
            "windowed_slope": np.nan,
            "windowed_intercept": np.nan,
            "slope_p15_85_f0": np.nan,
            "intercept_p15_85_f0": np.nan,
            "30perc_change_f0": np.nan,
            "nan_ratio": nan_ratio,
            "f0_range_p15_85_f0": np.nan
        })

    # ---------- percentiles ----------
    if clean_f0.size >= 2:
        p15, p85 = np.percentile(clean_f0, [15, 85])
        f0_15_85 = clean_f0[(clean_f0 >= p15) & (clean_f0 <= p85)]
    else:
        f0_15_85 = np.array([])

    # ---------- range ----------
    if f0_15_85.size >= 1:
        f0_range_15_85 = np.nanmax(f0_15_85) - np.nanmin(f0_15_85)
    else:
        f0_range_15_85 = np.nan

    # ---------- full regression ----------
    time = np.arange(len(clean_f0))
    slope, intercept = safe_linregress(time, clean_f0)

    # ---------- windowed ----------
    windows = [clean_f0[i:i+winsize] for i in range(0, len(clean_f0), winsize)]
    winmeans, x_win, thrs = [], [], []

    for idx, win in enumerate(windows):
        if win.size == 0:
            continue

        winmean = np.mean(win)
        winmeans.append(winmean)
        thrs.append(winmean * 0.30)

        start = idx * winsize
        end = start + len(win)
        x_win.append(np.mean(time[start:end]))

    # ---------- 30% steps ----------
    steps = []
    for i in range(1, len(winmeans)):
        steps.append(int(winmeans[i] > winmeans[i-1] + thrs[i-1]))

    n_steps = np.sum(steps) if len(steps) > 0 else np.nan

    # ---------- window regression ----------
    robust_slope, robust_intercept = safe_linregress(
        np.asarray(x_win), np.asarray(winmeans)
    )

    # ---------- p15–85 regression ----------
    if f0_15_85.size >= 2:
        time_1585 = np.arange(len(f0_15_85))
        slope_1585, intercept_1585 = safe_linregress(time_1585, f0_15_85)
    else:
        slope_1585, intercept_1585 = np.nan, np.nan

    return pd.Series({
        "mean_full_f0": np.mean(clean_f0),
        "median_full_f0": np.median(clean_f0),
        "std_full_f0": np.std(clean_f0),
        "mean_p15_85_f0": np.mean(f0_15_85) if f0_15_85.size > 0 else np.nan,
        "median_p15_85_f0": np.median(f0_15_85) if f0_15_85.size > 0 else np.nan,
        "std_p15_85_f0": np.std(f0_15_85) if f0_15_85.size > 0 else np.nan,
        "slope_full_f0": slope,
        "intercept_full_f0": intercept,
        "windowed_slope": robust_slope,
        "windowed_intercept": robust_intercept,
        "slope_p15_85_f0": slope_1585,
        "intercept_p15_85_f0": intercept_1585,
        "30perc_change_f0": n_steps,
        "nan_ratio": nan_ratio,
        "f0_range_p15_85_f0": f0_range_15_85
    })