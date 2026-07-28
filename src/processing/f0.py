# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 11:26:23 2025
@author: Francesco
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed
from typing import Dict, Tuple


def _process_one_file(path: str, target_sr: int = 11025) -> Tuple[str, float, np.ndarray]:
    """
    Worker function: process a single audio file for fundamental frequency (F0).
    """
    sname = os.path.splitext(os.path.basename(path))[0]

    try:
        # Resample ONLY for f0 extraction
        y, sr = librosa.load(path, sr=target_sr)

        f0, _, _ = librosa.pyin(
            y,
            sr=sr,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            frame_length=1024,
            hop_length=512,
            fill_na=np.nan,
        ) 

        nan_ratio = np.mean(np.isnan(f0))
        return sname, nan_ratio, f0

    except Exception as e:
        print(f"❌ Error processing {sname}: {e}")
        return sname, np.nan, np.array([])


def extract_f0_with_dicts(datapath: str,
                          ext: str,
                          savetable: bool,
                          outname: str,
                          outpath: str
                          ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Extracts the fundamental frequency (f0) from a list of audio files
    using parallel processing to dramatically speed up librosa.pyin.
    """
    search_path = os.path.join(datapath, "**", f"*.{ext}")
    samplepaths = glob.glob(search_path, recursive=True)

    if len(samplepaths) == 0:
        raise RuntimeError("No audio files found.")

    # ---- PARALLEL EXECUTION ----
    results = Parallel(
        n_jobs=-1,          # use all available cores
        backend="loky",     # process-based (safe with librosa)
        verbose=10
    )(
        delayed(_process_one_file)(path)
        for path in samplepaths
    )

    # ---- REBUILD DICTS ----
    nan_ratio_dict = {k: v for k, v, _ in results}
    f0_array_dict  = {k: f for k, _, f in results}

    # ---- SAVE OUTPUTS ----
    if savetable:
        os.makedirs(outpath, exist_ok=True)

        nan_df = pd.DataFrame(
            nan_ratio_dict.items(),
            columns=["id", "nan_ratio"]
        )
        output_nan_path = os.path.join(outpath, f"{outname}_nan_ratios.xlsx")
        nan_df.to_excel(output_nan_path, index=False)

        output_dict_path = os.path.join(outpath, f"{outname}_f0_arrays.pkl")
        with open(output_dict_path, "wb") as f:
            pickle.dump(f0_array_dict, f)

        print(f"\n✅ Results saved to: {outpath}")

    return nan_ratio_dict, f0_array_dict