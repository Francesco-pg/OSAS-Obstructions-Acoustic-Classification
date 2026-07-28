# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 20:00:58 2025 
@author: bianc
"""

import os
import librosa
import soundfile as sf
from tqdm import tqdm

def nuovo_campionamento(target_path: str, 
                        new_folder_path: str,
                        freq_normal: int,
                        freq_resampled: int):
    """
    Resamples .wav files from a normal frequency to a target frequency.
    Uses os.walk to safely capture all .wav files regardless of folder depth.
    """
    files_to_process = []
    
    # Fix: Replaced os.listdir with os.walk to ensure files at the root 
    # or nested deeper are not skipped.
    for root, _, files in os.walk(target_path):
        for file in files:
            if file.lower().endswith(".wav"):
                files_to_process.append((os.path.join(root, file), file))

    os.makedirs(new_folder_path, exist_ok=True)

    for file_path, file_name in tqdm(files_to_process, desc="Resampling files"):
        try:
            data, sample_rate = librosa.load(file_path, sr=freq_normal)
            resampled_data = librosa.resample(data, orig_sr=freq_normal, target_sr=freq_resampled)
        except Exception as e:
            print(f"❌ Errore nel file {file_path}: {e}")
            continue
 
        outpath = os.path.join(new_folder_path, file_name)

        # Salva
        sf.write(outpath, resampled_data, freq_resampled)