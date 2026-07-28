# -*- coding: utf-8 -*-
import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
 
def compute_rms_librosa(y):
    """
    Calcola RMS globale usando librosa.
    """
    rms = librosa.feature.rms(y=y)[0]
    return np.mean(rms)

def normalize_rms_wav_librosa(root_folder, output_folder_name='normalized_wav', target='mean'):
    """
    Memory-efficient RMS normalization. 
    Calculates the target RMS in pass 1, then normalizes in pass 2.
    """
    rms_values = {}
    all_files_to_process = []

    # Gather all file paths
    for root, _, files in os.walk(root_folder):
        rel_root = os.path.relpath(root, root_folder)
        if output_folder_name in rel_root.split(os.sep):
            continue
        for file in files:
            if file.lower().endswith('.wav'):
                all_files_to_process.append(os.path.join(root, file))

    if not all_files_to_process:
        print("⚠️ Nessun file .wav trovato.")
        return

    # --- STEP 1: Calculate RMS (Only store the float value, not the audio) ---
    for full_path in tqdm(all_files_to_process, desc="Pass 1/2: Calculating RMS"):
        try:
            # We use librosa.load with sr=None to keep original sample rate
            y, _ = librosa.load(full_path, sr=None)
            rms_values[full_path] = compute_rms_librosa(y)
            # 'y' gets garbage collected here when we loop to the next file
        except Exception as e:
            print(f"Error loading {full_path}: {e}")

    # Calculate global average target
    if target == 'min':
        target_rms = min(rms_values.values())
    elif target == 'max':
        target_rms = max(rms_values.values())
    elif target == 'mean':
        target_rms = np.mean(list(rms_values.values()))
    elif target == 'median':
        target_rms = np.median(list(rms_values.values()))
    
    print(f"📈 Target RMS set to: {target_rms:.4f}")

    # Prepare output directory
    # We place the output folder alongside the root_folder (parent dir)
    parent_folder = os.path.dirname(os.path.abspath(root_folder))
    output_root = os.path.join(parent_folder, output_folder_name)

    # --- STEP 2: Normalize and Save (One by one) ---
    for filepath in tqdm(all_files_to_process, desc="Pass 2/2: Normalizing"):
        if filepath not in rms_values: 
            continue
        
        y, sr = librosa.load(filepath, sr=None)
        current_rms = rms_values[filepath]
        
        # Calculate scale factor
        scale = target_rms / current_rms if current_rms > 0 else 0
        y_norm = y * scale

        # Peak Limiting to prevent clipping
        peak = np.max(np.abs(y_norm))
        if peak > 1.0:
            y_norm = y_norm / peak

        # Create subfolder structure and save
        relative_path = os.path.relpath(filepath, root_folder)
        output_path = os.path.join(output_root, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        sf.write(output_path, y_norm, sr)

    print(f"✅ Success! Files saved to: {output_root}")