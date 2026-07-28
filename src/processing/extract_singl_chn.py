# -*- coding: utf-8 -*-
"""
Created on Mon May 19 18:00:39 2025
@author: Francesco
"""

import os
import soundfile as sf 
from tqdm import tqdm

def extract_single_channel(target_path: str, output_path: str, outdir_name: str):
    """
    Extracts the left channel (channel 0) from stereo .wav files and saves 
    them to a new directory while maintaining the original folder structure.
    """
    # Create the full output directory path
    new_folder_path = os.path.join(output_path, outdir_name)
    os.makedirs(new_folder_path, exist_ok=True)

    wav_files = []
    for root, _, files in os.walk(target_path):
        for file in files:
            if file.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, file))

    for file_path in tqdm(wav_files, desc="Extracting single channel"):
        try:
            data, sample_rate = sf.read(file_path)

            # Check if data has multiple channels
            if len(data.shape) == 1:
                # Mono — no changes needed
                left_channel = data
            else:
                # Stereo or more — take left channel (channel 0)
                left_channel = data[:, 0]

        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            continue

        # Build relative path and output path
        rel_path = os.path.relpath(file_path, target_path)
        parts = rel_path.split(os.sep)

        folders_to_keep = os.path.join(*parts[:-1]) if len(parts) > 1 else ""
        file_name = os.path.splitext(parts[-1])[0] + ".wav"

        full_output_dir = os.path.join(new_folder_path, folders_to_keep)
        os.makedirs(full_output_dir, exist_ok=True)
        outpath = os.path.join(full_output_dir, file_name)

        try:
            # Save the left channel only
            sf.write(outpath, left_channel, sample_rate)

        except Exception as e:
            print(f"❌ Error saving {outpath}: {e}")