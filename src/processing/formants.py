# -*- coding: utf-8 -*-
"""
Module for Formant extraction using Linear Predictive Coding (LPC).
"""

import librosa
import numpy as np
import scipy.signal

def extract_formants(audio_path, n_formants=4):
    """
    Extracts the first n_formants (mean and std) from an audio file using LPC.
    
    Args:
        audio_path (Path or str): Path to the .wav file.
        n_formants (int): Number of formants to extract (default 4).
        
    Returns:
        dict: Dictionary containing mean and std for each formant (e.g., 'formant_1_mean').
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Pre-emphasis filter (standard for formant analysis to flatten spectral tilt)
        y = librosa.effects.preemphasis(y, coef=0.97)
        
        # LPC Order Calculation
        # Rule of thumb: 2 poles per kHz + 2 for radiation/glottal effects.
        # e.g., for 22050Hz -> 11kHz Nyquist -> 22 poles + 2 = 24.
        order = int(2 + (sr / 1000))
        
        # Frame settings (standard speech analysis windows)
        frame_length = int(0.025 * sr) # 25ms window
        hop_length = int(0.010 * sr)   # 10ms step
        
        # Frame the signal
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        
        # Window function (Hamming)
        window = np.hamming(frame_length)
        
        formants_over_time = []
        vtl_over_time = []
        
        # Process each frame
        for i in range(frames.shape[1]):
            frame = frames[:, i] * window
            
            # LPC coefficients (Linear Predictive Coding)
            # librosa.lpc returns [1, -a1, -a2, ...]
            a = librosa.lpc(frame, order=order)
            
            # Roots of the polynomial
            roots = np.roots(a)
            
            # Keep roots with positive imaginary part (conjugate pairs)
            roots = roots[np.imag(roots) >= 0]
            
            # Calculate angles and convert to Frequency (Hz)
            angles = np.arctan2(np.imag(roots), np.real(roots))
            freqs = sorted(angles * (sr / (2 * np.pi)))
            
            # Filter frequencies: Formants are usually > 90Hz and < Nyquist
            freqs = [f for f in freqs if f > 90]
            
            # Take the first n_formants
            current_formants = freqs[:n_formants]
            
            # Pad with NaN if fewer formants found in this frame
            if len(current_formants) < n_formants:
                current_formants += [np.nan] * (n_formants - len(current_formants))
                
            formants_over_time.append(current_formants)

            # --- VTL Calculation ---
            # Estimate VTL for this frame based on available formants
            # Formula: L = (2n-1) * c / (4 * Fn)
            c = 35000  # Speed of sound in cm/s
            frame_vtls = []
            for idx, f in enumerate(current_formants):
                if not np.isnan(f) and f > 0:
                    n = idx + 1
                    l = (2 * n - 1) * c / (4 * f)
                    frame_vtls.append(l)
            
            vtl_over_time.append(np.mean(frame_vtls) if frame_vtls else np.nan)
            
        formants_arr = np.array(formants_over_time)
        
        # Calculate statistics (Mean and Std) ignoring NaNs
        features = {}
        for i in range(n_formants):
            f_col = formants_arr[:, i]
            # Filter NaNs
            f_clean = f_col[~np.isnan(f_col)]
            
            if len(f_clean) > 0:
                features[f'formant_{i+1}_mean'] = np.mean(f_clean)
                features[f'formant_{i+1}_std'] = np.std(f_clean)
                features[f'formant_{i+1}_median'] = np.median(f_clean)
                features[f'formant_{i+1}_range'] = np.max(f_clean) - np.min(f_clean)
            else:
                features[f'formant_{i+1}_mean'] = 0
                features[f'formant_{i+1}_std'] = 0
                features[f'formant_{i+1}_median'] = 0
                features[f'formant_{i+1}_range'] = 0
                
        # Calculate VTL statistics
        vtl_arr = np.array(vtl_over_time)
        vtl_clean = vtl_arr[~np.isnan(vtl_arr)]
        
        if len(vtl_clean) > 0:
            features['vtl_mean'] = np.mean(vtl_clean)
            features['vtl_std'] = np.std(vtl_clean)
            features['vtl_median'] = np.median(vtl_clean)
            features['vtl_range'] = np.max(vtl_clean) - np.min(vtl_clean)
        else:
            for stat in ['mean', 'std', 'median', 'range']:
                features[f'vtl_{stat}'] = 0

        return features

    except Exception as e:
        print(f"Error extracting formants for {audio_path}: {e}")
        # Return zeros in case of error to maintain dataframe structure
        feats = {f'formant_{i+1}_{stat}': 0 for i in range(n_formants) for stat in ['mean', 'std', 'median', 'range']}
        feats.update({f'vtl_{stat}': 0 for stat in ['mean', 'std', 'median', 'range']})
        return feats