# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:01:12 2025
@author: Francesco 
"""

import os
import numpy as np
import pandas as pd
import librosa
import scipy.signal
import parselmouth 
from parselmouth.praat import call
import config  # Import config to read the FORMANT_METHOD toggle


def summarize_feature(feature_matrix, name, labels=None):
    """Summarize a 2D feature matrix into a dictionary of summary stats."""
    mean   = np.mean(feature_matrix, axis=1)
    std    = np.std(feature_matrix, axis=1)
    median = np.median(feature_matrix, axis=1)
    range_ = np.max(feature_matrix, axis=1) - np.min(feature_matrix, axis=1)

    summary = {}
    for i in range(feature_matrix.shape[0]):
        if labels is not None and i < len(labels):
            suffix = labels[i]
        else:
            suffix = str(i+1)
        
        summary[f"{name}_{suffix}_mean"]   = mean[i]
        summary[f"{name}_{suffix}_std"]    = std[i]
        summary[f"{name}_{suffix}_median"] = median[i]
        summary[f"{name}_{suffix}_range"]  = range_[i]
    return summary


def summarize_vector(vector, name):
    """Summarize a 1D vector into a dictionary of summary stats."""
    return {
        f"{name}_mean": np.mean(vector),
        f"{name}_std": np.std(vector),
        f"{name}_median": np.median(vector),
        f"{name}_range": np.max(vector) - np.min(vector)
    }


def extract_spect_features(audio_path, y=None, sr=None):
    """Extracts summarized audio features and returns a pandas DataFrame."""
    sname = os.path.splitext(os.path.basename(audio_path))[0]

    # Params
    w = 5
    hop_length = 512
    n_fft      = 1024

    # Load audio
    if y is None or sr is None:
        y, sr = librosa.load(audio_path)
    
    ### MFCC
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=20, hop_length=hop_length, n_fft=n_fft
    )
    mfccs_delta = librosa.feature.delta(mfccs, width=w)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2, width=w)

    ### Spectral Rolloff 50% and 70%
    rolloff_50 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.5, hop_length=hop_length, n_fft=n_fft)
    rolloff_70 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.7, hop_length=hop_length, n_fft=n_fft)
    rolloff_90 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.9, hop_length=hop_length, n_fft=n_fft)

    ### Delta 1
    rolloff_50_delta = librosa.feature.delta(rolloff_50, width=w)
    rolloff_70_delta = librosa.feature.delta(rolloff_70, width=w)
    rolloff_90_delta = librosa.feature.delta(rolloff_90, width=w)

    ### Delta 2
    rolloff_50_delta2 = librosa.feature.delta(rolloff_50, width=w, order=2)
    rolloff_70_delta2 = librosa.feature.delta(rolloff_70, width=w, order=2)
    rolloff_90_delta2 = librosa.feature.delta(rolloff_90, width=w, order=2)

    ### Spectral Flatness
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length, n_fft=n_fft)
    flatness_delta = librosa.feature.delta(flatness, width=w)
    flatness_delta2 = librosa.feature.delta(flatness, order=2, width=w)
    
    ### Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    zcr_delta = librosa.feature.delta(zcr, width=w)
    zcr_delta2 = librosa.feature.delta(zcr, order=2, width=w)

    ### Spectral Contrast
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, fmin=100
    )
    contrast_delta = librosa.feature.delta(contrast, width=w)
    contrast_delta2 = librosa.feature.delta(contrast, order=2, width=w)
    
    ### Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    centroid_delta = librosa.feature.delta(centroid, width=w)
    centroid_delta2 = librosa.feature.delta(centroid, order=2, width=w)

    ### Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    bandwidth_delta = librosa.feature.delta(bandwidth, width=w)
    bandwidth_delta2 = librosa.feature.delta(bandwidth, order=2, width=w)
    
    ### Chroma Features
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
    chroma_delta = librosa.feature.delta(chroma, width=w)
    chroma_delta2 = librosa.feature.delta(chroma, order=2, width=w)
    
    ### Mel Spectrogram
    mel_spec    = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    ### Divide Mel Spectrogram into custom frequency bands
    mel_frequencies  = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr/2)
    
    sub_band_mask    = (mel_frequencies >= 20) & (mel_frequencies < 100)
    low_band_mask    = (mel_frequencies >= 100) & (mel_frequencies < 300)
    medlow_band_mask = (mel_frequencies >= 300) & (mel_frequencies < 1000)
    med_mask         = (mel_frequencies >= 1000) & (mel_frequencies < 2500)
    medhi_band_mask  = (mel_frequencies >= 2500) & (mel_frequencies < 5000)
    hi_band_mask     = (mel_frequencies >= 5000)

    sub_band_energy    = mel_spec_db[sub_band_mask, :].mean(axis=0)
    low_band_energy    = mel_spec_db[low_band_mask, :].mean(axis=0)
    medlow_band_energy = mel_spec_db[medlow_band_mask, :].mean(axis=0)
    med_energy         = mel_spec_db[med_mask, :].mean(axis=0)
    medhi_energy       = mel_spec_db[medhi_band_mask, :].mean(axis=0)
    hi_energy          = mel_spec_db[hi_band_mask, :].mean(axis=0)

    ### Collect summaries in a dictionary
    features = {}
    features["id"] = sname
    
    features.update(summarize_feature(mfccs, "mfcc"))
    features.update(summarize_feature(mfccs_delta, "mfcc_delta"))
    features.update(summarize_feature(mfccs_delta2, "mfcc_delta2"))
    
    features.update(summarize_feature(rolloff_50, "rolloff_50"))
    features.update(summarize_feature(rolloff_70, "rolloff_70"))
    features.update(summarize_feature(rolloff_90, "rolloff_90"))
    
    features.update(summarize_feature(rolloff_50_delta, "rolloff_50_delta"))
    features.update(summarize_feature(rolloff_70_delta, "rolloff_70_delta"))
    features.update(summarize_feature(rolloff_90_delta, "rolloff_90_delta"))

    features.update(summarize_feature(rolloff_50_delta2, "rolloff_50_delta2"))
    features.update(summarize_feature(rolloff_70_delta2, "rolloff_70_delta2"))
    features.update(summarize_feature(rolloff_90_delta2, "rolloff_90_delta2"))

    features.update(summarize_feature(flatness, "flatness"))
    features.update(summarize_vector(flatness_delta, "flatness_delta"))
    features.update(summarize_vector(flatness_delta2, "flatness_delta2"))

    features.update(summarize_feature(zcr, "zcr"))
    features.update(summarize_feature(zcr_delta, "zcr_delta"))
    features.update(summarize_feature(zcr_delta2, "zcr_delta2"))
    
    features.update(summarize_feature(contrast, "contrast"))
    features.update(summarize_feature(contrast_delta, "contrast_delta"))
    features.update(summarize_feature(contrast_delta2, "contrast_delta2"))

    features.update(summarize_feature(centroid, "spectral_centroid"))
    features.update(summarize_feature(centroid_delta, "spectral_centroid_delta"))
    features.update(summarize_feature(centroid_delta2, "spectral_centroid_delta2"))
    
    features.update(summarize_feature(bandwidth, "spectral_bandwidth"))
    features.update(summarize_feature(bandwidth_delta, "spectral_bandwidth_delta"))
    features.update(summarize_feature(bandwidth_delta2, "spectral_bandwidth_delta2"))

    # Use note names for chroma features
    chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    features.update(summarize_feature(chroma, "chroma", labels=chroma_labels))
    features.update(summarize_feature(chroma_delta, "chroma_delta", labels=chroma_labels))
    features.update(summarize_feature(chroma_delta2, "chroma_delta2", labels=chroma_labels))

    features.update(summarize_vector(sub_band_energy, "sub_band_energy"))
    features.update(summarize_vector(low_band_energy, "low_band_energy"))
    features.update(summarize_vector(medlow_band_energy, "medlow_band_energy"))
    features.update(summarize_vector(med_energy, "med_energy"))
    features.update(summarize_vector(medhi_energy, "medhi_energy"))
    features.update(summarize_vector(hi_energy, "hi_energy"))

    df = pd.Series(features)
    return df


def _compute_formants(y, sr, n_formants=4):
    """
    Extracts formants and VTL statistics using either custom LPC or Praat (Parselmouth).
    Controlled by config.FORMANT_METHOD ('lpc' or 'praat').
    """
    method = config.FORMANT_METHOD  
    features = {}
    
    # === METHOD 1: PRAAT (Gold Standard) ===
    if method == 'praat':
        try:
            sound = parselmouth.Sound(y, sampling_frequency=sr)
            
            formant_obj = sound.to_formant_burg(
                time_step=0.010, 
                max_number_of_formants=5, 
                maximum_formant=5500, 
                window_length=0.025, 
                pre_emphasis_from=50
            )
            
            formants_over_time = []
            vtl_over_time = []
            
            n_frames = formant_obj.get_number_of_frames()
            for i in range(1, n_frames + 1):
                t = formant_obj.get_time_from_frame_number(i)
                current_formants = []
                
                for f_idx in range(1, n_formants + 1):
                    freq = formant_obj.get_value_at_time(f_idx, t)
                    current_formants.append(freq if not np.isnan(freq) else np.nan)
                
                formants_over_time.append(current_formants)
                
                c = 35000 
                frame_vtls = []
                for idx, f in enumerate(current_formants):
                    if not np.isnan(f) and f > 0:
                        n = idx + 1
                        l = (2 * n - 1) * c / (4 * f)
                        frame_vtls.append(l)
                vtl_over_time.append(np.mean(frame_vtls) if frame_vtls else np.nan)

            formants_arr = np.array(formants_over_time)
            vtl_arr = np.array(vtl_over_time)

        except Exception as e:
            print(f"Error in Praat extraction: {e}")
            return {f'formant_{i+1}_{stat}': 0 for i in range(n_formants) for stat in ['mean', 'std', 'median', 'range']}

    # === METHOD 2: CUSTOM LPC (Original) ===
    elif method == 'lpc':
        try:
            y_pre = librosa.effects.preemphasis(y, coef=0.97)
            
            order = int(2 + (sr / 1000))
            frame_length = int(0.025 * sr)
            hop_length = int(0.010 * sr)
            
            frames = librosa.util.frame(y_pre, frame_length=frame_length, hop_length=hop_length)
            window = np.hamming(frame_length)
            
            formants_over_time = []
            vtl_over_time = []
            
            for i in range(frames.shape[1]):
                frame = frames[:, i] * window
                a = librosa.lpc(frame, order=order)
                roots = np.roots(a)
                roots = roots[np.imag(roots) >= 0]
                
                angles = np.arctan2(np.imag(roots), np.real(roots))
                freqs = sorted(angles * (sr / (2 * np.pi)))
                freqs = [f for f in freqs if f > 90] 
                
                current_formants = freqs[:n_formants]
                if len(current_formants) < n_formants:
                    current_formants += [np.nan] * (n_formants - len(current_formants))
                formants_over_time.append(current_formants)
                
                c = 35000
                frame_vtls = []
                for idx, f in enumerate(current_formants):
                    if not np.isnan(f) and f > 0:
                        n = idx + 1
                        l = (2 * n - 1) * c / (4 * f)
                        frame_vtls.append(l)
                vtl_over_time.append(np.mean(frame_vtls) if frame_vtls else np.nan)
                
            formants_arr = np.array(formants_over_time)
            vtl_arr = np.array(vtl_over_time)
            
        except Exception as e:
            print(f"Error in LPC extraction: {e}")
            return {f'formant_{i+1}_{stat}': 0 for i in range(n_formants) for stat in ['mean', 'std', 'median', 'range']}
    
    else:
        print(f"Unknown formant method: {method}. Defaulting to zeros.")
        return {f'formant_{i+1}_{stat}': 0 for i in range(n_formants) for stat in ['mean', 'std', 'median', 'range']}

    # === COMMON STATISTICS CALCULATION ===
    for i in range(n_formants):
        if 'formants_arr' in locals() and formants_arr.ndim > 1 and formants_arr.shape[1] > i:
            f_col = formants_arr[:, i]
            f_clean = f_col[~np.isnan(f_col)]
        else:
            f_clean = []

        if len(f_clean) > 0:
            features[f'formant_{i+1}_mean'] = np.mean(f_clean)
            features[f'formant_{i+1}_std'] = np.std(f_clean)
            features[f'formant_{i+1}_median'] = np.median(f_clean)
            features[f'formant_{i+1}_range'] = np.max(f_clean) - np.min(f_clean)
        else:
            for stat in ['mean', 'std', 'median', 'range']: 
                features[f'formant_{i+1}_{stat}'] = 0
            
    if 'vtl_arr' in locals():
        vtl_clean = vtl_arr[~np.isnan(vtl_arr)]
        if len(vtl_clean) > 0:
            features['vtl_mean'] = np.mean(vtl_clean)
            features['vtl_std'] = np.std(vtl_clean)
            features['vtl_median'] = np.median(vtl_clean)
            features['vtl_range'] = np.max(vtl_clean) - np.min(vtl_clean)
        else:
            for stat in ['mean', 'std', 'median', 'range']: 
                features[f'vtl_{stat}'] = 0
    else:
        for stat in ['mean', 'std', 'median', 'range']: 
            features[f'vtl_{stat}'] = 0
        
    return features


def _compute_advanced(y, sr, f0=None):
    """Computes Flux, HNR, CPP, etc."""
    features = {}
    
    # 1. Spectral Flux
    S = np.abs(librosa.stft(y))
    S_norm = S / (S.sum(axis=0, keepdims=True) + 1e-9)
    flux = np.sqrt(np.sum((np.diff(S_norm, axis=1))**2, axis=0))
    flux_delta = librosa.feature.delta(flux)
    flux_delta2 = librosa.feature.delta(flux, order=2)
    
    features.update(summarize_vector(flux, "spectral_flux"))
    features.update(summarize_vector(flux_delta, "spectral_flux_delta"))
    features.update(summarize_vector(flux_delta2, "spectral_flux_delta2"))
    
    # Band-wise Flux
    freqs = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0]*2 - 2)
    band_masks = {
        "sub":    (freqs >= 20)   & (freqs < 100),
        "low":    (freqs >= 100)  & (freqs < 300),
        "medlow": (freqs >= 300)  & (freqs < 1000),
        "med":    (freqs >= 1000) & (freqs < 2500),
        "medhi":  (freqs >= 2500) & (freqs < 5000),
        "hi":     (freqs >= 5000),
    }
    
    for name, mask in band_masks.items():
        S_band = S_norm[mask, :]
        if S_band.size > 0:
            flux_band = np.sqrt(np.sum((np.diff(S_band, axis=1))**2, axis=0))
            flux_band_delta = librosa.feature.delta(flux_band)
            flux_band_delta2 = librosa.feature.delta(flux_band, order=2)
            features.update(summarize_vector(flux_band, f"spectral_flux_{name}"))
            features.update(summarize_vector(flux_band_delta, f"spectral_flux_{name}_delta"))
            features.update(summarize_vector(flux_band_delta2, f"spectral_flux_{name}_delta2"))
        else:
            for suffix in ["", "_delta", "_delta2"]:
                for stat in ["mean", "std", "median", "range"]:
                    features[f"spectral_flux_{name}{suffix}_{stat}"] = np.nan

    # 2. HNR & Voicing
    if f0 is not None and not np.all(np.isnan(f0)):
        y_harm = librosa.effects.harmonic(y)
        y_noise = y - y_harm
        hnr = 10 * np.log10((np.sum(y_harm**2) + 1e-9) / (np.sum(y_noise**2) + 1e-9))
        features["hnr"] = float(hnr)
        features["voicing_ratio"] = float(np.mean(~np.isnan(f0)))
    else:
        features["hnr"] = np.nan
        features["voicing_ratio"] = 0.0

    # 3. CPP (Cepstral Peak Prominence)
    try:
        frame_length = int(0.04 * sr)
        hop_length = frame_length // 2
        S_cpp = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
        frame_idx = np.argmax(S_cpp.sum(axis=0))
        mag = S_cpp[:, frame_idx] + 1e-9
        log_mag = np.log(mag)
        cepstrum = np.fft.irfft(log_mag)
        
        q_min, q_max = 0.0025, 0.03
        idx_min = int(q_min * sr)
        idx_max = min(int(q_max * sr), len(cepstrum))
        
        region = cepstrum[idx_min:idx_max]
        if len(region) > 0:
            peak_idx_rel = np.argmax(region)
            peak_idx = idx_min + peak_idx_rel
            peak_val = cepstrum[peak_idx]
            
            x = np.array([idx_min, idx_max - 1])
            yb = cepstrum[x]
            a = (yb[1] - yb[0]) / (x[1] - x[0] + 1e-9)
            b = yb[0] - a * x[0]
            baseline_at_peak = a * peak_idx + b
            features["cpp"] = float(peak_val - baseline_at_peak)
        else:
            features["cpp"] = 0.0
    except Exception as e:
        print(f"    ! CPP extraction failed: {e}")
        features["cpp"] = 0.0

    # 4. Autocorrelation Periodicity
    try:
        frame_length = int(0.04 * sr)
        hop_length = frame_length // 2
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        energies = np.sum(frames**2, axis=0)
        frame = frames[:, np.argmax(energies)]
        frame = frame - np.mean(frame)
        frame = frame / (np.std(frame) + 1e-9)
        
        acf = np.correlate(frame, frame, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / (acf[0] + 1e-9)
        acf_nozero = acf[1:]
        
        peak_idx = np.argmax(acf_nozero) + 1
        peak_val = acf[peak_idx]
        
        features["acf_peak_lag"] = float(peak_idx / sr)
        features["acf_peak_value"] = float(peak_val)
        features["acf_peak_to_mean"] = float(peak_val / (np.mean(acf_nozero) + 1e-9))
        features["acf_peak_to_std"] = float(peak_val / (np.std(acf_nozero) + 1e-9))
        features.update(summarize_vector(acf_nozero, "acf"))
    except Exception as e:
        print(f"    ! Autocorrelation extraction failed: {e}")
        features["acf_peak_value"] = 0.0
        
    return features


def extract_all_features(audio_path, f0=None):
    """
    Master function to extract ALL features (Spectral, Formants, Advanced) in one shot.
    """
    sname = os.path.splitext(os.path.basename(audio_path))[0]
    
    # 1. Load Audio ONCE
    y, sr = librosa.load(audio_path, sr=None)
    
    # 2. Extract Standard Spectral Features
    spect_series = extract_spect_features(audio_path, y=y, sr=sr)
    spect_dict = spect_series.to_dict()
    
    # 3. Extract Formants
    formant_dict = _compute_formants(y, sr)
    
    # 4. Extract Advanced Features
    adv_dict = _compute_advanced(y, sr, f0=f0)
    
    # 5. Merge All
    all_features = {"id": sname}
    all_features.update(spect_dict)
    all_features.update(formant_dict)
    all_features.update(adv_dict)
    
    return pd.Series(all_features)