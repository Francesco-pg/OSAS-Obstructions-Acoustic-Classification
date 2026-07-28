# -*- coding: utf-8 -*-
"""
Created on Tue May 13 14:36:55 2025

@author: bianc
"""
import os
import numpy as np
import librosa
# import soundfile as sf
from tqdm import tqdm

def _compute_rms_global(y, eps=1e-12):
    """RMS globale sui campioni (più semplice e stabile del mean dei frame RMS)."""
    return float(np.sqrt(np.mean(y**2) + eps))

def _collect_wavs(folder, exclude_folder_name=None):
    """Raccoglie tutti i .wav ricorsivamente, escludendo cartelle con quel nome (se dato)."""
    wavs = []
    for root, _, files in os.walk(folder):
        if exclude_folder_name is not None:
            rel = os.path.relpath(root, folder)
            if exclude_folder_name in rel.split(os.sep):
                continue
        for f in files:
            if f.lower().endswith(".wav"):
                wavs.append(os.path.join(root, f))
    return wavs

# def fit_transform_rms_median_train_test(
#     train_folder,
#     test_folder,
#     output_folder_name="normalized_wav_median_rms",
#     allow_amplify=True,
#     peak_limit_dbfs=-1.0,     # anti-clipping: -1 dBFS (0.891)
#     mono=True,
#     sr=None,                 # None = mantieni SR originale
#     exclude_output_name=None # es: "normalized_wav" se vuoi evitare di processare output precedenti
# ):
#     """
#     FIT: calcola target RMS = mediana degli RMS dei file nel train_folder
#     TRANSFORM: normalizza i wav in train_folder e test_folder verso quel target RMS
#                e salva in una cartella nuova mantenendo la struttura Train/... e Test/...

#     Returns:
#         target_rms (float), output_root (str)
#     """
#     # Cartella base comune (per preservare Train/Test nella struttura output)
#     base_root = os.path.commonpath([train_folder, test_folder])

#     # Output accanto alla base_root
#     output_root = os.path.join(base_root, output_folder_name)
#     os.makedirs(output_root, exist_ok=True)

#     # -------- FIT (solo train) --------
#     train_wavs = _collect_wavs(train_folder, exclude_folder_name=exclude_output_name)
#     if not train_wavs:
#         raise RuntimeError(f"Nessun .wav trovato in train_folder: {train_folder}")

#     train_rms = []
#     for p in tqdm(train_wavs, desc="FIT: computing train RMS"):
#         y, _sr = librosa.load(p, sr=sr, mono=mono)
#         train_rms.append(_compute_rms_global(y))

#     train_rms = np.array(train_rms, dtype=float)
#     target_rms = float(np.median(train_rms))

#     print("\n==============================")
#     print("RMS NORMALIZATION (FIT/TRANSFORM)")
#     print("==============================")
#     print(f"[FIT] target RMS (median of TRAIN): {target_rms:.6f}")
#     print(f"[FIT] train RMS min/med/max: {train_rms.min():.6f} / {np.median(train_rms):.6f} / {train_rms.max():.6f}")
#     print(f"[OUT] output_root: {output_root}")
#     print("==============================\n")

#     # Anti-clipping: limite di picco lineare
#     peak_limit = float(10 ** (peak_limit_dbfs / 20.0))  # es -1 dBFS -> 0.891

#     def _transform_folder(input_folder, label):
#         wavs = _collect_wavs(input_folder, exclude_folder_name=exclude_output_name)
#         if not wavs:
#             print(f"⚠️ Nessun .wav trovato in {label}: {input_folder}")
#             return

#         for p in tqdm(wavs, desc=f"TRANSFORM: normalizing {label}"):
#             y, sr_loaded = librosa.load(p, sr=sr, mono=mono)
#             cur_rms = _compute_rms_global(y)

#             if cur_rms <= 0:
#                 y_norm = np.zeros_like(y)
#             else:
#                 scale = target_rms / cur_rms
#                 if not allow_amplify:
#                     scale = min(1.0, scale)  # solo attenuazione
#                 y_norm = y * scale

#                 # Anti-clipping per-file
#                 peak = float(np.max(np.abs(y_norm))) if y_norm.size else 0.0
#                 if peak > peak_limit and peak > 0:
#                     y_norm = y_norm * (peak_limit / peak)

#             # Mantiene la struttura relativa rispetto alla base_root
#             rel_path = os.path.relpath(p, base_root)
#             out_path = os.path.join(output_root, rel_path)
#             os.makedirs(os.path.dirname(out_path), exist_ok=True)

#             sf.write(out_path, y_norm, sr_loaded)

#     # -------- TRANSFORM (train + test con stesso target) --------
#     _transform_folder(train_folder, "TRAIN")
#     _transform_folder(test_folder, "TEST")

#     print("\n✅ Done.")
#     print(f"Target RMS (train median): {target_rms:.6f}")
#     print(f"Saved normalized data in:  {output_root}\n")

#     return target_rms, output_root


# ======= ESEMPIO USO =======
# train = r"C:\Users\bianc\Desktop\osas-project-main\osas_new\data\Train"
# test  = r"C:\Users\bianc\Desktop\osas-project-main\osas_new\data\Test"
# fit_transform_rms_median_train_test(train, test, allow_amplify=True, peak_limit_dbfs=-1.0, exclude_output_name="normalized_wav")

# ESEMPIO USO:
# normalize_rms_to_median(r"C:\Users\bianc\Desktop\...\root_folder", allow_amplify=True)


# def compute_rms_librosa(y):
#     """Calcola RMS globale usando librosa"""
#     rms = librosa.feature.rms(y=y)[0]
#     return np.mean(rms)

# def normalize_rms_wav_librosa(root_folder, output_folder_name='normalized_wav'):
#     """
#     Normalizza tutti i file .wav (con librosa) in root_folder e sottocartelle,
#     salvando la struttura normalizzata nella cartella superiore (genitore di root_folder).

#     Parameters:
#         root_folder (str): percorso alla cartella che contiene i .wav
#         output_folder_name (str): nome della cartella in cui salvare i file normalizzati (default: 'normalized_wav')
#     """
#     wav_files = []
#     rms_values = {}
    
#     all_files_to_process = []
#     for root, _, files in os.walk(root_folder):
#         rel_root = os.path.relpath(root, root_folder)
#         if output_folder_name in rel_root.split(os.sep):
#             continue
#         for file in files:
#             if file.lower().endswith('.wav'):
#                 all_files_to_process.append(os.path.join(root, file))

#     # Step 1: Trova tutti i .wav e calcola RMS
#     for full_path in tqdm(all_files_to_process, desc="Calculating RMS"):
#         y, sr = librosa.load(full_path, sr=None)
#         rms = compute_rms_librosa(y)
#         rms_values[full_path] = rms
#         wav_files.append((full_path, y, sr))

#     if not wav_files:
#         print("⚠️ Nessun file .wav trovato.")
#         return

#     # Step 2: Calcola RMS minimo
#     min_rms = min(rms_values.values())

#     # 📁 Output folder = cartella genitore di root_folder + output_folder_name
#     parent_folder = os.path.dirname(root_folder)
#     output_root = os.path.join(parent_folder, output_folder_name)

#     # Step 3: Normalizza e salva
#     for filepath, y, sr in tqdm(wav_files, desc="Normalizing files"):
#         current_rms = rms_values[filepath]
#         scale = min_rms / current_rms if current_rms > 0 else 0
#         y_norm = y * scale

#         relative_path = os.path.relpath(filepath, root_folder)
#         output_path = os.path.join(output_root, relative_path)
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)

#         sf.write(output_path, y_norm, sr)


#%%


# normalize_rms_wav_librosa('C:/Users/bianc/OneDrive/Desktop/Lavoro/PhD/OSAS project/osas-project/data/Dataset OSAS_clean_resampled')


#%%
# import os
# import librosa
# import matplotlib.pyplot as plt


# original_file = 'C:/Users/bianc/OneDrive/Desktop/Lavoro/PhD/OSAS project/osas-project/data/Dataset_OSAS_resampled_renamed/Non epiglottide/Palato ap senza ugola CUT/OSA0259.wav'

# # Path equivalente nella cartella dei normalizzati
# normalized_file = original_file.replace(
#     'Dataset_OSAS_resampled_renamed', 'normalized_wav')


# y_orig, sr_orig = librosa.load(original_file, sr=None)
# y_norm, sr_norm = librosa.load(normalized_file, sr=None)


# plt.figure(figsize=(14, 5))

# plt.subplot(2, 1, 1)
# plt.plot(y_orig, color='gray')
# plt.title("Original Signal (RMS = {:.4f})".format(librosa.feature.rms(y=y_orig)[0].mean()))
# plt.xlabel("Time (samples)")
# plt.ylabel("Amplitude")

# plt.subplot(2, 1, 2)
# plt.plot(y_norm, color='green')
# plt.title("Normalized Signal (RMS = {:.4f})".format(librosa.feature.rms(y=y_norm)[0].mean()))
# plt.xlabel("Time (samples)")
# plt.ylabel("Amplitude")

# plt.tight_layout()
# plt.show()

# #%%

# import numpy as np
# import librosa

# # Carica i due file
# y_orig, sr_orig = librosa.load(original_file, sr=None)
# y_norm, sr_norm = librosa.load(normalized_file, sr=None)

# # Controlla se la frequenza di campionamento è uguale
# if sr_orig != sr_norm:
#     print(f"⚠️ Sampling rate diverso: originale={sr_orig}, normalizzato={sr_norm}")
# else:
#     print(f"✅ Sampling rate uguale: {sr_orig} Hz")

# # Verifica se la lunghezza è la stessa
# if len(y_orig) != len(y_norm):
#     print(f"⚠️ Lunghezza diversa: originale={len(y_orig)}, normalizzato={len(y_norm)}")
# else:
#     print(f"✅ Lunghezza uguale: {len(y_orig)} campioni")

# # Confronta RMS
# rms_orig = librosa.feature.rms(y=y_orig)[0].mean()
# rms_norm = librosa.feature.rms(y=y_norm)[0].mean()

# print(f"📊 RMS originale: {rms_orig:.6f}")
# print(f"📊 RMS normalizzato: {rms_norm:.6f}")
# print(f"🔁 Rapporto (norm/orig): {rms_norm / rms_orig:.3f}")

# # Controlla se i segnali sono uguali (entro una tolleranza)
# if np.allclose(y_orig, y_norm, atol=1e-6):
#     print("🟢 I due segnali sono (praticamente) identici.")
# else:
#     diff = np.abs(y_orig - y_norm)
#     print(f"🔺 Differenza media: {np.mean(diff):.6f}")
#     print(f"🔺 Differenza massima: {np.max(diff):.6f}")

# # riga a caso
# print('hello world')




