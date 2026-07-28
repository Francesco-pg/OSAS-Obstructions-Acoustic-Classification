#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Features
Module:      extract_features_only.py
Purpose:     Standalone acoustic-feature extraction, decoupled from the full
             pipeline. No dataset layout, no train/test split, no labels — just
             signals in, one feature row per signal out.

             Reuses the exact same feature engine as Run02 (spectral + formants
             + advanced), so the columns match `full_dataset.xlsx`.

             The three Run01 preprocessing steps are available as *optional*
             flags, so you can reproduce the paper's pipeline or feed raw
             signals straight in:
                 1. single-channel extraction (left channel)   --single-channel
                 2. global RMS normalization                    --rms-norm MODE
                 3. resampling                                  --target-sr HZ
             `--run01` turns on all three with the paper's settings (from config).

Usage (from the src/ directory):

    # A) Raw features, no preprocessing
    python extract_features_only.py /path/to/wavs -o features.csv

    # B) Reproduce the full Run01 preprocessing, then extract
    python extract_features_only.py /path/to/wavs -o features.csv --run01

    # C) Pick individual steps
    python extract_features_only.py /path/to/wavs -o features.csv \
        --single-channel --rms-norm min --target-sr 22050

    # D) Your own in-memory time series (import it):
    from extract_features_only import extract_features_from_array
    row = extract_features_from_array(y, sr, name="clip_001",
                                      channel=0, rms_target=0.05, target_sr=22050)
"""

import argparse
import glob
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# The feature engine (imports config for the FORMANT_METHOD toggle, so run this
# from the src/ directory just like the Run* scripts).
from processing import spectral_feat
import config

# F0 tracking settings — identical to processing/f0.py so HNR / voicing_ratio
# match what the main pipeline produces.
F0_SR = 11025


# ------------------------------------------------------------------ #
# Preprocessing helpers (mirror the three Run01 steps, in-memory)
# ------------------------------------------------------------------ #
def _select_channel(y, channel=None):
    """Reduce a multichannel signal to 1-D.

    channel=None -> average all channels (standard mono downmix)
    channel=int  -> pick that channel (Run01 step 1 uses channel 0 = left)
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return y
    # Normalize orientation to (n_channels, n_samples)
    if y.shape[0] > y.shape[1]:
        y = y.T
    if channel is None:
        return y.mean(axis=0)
    return y[channel]


def _quantize_pcm16(y):
    """Emulate a soundfile WAV round-trip at the default PCM_16 subtype.

    Run01 writes each intermediate .wav with soundfile's default subtype
    (PCM_16 for WAV), so every stage quantizes to 16-bit. Reproducing that here
    lets the in-memory path match `full_dataset.xlsx` (notably spectral-contrast
    features, which are sensitive to the quantization noise floor). librosa/
    libsndfile scale float<->int16 by 32768.
    """
    q = np.clip(np.round(np.clip(y, -1.0, 1.0) * 32768.0), -32768, 32767)
    return (q / 32768.0).astype(np.float64)


def _rms(y):
    """Global RMS as defined in processing/normalization.py (mean of frame RMS)."""
    return float(np.mean(librosa.feature.rms(y=y)[0]))


def _rms_normalize(y, target_rms):
    """Scale `y` to `target_rms`, with the same anti-clipping as Run01."""
    cur = _rms(y)
    if cur > 0:
        y = y * (target_rms / cur)
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 1.0:                      # peak limiting to avoid clipping
        y = y / peak
    return y


def _resample(y, sr, target_sr):
    if target_sr and sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y, sr


def compute_f0(y, sr):
    """pYIN fundamental-frequency track, matching the pipeline's settings."""
    if sr != F0_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=F0_SR)
    f0, _, _ = librosa.pyin(
        y,
        sr=F0_SR,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        frame_length=1024,
        hop_length=512,
        fill_na=np.nan,
    )
    return f0


# ------------------------------------------------------------------ #
# Feature extraction
# ------------------------------------------------------------------ #
def _extract(y, sr, name, with_f0):
    """Pure feature extraction on an already-preprocessed 1-D signal."""
    f0 = compute_f0(y, sr) if with_f0 else None

    # Same three feature groups, same merge order as spectral_feat.extract_all_features
    spect   = spectral_feat.extract_spect_features(name, y=y, sr=sr).to_dict()
    formant = spectral_feat._compute_formants(y, sr)
    adv     = spectral_feat._compute_advanced(y, sr, f0=f0)

    row = {"id": name}
    row.update(spect)
    row.update(formant)
    row.update(adv)
    return row


def preprocess(y, sr, channel=None, rms_target=None, target_sr=None, emulate_io=False):
    """Apply the Run01 steps in order (channel -> RMS -> resample) to one signal.

    emulate_io=True reproduces the 16-bit PCM WAV round-trip that Run01 performs
    when it writes each intermediate file — needed to match `full_dataset.xlsx`.
    """
    y = _select_channel(y, channel)
    if emulate_io:
        y = _quantize_pcm16(y)                     # emulate the mono .wav write
    if rms_target is not None:
        y = _rms_normalize(y, rms_target)
        if emulate_io:
            y = _quantize_pcm16(y)                 # emulate the normalized .wav write
    y, sr = _resample(y, sr, target_sr)
    if emulate_io and target_sr:
        y = _quantize_pcm16(y)                     # emulate the resampled .wav write
    return y, sr


def extract_features_from_array(y, sr, name="signal", with_f0=True,
                                channel=None, rms_target=None, target_sr=None,
                                emulate_io=False):
    """
    Extract the full feature set from a single signal, optionally applying the
    Run01 preprocessing steps first (in Run01 order: channel -> RMS -> resample).

    Args:
        y (np.ndarray): audio samples (1-D, or multichannel).
        sr (int):       sample rate of `y`.
        name (str):     identifier written to the 'id' column.
        with_f0 (bool): compute F0 (needed for HNR / voicing_ratio).
        channel (int|None): None -> mono downmix; int -> pick that channel
                            (pass 0 to mimic Run01's single-channel step).
        rms_target (float|None): if set, RMS-normalize the signal to this value.
                            NB: Run01 uses a *global* target across the batch;
                            for a lone array you must supply the value yourself
                            (extract_folder computes it for you).
        target_sr (int|None): if set, resample to this rate before extraction.
        emulate_io (bool): reproduce Run01's 16-bit WAV quantization so values
                            match `full_dataset.xlsx` (see _quantize_pcm16).

    Returns:
        dict: {'id': name, <feature>: value, ...}
    """
    y, sr = preprocess(y, sr, channel=channel, rms_target=rms_target,
                       target_sr=target_sr, emulate_io=emulate_io)
    return _extract(y, sr, name, with_f0)


def _load(path, channel=None, emulate_io=False):
    """Load a wav at native sample rate and select one channel (Run01 step 1)."""
    y, sr = librosa.load(path, sr=None, mono=False)
    y = _select_channel(y, channel)
    if emulate_io:
        y = _quantize_pcm16(y)     # Run01 writes the mono file as 16-bit PCM
    return y, sr


def extract_folder(input_dir, with_f0=True, single_channel=False,
                   rms_mode=None, target_sr=None, emulate_io=True):
    """
    Extract features for every .wav under `input_dir` into a DataFrame,
    optionally applying the Run01 preprocessing steps.

    Args:
        single_channel (bool): take the left channel (Run01 step 1).
        rms_mode (str|None):   'min' | 'max' | 'mean' | 'median' -> apply GLOBAL
                               RMS normalization to that target (Run01 step 2).
        target_sr (int|None):  resample to this rate (Run01 step 3).
        emulate_io (bool):     reproduce Run01's 16-bit WAV quantization so the
                               output matches full_dataset.xlsx. Set False for
                               cleaner (lossless) float processing.
    """
    paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True))
    if not paths:
        raise SystemExit(f"No .wav files found under: {input_dir}")

    channel = 0 if single_channel else None

    # --- Run01 step 2 is GLOBAL: first pass computes the batch target RMS ---
    rms_target = None
    if rms_mode is not None:
        reducer = {"min": np.min, "max": np.max,
                   "mean": np.mean, "median": np.median}[rms_mode]
        rms_vals = []
        for p in tqdm(paths, desc=f"RMS pass (global {rms_mode})"):
            try:
                y, _ = _load(p, channel, emulate_io)
                rms_vals.append(_rms(y))
            except Exception as e:
                print(f"  ! RMS pass skipped {p}: {e}")
        if not rms_vals:
            raise SystemExit("Could not compute RMS on any file.")
        rms_target = float(reducer(rms_vals))
        print(f"  -> Global target RMS ({rms_mode}) = {rms_target:.6f}")

    # --- Second pass: preprocess + extract ---
    rows = []
    for p in tqdm(paths, desc="Extracting features"):
        try:
            y, sr = _load(p, channel, emulate_io)   # mono selected (+ quantized)
            # channel already handled; quantizing again in preprocess is idempotent
            y, sr = preprocess(y, sr, channel=None, rms_target=rms_target,
                               target_sr=target_sr, emulate_io=emulate_io)
            rows.append(_extract(y, sr, Path(p).stem, with_f0))
        except Exception as e:
            print(f"  ! Skipped {p}: {e}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Standalone acoustic feature extraction, with optional Run01 preprocessing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input_dir", help="folder containing .wav files (searched recursively)")
    ap.add_argument("-o", "--output", default="features.csv",
                    help="output file (.csv or .xlsx)")
    ap.add_argument("--no-f0", action="store_true",
                    help="skip F0 tracking (leaves HNR / voicing_ratio empty, runs faster)")

    # --- optional Run01 preprocessing steps ---
    ap.add_argument("--single-channel", action="store_true",
                    help="Run01 step 1: take the left channel instead of mixing to mono")
    ap.add_argument("--rms-norm", choices=["min", "max", "mean", "median"], default=None,
                    help="Run01 step 2: global RMS normalization to this batch target")
    ap.add_argument("--target-sr", type=int, default=None,
                    help="Run01 step 3: resample to this sample rate (Hz)")
    ap.add_argument("--run01", action="store_true",
                    help="apply ALL three Run01 steps with the paper's settings "
                         f"(single-channel, rms='{config.rmsTarget}', target-sr={config.TARGET_SR})")
    ap.add_argument("--no-quantize", action="store_true",
                    help="do NOT emulate Run01's 16-bit WAV round-trip; process in "
                         "lossless float (cleaner, but will not match full_dataset.xlsx)")

    args = ap.parse_args()

    single_channel = args.single_channel
    rms_mode       = args.rms_norm
    target_sr      = args.target_sr

    if args.run01:   # convenience: fill in the paper's defaults where not overridden
        single_channel = True
        if rms_mode is None:
            rms_mode = config.rmsTarget
        if target_sr is None:
            target_sr = config.TARGET_SR

    emulate_io = not args.no_quantize

    steps = []
    if single_channel: steps.append("single-channel")
    if rms_mode:       steps.append(f"rms-norm({rms_mode})")
    if target_sr:      steps.append(f"resample({target_sr}Hz)")
    if steps and emulate_io: steps.append("16-bit quantization")
    print(f"Preprocessing: {' -> '.join(steps) if steps else 'none (raw signals)'}\n")

    df = extract_folder(
        args.input_dir,
        with_f0=not args.no_f0,
        single_channel=single_channel,
        rms_mode=rms_mode,
        target_sr=target_sr,
        emulate_io=emulate_io,
    )

    if args.output.lower().endswith((".xlsx", ".xls")):
        df.to_excel(args.output, index=False)
    else:
        df.to_csv(args.output, index=False)

    print(f"\n✅ {len(df)} signals -> {args.output}  ({df.shape[1] - 1} features per signal)")


if __name__ == "__main__":
    main()
