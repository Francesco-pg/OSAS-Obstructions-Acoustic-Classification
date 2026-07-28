#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Features
Module:      Run00_SmokeTest.py
Purpose:     End-to-end smoke / regression test on SYNTHETIC data.

             Generates a tiny synthetic cohort (no private dataset required),
             redirects all paths to a throw-away temporary workspace, and runs
             the real pipeline Run01 -> Run02 -> Run04 -> Run05 -> Run06.
             It then asserts that every expected artifact was produced and that
             the aggregated metrics are finite.

             Two uses:
               1. Reviewers can confirm the pipeline executes without the data.
               2. It is deterministic (fixed seeds + tiny grids), so running it
                  on two versions of the code and diffing the printed
                  "REGRESSION FINGERPRINT" tells you whether a change altered
                  the numerical result of the active configuration.

Usage (from the src/ directory):
    python Run00_SmokeTest.py           # run, then delete the workspace
    python Run00_SmokeTest.py --keep    # keep the workspace for inspection

Exit code: 0 = PASS, 1 = FAIL.
"""

import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

# Import the config modules FIRST so we can monkeypatch their attributes
# before any Run* module reads them (every Run* module accesses config.X at
# call time, so overriding the attributes here is sufficient).
import config
import model_config

# ------------------------------------------------------------------ #
# Synthetic-cohort parameters (kept tiny on purpose)
# ------------------------------------------------------------------ #
N_SUBJECTS_PER_CLASS = 6      # 6 palato + 6 epiglottide = 12 subjects
N_CLIPS_PER_SUBJECT  = 2      # -> 24 clips total
SR                   = 44100  # must match config.CURRENT_SR
DURATION_S           = 1.5

# Class-specific pseudo-formant centres so the two classes are separable
# enough to avoid degenerate single-class predictions (Hz).
CLASS_FORMANTS = {
    "palato":      [700.0, 1200.0, 2600.0],
    "epiglottide": [500.0, 1700.0, 3200.0],
}


def _synth_clip(rng, f0, formants):
    """Build one snore-like stereo clip: harmonic source + resonant bands + noise."""
    n = int(DURATION_S * SR)
    t = np.arange(n) / SR

    sig = np.zeros(n)
    for h in range(1, 25):                       # harmonic (voiced) source
        sig += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)

    bump = np.exp(-((t - DURATION_S / 2) ** 2) / (2 * 0.30 ** 2))
    for fc in formants:                          # crude formant/resonance shaping
        sig += 0.5 * np.sin(2 * np.pi * fc * t) * bump

    sig += 0.05 * rng.standard_normal(n)         # broadband energy in every band
    sig *= np.hanning(n)                          # snore-like envelope
    sig /= (np.max(np.abs(sig)) + 1e-9)

    stereo = np.stack([sig, sig * 0.98], axis=1).astype(np.float32)
    return stereo


def _make_synthetic_dataset(stereo_root, rng):
    """Write stereo .wav files named '<SID> <class> <NN>.wav' into subject folders."""
    specs = []
    for i in range(N_SUBJECTS_PER_CLASS):
        specs.append((f"S{i + 1:02d}", "palato"))
    for i in range(N_SUBJECTS_PER_CLASS):
        specs.append((f"S{i + 1 + N_SUBJECTS_PER_CLASS:02d}", "epiglottide"))

    n_written = 0
    for sid, cls in specs:
        subj_dir = stereo_root / f"folder_{sid}"
        subj_dir.mkdir(parents=True, exist_ok=True)
        for c in range(N_CLIPS_PER_SUBJECT):
            f0 = 100.0 + 20.0 * rng.random()
            clip = _synth_clip(rng, f0, CLASS_FORMANTS[cls])
            # NB: the filename parser splits on spaces -> "<subject> <class> <idx>"
            fname = f"{sid} {cls} {c + 1:02d}.wav"
            sf.write(subj_dir / fname, clip, SR)
            n_written += 1
    return n_written


def _override_paths(workspace):
    """Redirect every config path into the temporary workspace."""
    data = workspace / "subjects"
    out = workspace / "outputs"

    config.datadir                             = data
    config.stereo_dataset_dir                  = data / "full_stereo"
    config.mono_dataset_dir                    = data / config.mono_dataset_name
    config.normalized_mono_dataset_dir         = data / config.normalized_mono_dataset_name
    config.resampled_normalized_mono_dataset_dir = data / config.resampled_normalized_mono_dataset_name
    config.source_dataset_dir                  = config.resampled_normalized_mono_dataset_dir

    config.outputsdir          = out
    config.full_dataset_path   = out / f"{config.full_dataset_name}.xlsx"
    config.classes_report_path = out / config.classes_report_filename
    config.ModelOutdir         = out / config.currentOutName / "model"
    config.AnalysisOutdir      = out / config.currentOutName / "analysis"

    # Small enough for a tiny cohort, still exercises the outer/inner CV loops.
    config.N_FOLDS = 2

    config.stereo_dataset_dir.mkdir(parents=True, exist_ok=True)
    config.ModelOutdir.mkdir(parents=True, exist_ok=True)
    config.AnalysisOutdir.mkdir(parents=True, exist_ok=True)


def _override_models():
    """Shrink grids / iterations so the ML stage runs in seconds, not minutes."""
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    rs = model_config.RANDOM_STATE
    model_config.N_SPLITS            = 2   # inner-CV splits (must be <= subjects/class in train)
    model_config.N_FEATURES_TO_SELECT = 10

    model_config.MODELS = {
        "SVM": {
            "estimator": SVC(probability=True, random_state=rs),
            "scale": True,
            "params": [{
                "clf__kernel": ["rbf"],
                "clf__C": [1],
                "clf__gamma": ["scale"],
                "clf__class_weight": [None],
            }],
        },
        "MLP": {
            "estimator": MLPClassifier(max_iter=200, random_state=rs, early_stopping=False),
            "scale": True,
            "params": {
                "clf__hidden_layer_sizes": [(16,)],
                "clf__activation": ["relu"],
                "clf__alpha": [0.001],
                "clf__solver": ["adam"],
            },
        },
    }


def _read_fingerprint():
    """Collect a compact, diffable summary of the produced global results."""
    lines = []
    for txt in sorted(config.ModelOutdir.glob("GLOBAL_RESULTS_*.txt")):
        content = txt.read_text(encoding="utf-8").splitlines()
        bal = next((l.strip() for l in content if l.startswith("Balanced Accuracy")), "Balanced Accuracy: <missing>")
        auc = next((l.strip() for l in content if l.startswith("ROC AUC")), "ROC AUC: <missing>")
        lines.append(f"{txt.name}\n    {bal}\n    {auc}")
    return lines


def _assert(cond, msg, failures):
    if not cond:
        failures.append(msg)
    return cond


def run(keep=False):
    # The pipeline prints emoji; on a default Windows console (cp1252) that
    # raises UnicodeEncodeError. Force UTF-8 (replace on failure) so the test
    # is self-contained regardless of the host console encoding.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    workspace = Path(tempfile.mkdtemp(prefix="osas_smoke_"))
    # config was imported with its hard-coded paths, which created an (empty)
    # outputs dir next to the repo. Remember it so we can tidy up afterwards.
    stray_outputs = config.outputsdir
    failures = []

    print("=" * 70)
    print("OSAS PIPELINE SMOKE TEST (synthetic data)")
    print(f"Workspace: {workspace}")
    print("=" * 70)

    try:
        _override_paths(workspace)
        _override_models()

        rng = np.random.default_rng(0)
        n = _make_synthetic_dataset(config.stereo_dataset_dir, rng)
        print(f"[setup] wrote {n} synthetic stereo clips "
              f"({2 * N_SUBJECTS_PER_CLASS} subjects, {N_CLIPS_PER_SUBJECT} clips each)\n")

        # Import the pipeline stages only now that the config is patched.
        import Run01_Dataset
        import Run02_FeatureExtraction
        import Run04_TrainTestSplit
        import Run05_TrainTest
        import Run06_AggregateResults

        print("\n########## Run01: preprocessing ##########")
        Run01_Dataset.main()
        print("\n########## Run02: feature extraction ##########")
        Run02_FeatureExtraction.main()
        print("\n########## Run04: train/test split ##########")
        Run04_TrainTestSplit.main()
        print("\n########## Run05: nested-CV train/test ##########")
        Run05_TrainTest.main()
        print("\n########## Run06: aggregation ##########")
        Run06_AggregateResults.main()

        # ---------------- assertions ---------------- #
        print("\n" + "=" * 70)
        print("VERIFYING ARTIFACTS")
        print("=" * 70)

        _assert(config.full_dataset_path.exists(),
                f"missing feature table {config.full_dataset_path.name}", failures)

        for fold_idx in range(1, config.N_FOLDS + 1):
            paths = config.get_fold_paths(fold_idx)
            _assert(paths["train_features"].exists(),
                    f"fold {fold_idx}: missing train_features.csv", failures)
            _assert(paths["test_features"].exists(),
                    f"fold {fold_idx}: missing test_features.csv", failures)
            _assert((paths["mod_output_dir"] / f"rfe_scores_fold_{fold_idx:02d}.csv").exists(),
                    f"fold {fold_idx}: missing rfe_scores csv", failures)
            for model_name in ("SVM", "MLP"):
                mdl = paths["mod_output_dir"] / f"best_model_{config.model_n:02d}_{model_name}.pkl"
                _assert(mdl.exists(),
                        f"fold {fold_idx}: {model_name} model not saved "
                        f"(training likely raised and was swallowed) -> {mdl.name}",
                        failures)

        global_reports = list(config.ModelOutdir.glob("GLOBAL_RESULTS_*.txt"))
        _assert(len(global_reports) >= 2,
                f"expected >=2 GLOBAL_RESULTS reports (SVM, MLP), found {len(global_reports)}",
                failures)

        print("\nREGRESSION FINGERPRINT (diff this across code versions):")
        print("-" * 70)
        for block in _read_fingerprint():
            print(block)
        print("-" * 70)

    except Exception:
        failures.append("uncaught exception:\n" + traceback.format_exc())
    finally:
        if keep:
            print(f"\n[--keep] workspace retained at: {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)
        # Remove the empty outputs tree that importing config created next to the repo.
        try:
            if stray_outputs.exists() and not any(stray_outputs.rglob("*.txt")) \
                    and not any(stray_outputs.rglob("*.xlsx")) and not any(stray_outputs.rglob("*.pkl")):
                shutil.rmtree(stray_outputs, ignore_errors=True)
        except Exception:
            pass

    print("\n" + "=" * 70)
    if failures:
        print(f"RESULT: FAIL ({len(failures)} problem(s))")
        for i, msg in enumerate(failures, 1):
            print(f"  [{i}] {msg}")
        print("=" * 70)
        return 1

    print("RESULT: PASS — pipeline ran end-to-end and all artifacts were produced.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic end-to-end smoke test.")
    parser.add_argument("--keep", action="store_true",
                        help="keep the temporary workspace instead of deleting it")
    args = parser.parse_args()
    sys.exit(run(keep=args.keep))
