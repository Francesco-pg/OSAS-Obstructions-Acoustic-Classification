# OSAS Classification from Acoustic Features
 
Classify **Obstructive Sleep Apnea Syndrome (OSAS)** obstruction sites — **palatal** vs. **epiglottic** — using acoustic features extracted from audio recordings.
 
---
 
## Overview
 
This repository provides an end-to-end machine learning pipeline that extracts acoustic metrics from snoring recordings and trains classifiers (SVM, MLP) to distinguish the anatomical site of airway obstruction. The pipeline is based on a  **Nested Cross-Validation** scheme.
 
## Data Requirements
 
The pipeline requires input audio/video that has been **manually cleaned** beforehand:
 
- All silences **before and after** the snoring obstruction event should be be cropped out. 
The entire pipeline relies on this manually cleaned audio data.
---
 
## ⚙️ Setup & Configuration
 
There is **only one variable** you need to change before running the pipeline.
 
Open `config.py` and set `datadir` to the absolute path of your raw, manually cleaned audio dataset. 
```python
# In config.py
datadir = Path("D://Datasets//AI_healthcare//osas-project//osas_data_cleaned//subjects")
```

Audio files should be stored in stereo format into folders, where every folder is a different subject with its own ID and class in the filename (e.g folder_S01 containing S01_palatal01.wav, S01_palatal02.wav etc.)
 

> **💡 Output directory structure**
> The pipeline resolves paths dynamically so that all processed data, trained models, and generated plots are saved in a `outputs` directory created **next to** (at the same level as) the cloned repository.
 
---
 
## Pipeline
 
The pipeline is split into sequential scripts prefixed with `Run`. Run them in order.
 
| # | Script | Description | Input | Output |
|---|--------|-------------|-------|--------|
| 01 | `Run01_Dataset.py` | Extracts a single mono channel from stereo recordings, applies Global RMS Normalization, resamples to a uniform 22.05 kHz, and generates a baseline classes report. | Raw stereo `.wav` files (output from manual cropping) | Processed mono/normalized/resampled `.wav`; `classes_report.xlsx` |
| 02 | `Run02_FeatureExtraction.py` | Core feature engine. Extracts acoustic metrics: F0 tracking, time-domain, spectral/cepstral features and formants (LPC). | Resampled/normalized `.wav` files from Run01 | Unified `full_dataset.xlsx` |
| 03 | `Run03_DatasetSummary.py` | Initial audit of extracted features. Prints global statistics and verifies class balance and feature integrity. | `full_dataset.xlsx` | Console printout (sample counts, unique subjects, class distribution) |
| 04 | `Run04_TrainTestSplit.py` | Generates stratified N folds grouped by subject, preventing subject-level data leakage (reroll logic ensures both classes appear in every test set). | `full_dataset.xlsx` | Fold subdirectories with `train_features.csv` / `test_features.csv` |
| 04a | `Run04a_CheckNestedCV.py` | Leakage check. Verifies test exclusivity and that no subject ID overlaps between train and test in any fold. | Per-fold `train_features.csv` / `test_features.csv` | `NestedCV_Report_Model_XX.txt` |
| 05 | `Run05_TrainTest.py` | Model training, hyperparameter tuning, and Recursive Feature Elimination (RFE). Filters multicollinearity, scales features, runs RFE, and trains SVM & MLP via inner CV (`GridSearchCV`) on every fold. | Per-fold `train_features.csv` / `test_features.csv` | `best_model_XX.pkl`, `rfe_scores_fold_XX.csv`, per-fold result `.txt` |
| 06 | `Run06_AggregateResults.py` | Aggregates performance across folds: mean and std of Balanced Accuracy, ROC AUC, Precision, Recall, F1; global confusion matrices. | Per-fold `.txt` results and `.pkl` models | `GLOBAL_RESULTS_[...].txt` |
| 07 | `Run07_PlotSingleFoldResults.py` | Per-fold visualizations: confusion matrices, ROC curves, feature importance, dataset summary tables. | `.pkl` models and per-fold `.txt` results | `.png` plots in `analysis/out_fold_XX` |
| 08 | `Run08_PlotGlobalResults.py` | Cohort-level plots: mean ROC curve (with std shading) and the global aggregated confusion matrix. | `GLOBAL_RESULTS_[...].txt` and per-fold test probabilities | `.png` (e.g. `roc_mean_final.png`, `global_confusion_matrix.png`) in `global_results` |
| 09 | `Run09_PlotHyperparameters.py` | Hyperparameter stability analysis. Yelds best hyperparameters from each inner CV loop, plotting grids and frequency tables. | Per-fold result `.txt` files | `.png` grid-selection and parameter-frequency tables |
| 10 | `Run10_AcousticFeaturesAnalysis.py` | Report of stable acoustic biomarkers. Aggregates RFE scores across folds, ranks features/families, plots distributions, radar charts, and aligned KDEs. | Per-fold `rfe_scores_fold_XX.csv` | `.csv` rankings and `.png` plots (e.g. `feature_radar_top10.png`, `aligned_biomarker_summary.png`) |
 
---
 
## ▶️ Execution Order
 
Run the scripts sequentially from the terminal. The scripts use top-level
imports (`import config`), so run them from **inside the `src/` directory**:
 
```bash
cd src
python Run01_Dataset.py
python Run02_FeatureExtraction.py
python Run03_DatasetSummary.py
python Run04_TrainTestSplit.py
python Run04a_CheckNestedCV.py
python Run05_TrainTest.py
python Run06_AggregateResults.py
python Run07_PlotSingleFoldResults.py
python Run08_PlotGlobalResults.py
python Run09_PlotHyperparameters.py
python Run10_AcousticFeaturesAnalysis.py
```
 
---
 
## ✅ Smoke Test (no dataset required)

`Run00_SmokeTest.py` runs the whole pipeline (`Run01`→`Run06`) on a small
**synthetic** cohort it generates on the fly, in a temporary workspace that is
deleted afterwards. It needs no access to the private clinical dataset.

```bash
cd src
python Run00_SmokeTest.py          # PASS/FAIL, exit code 0/1
python Run00_SmokeTest.py --keep   # keep the temporary workspace for inspection
```

Use it to:
- **confirm the environment is set up correctly** and the pipeline executes end
  to end (reviewers, fresh checkouts, CI);
- **guard against regressions**: the test is deterministic (fixed seeds, tiny
  model grids), so it prints a `REGRESSION FINGERPRINT` (balanced accuracy / ROC
  AUC per model). Run it before and after a code change and diff the fingerprint
  to confirm the change did not alter the numerical result of the pipeline.

> The synthetic classes are deliberately separable, so the fingerprint metrics
> are near-perfect by construction — the test checks **execution and stability**,
> not scientific performance.

---
 
## 🎧 Extract Features Only

If you just want the acoustic features out of your own signals — without the
splitting, training, or labels — use `extract_features_only.py`. It reuses the
**same feature engine as `Run02`**, so the columns match `full_dataset.xlsx`
(≈690 features per signal). The three `Run01` preprocessing steps are available
as *optional* flags.

```bash
cd src

# Raw features, no preprocessing (one row per .wav, searched recursively)
python extract_features_only.py /path/to/wavs -o features.csv

# Reproduce the paper's Run01 preprocessing, then extract
python extract_features_only.py /path/to/wavs -o features.xlsx --run01

# Cherry-pick individual steps
python extract_features_only.py /path/to/wavs -o features.csv \
    --single-channel --rms-norm min --target-sr 22050
```

**Options**

| Flag | Effect |
|------|--------|
| `-o, --output` | output file; `.csv` or `.xlsx` (default `features.csv`) |
| `--single-channel` | *Run01 step 1* — take the left channel (not a stereo mean) |
| `--rms-norm {min,max,mean,median}` | *Run01 step 2* — **global** RMS normalization to that batch target |
| `--target-sr HZ` | *Run01 step 3* — resample (paper uses `22050`) |
| `--run01` | apply all three steps with the paper's settings (from `config.py`) |
| `--no-quantize` | process in lossless float instead of emulating Run01's 16-bit WAV round-trip |
| `--no-f0` | skip F0 tracking — leaves only `hnr` / `voicing_ratio` empty, runs faster |

You can also call it on **in-memory arrays**:

```python
from extract_features_only import extract_features_from_array
row = extract_features_from_array(y, sr, name="clip_001",
                                  channel=0, rms_target=0.05, target_sr=22050)
```

---
 
## Custom Modules
 
The repository relies on custom modules.
 
### `config.py` & `model_config.py`
Central registries for input/output paths, global pipeline settings, sampling rates, RFE estimators, ML hyperparameters, and Nested CV architectures.
 
### `processing/`
Audio processing scripts and feature extraction scripts:
 
- `extract_singl_chn.py` — single-channel extraction
- `resampling.py` — sample-rate conversion
- `normalization.py` — RMS amplitude normalization
- `f0.py` — pyIN fundamental-frequency tracking
- `formants.py` — LPC formant extraction
- `spectral_feat.py` — master spectral feature extraction
### `training/`
Machine learning backend:
 
- `splitting.py` — Stratified Group K-Fold
- `model_pipeline.py` — Scikit-Learn pipelines, sample weighting, information-gain, multicollinearity dropping, evaluation metrics
### `plotting/`
Plots (matplotlib / seaborn):
 
- `eval_plots.py` — tables from evaluation of the model
- `global_plots.py` — global ROCs
- `hyperparameter_plots.py` — hyperparameter grids
- `biomarker_plots.py`, `plot_radar.py`, `spectroplots.py`, `confidence_plots.py` — acoustic biomarker visualizations
### `utils/`
Helper scripts:
 
- `files_handle.py` — recursive file scanning/parsing
- `ful_df_from_dict.py` — linking dictionaries back to full pandas DataFrames
- `get_features.py` — formatting top features
---
 
## 📊 Outputs at a Glance
 
- **Dataset:** `full_dataset.xlsx`, `classes_report.xlsx`
- **Models:** `best_model_XX.pkl`
- **Feature selection:** `rfe_scores_fold_XX.csv`, feature rankings
- **Results:** per-fold `.txt`, `GLOBAL_RESULTS_[...].txt`
- **Plots:** confusion matrices, ROC curves, hyperparameter grids, biomarker radar charts