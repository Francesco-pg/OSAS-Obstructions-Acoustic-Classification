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
| 04a | `Run04a_ChecckNestedCV.py` | Leakage check. Verifies test exclusivity and that no subject ID overlaps between train and test in any fold. | Per-fold `train_features.csv` / `test_features.csv` | `NestedCV_Report_Model_XX.txt` |
| 05 | `Run05_TrainTest.py` | Model training, hyperparameter tuning, and Recursive Feature Elimination (RFE). Filters multicollinearity, scales features, runs RFE, and trains SVM & MLP via inner CV (`GridSearchCV`) on every fold. | Per-fold `train_features.csv` / `test_features.csv` | `best_model_XX.pkl`, `rfe_scores_fold_XX.csv`, per-fold result `.txt` |
| 06 | `Run06_AggegateResults.py` | Aggregates performance across folds: mean and std of Balanced Accuracy, ROC AUC, Precision, Recall, F1; global confusion matrices. | Per-fold `.txt` results and `.pkl` models | `GLOBAL_RESULTS_[...].txt` |
| 07 | `Run07_PlotSingleFoldResults.py` | Per-fold visualizations: confusion matrices, ROC curves, feature importance, dataset summary tables. | `.pkl` models and per-fold `.txt` results | `.png` plots in `analysis/out_fold_XX` |
| 08 | `Run08_PlotGlobalResults.py` | Cohort-level plots: mean ROC curve (with std shading) and the global aggregated confusion matrix. | `GLOBAL_RESULTS_[...].txt` and per-fold test probabilities | `.png` (e.g. `roc_mean_final.png`, `global_confusion_matrix.png`) in `global_results` |
| 09 | `Run09_PlotHyperparameters.py` | Hyperparameter stability analysis. Yelds best hyperparameters from each inner CV loop, plotting grids and frequency tables. | Per-fold result `.txt` files | `.png` grid-selection and parameter-frequency tables |
| 10 | `Run10_AcoustiFeaturesAnalysis.py` | Report of stable acoustic biomarkers. Aggregates RFE scores across folds, ranks features/families, plots distributions, radar charts, and aligned KDEs. | Per-fold `rfe_scores_fold_XX.csv` | `.csv` rankings and `.png` plots (e.g. `feature_radar_top10.png`, `aligned_biomarker_summary.png`) |
 
---
 
## ▶️ Execution Order
 
Run the scripts sequentially from the terminal:
 
```bash
python Run01_Dataset.py
python Run02_FeatureExtraction.py
python Run03_DatasetSummary.py
python Run04_TrainTestSplit.py
python Run04a_ChecckNestedCV.py
python Run05_TrainTest.py
python Run06_AggegateResults.py
python Run07_PlotSingleFoldResults.py
python Run08_PlotGlobalResults.py
python Run09_PlotHyperparameters.py
python Run10_AcoustiFeaturesAnalysis.py
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
- `normalization.py`, `rms_norm.py` — RMS amplitude normalization
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
---
 
## 👥 Authors
 
This project was developed by:
 
- **Francesco Pietrogiacomi** — [@Francesco-pg](https://github.com/Francesco-pg)
- **Linda Fiorini** — [@LindaFiorini](https://github.com/LindaFiorini)
- **Emanuele Agrimi** — [@Emaagr](https://github.com/Emaagr)
 
For questions about the pipeline, please open an [issue](../../issues).
