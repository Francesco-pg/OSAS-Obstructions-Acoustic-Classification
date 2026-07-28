#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Analysis
Module:      Run10_AcousticFeaturesAnalysis.py
Purpose:     Aggregates RFE scores to rank individual features and feature families,
             then generates summary tables and plots, including aligned distributions.
Author:      Francesco Pietrogiacomi
Created:     2026-03-02
"""

import pandas as pd
import numpy as np
from collections import defaultdict

import config as cfg
import model_config as m_cfg
from training import model_pipeline
from plotting.biomarker_plots import (
    save_feature_ranking_table, 
    plot_feature_distribution,
    save_family_ranking_table, 
    plot_family_importance, 
    plot_feats_radar,
    plot_aligned_biomarker_summary
)


def main():
    """
    Aggregates RFE scores across all folds, generates rankings for individual features 
    and feature families, and outputs distribution, radar, and aligned summary plots.
    """
    trained_models = ["SVM", "MLP"]
    
    for target_model in trained_models:
        if target_model not in m_cfg.MODELS: 
            continue
        
        analysis_out = cfg.AnalysisOutdir / target_model
        if not analysis_out.exists(): 
            continue

        print(f"\n--- BIOMARKER DISCOVERY: {target_model} ---")
        
        # --- Setup Directories ---
        biomarker_dir   = analysis_out / "biomarker_analysis"
        dist_dir        = biomarker_dir / "stable_biomarkers_dist"
        table_parts_dir = biomarker_dir / "feature_ranking_tables"
        family_dir      = biomarker_dir / "family_analysis"
        
        for d in [dist_dir, table_parts_dir, family_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # --- 1. Aggregate Scores from all Folds ---
        feature_tracker = defaultdict(lambda: {'score_sum': 0.0, 'count': 0})
        family_tracker  = defaultdict(lambda: {'total_score': 0.0, 'occurrences': 0, 'stats': defaultdict(int)})
        
        for fold_idx in range(1, cfg.N_FOLDS + 1):
            paths = cfg.get_fold_paths(fold_idx)
            score_path = paths['mod_output_dir'] / f"rfe_scores_fold_{fold_idx:02d}.csv"
            
            if score_path.exists():
                df_scores = pd.read_csv(score_path)
                for _, row in df_scores.iterrows():
                    feat  = row['Feature']
                    score = abs(row['Score'])
                    
                    # A. Track individual feature
                    feature_tracker[feat]['score_sum'] += score
                    feature_tracker[feat]['count'] += 1
                    
                    # B. Track feature family 
                    parts = feat.rsplit('_', 1)
                    root = parts[0]
                    stat = parts[1] if len(parts) > 1 else 'raw'
                    
                    family_tracker[root]['total_score'] += score
                    family_tracker[root]['occurrences'] += 1
                    family_tracker[root]['stats'][stat] += 1

        if not feature_tracker: 
            print(f"  -> No RFE scores found for {target_model}. Skipping.")
            continue

        # --- 2. Individual Feature Analysis ---
        print("  -> Analyzing individual feature stability...")
        feature_results = []
        for feat, data in feature_tracker.items():
            feature_results.append({
                'Feature': feat, 
                'Score Sum': data['score_sum'], 
                'Frequency': f"{data['count']}/{cfg.N_FOLDS}_folds"
            })

        df_features = pd.DataFrame(feature_results).sort_values(by='Score Sum', ascending=False).reset_index(drop=True)
        df_features.index += 1 
        df_features = df_features.reset_index().rename(columns={'index': 'Rank'})

        # Save Full CSV for reference
        df_features.to_csv(biomarker_dir / "all_features_ranked_by_sum.csv", index=False)

        # Generate chunked tables
        chunk_size = 20
        num_chunks = int(np.ceil(len(df_features) / chunk_size))
        for i in range(num_chunks):
            chunk = df_features[['Rank', 'Feature', 'Frequency', 'Score Sum']].iloc[i*chunk_size : (i+1)*chunk_size].copy()
            if len(chunk) < chunk_size:
                padding = pd.DataFrame([[""] * len(chunk.columns)] * (chunk_size - len(chunk)), columns=chunk.columns)
                chunk = pd.concat([chunk, padding], ignore_index=True)
            
            part_name = f"feature_ranking_part_{i+1:02d}.png"
            save_feature_ranking_table(chunk, f"Biomarker Rankings Part {i+1}", table_parts_dir / part_name)
        print(f"    - Generated {num_chunks} feature ranking tables.")

        # Aggregate full test dataframe for distribution plotting
        all_test_data = []
        for fold_idx in range(1, cfg.N_FOLDS + 1):
            paths = cfg.get_fold_paths(fold_idx)
            df_fold = pd.read_csv(paths['test_features'])
            df_fold = model_pipeline.prepare_split(df_fold) 
            all_test_data.append(df_fold)
        full_test_df = pd.concat(all_test_data)

        # Generate distribution plots for top features
        top_20_list = df_features.head(20)['Feature'].tolist()
        for feat in top_20_list:
            if feat in full_test_df.columns:
                plot_feature_distribution(full_test_df, feat, dist_dir / f"dist_{feat}.png")
        print(f"    - Generated distribution plots for top {len(top_20_list)} features.")

        # --- 3. Feature Family Analysis ---
        print("  -> Analyzing feature family importance...")
        family_rows = []
        for root, data in family_tracker.items():
            comp_list = [f"{k} (x{v})" for k, v in data['stats'].items()]
            comp_str = ", ".join(comp_list)
            family_rows.append({
                'Feature Family': root,
                'Total Occurrences': data['occurrences'],
                'Composition': comp_str,
                'Family Score Sum': round(data['total_score'], 3)
            })

        df_family = pd.DataFrame(family_rows).sort_values(by='Family Score Sum', ascending=False).reset_index(drop=True)
        df_family.index += 1
        df_family = df_family.reset_index().rename(columns={'index': 'Rank'})

        # Save standard family outputs
        df_family.to_csv(family_dir / "biomarker_families_full.csv", index=False)
        save_family_ranking_table(df_family.head(20), f"Top 20 Biomarker Families ({target_model})", family_dir / "biomarker_families_summary.png")
        plot_family_importance(df_family, target_model, family_dir / "biomarker_family_plot.png")
        print("    - Generated family ranking table and plot.")

        # --- 4. Advanced Visualizations (Radar & Aligned Summary) ---
        TOP_N = 10 
        
        # Pull the absolute top N individual features based on aggregated RFE scores
        df_top = df_features.sort_values(by='Score Sum', ascending=False).head(TOP_N)
        
        print(f"\n--- GENERATING ADVANCED PLOTS FOR TOP {TOP_N} ---")
        output_radar = biomarker_dir / f"feature_radar_top{TOP_N}.png"
        plot_feats_radar(df_top, output_radar, TOP_N)
        print(f"    - Radar Plot saved to: {output_radar.name}")
        
        # B. Aligned GridSpec Biomarker Plot (Bar + KDE) using individual features
        output_aligned = biomarker_dir / f"aligned_biomarker_summary_top{TOP_N}.png"
        plot_aligned_biomarker_summary(df_top, full_test_df, output_aligned)
        print(f"    - Aligned Biomarker Plot saved to: {output_aligned.name}")

        print(f"✅ Biomarker analysis for {target_model} complete.")


if __name__ == "__main__":
    main()