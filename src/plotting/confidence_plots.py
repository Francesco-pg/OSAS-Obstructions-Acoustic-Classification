#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Analysis
Module:      plotting/confidence_plots.py
Purpose:     Plotting functions for model confidence analysis.
Author:      Francesco Pietrogiacomi
Contact:     
Created:     2026-03-02
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- SETTINGS ---
SNS_STYLE = "whitegrid"
# Green for Correct, Red for Incorrect
PALETTE = {True: "#2ca02c", False: "#d62728"}

def plot_confidence_landscape(df, model_name, output_path):
    """
    Generates a scatter plot visualizing the relationship between a model's
    geometric decision score and its statistical probability.
    """
    sns.set_style(SNS_STYLE)
    plt.figure(figsize=(10, 8))
    
    df_plot = df.copy().sort_values(by='Prediction', ascending=False)
    df_plot['Class_Name'] = df_plot['Actual'].map({0: 'Palato', 1: 'Epiglottide'})

    sns.scatterplot(
        data=df_plot,
        x="Decision_Score_Raw",
        y="Prob_Epiglottide",
        hue="Prediction",
        palette=PALETTE,
        style="Class_Name",
        markers={"Palato": "o", "Epiglottide": "X"},
        s=80, alpha=0.75, edgecolor="k", linewidth=0.5
    )
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label="Geometric Boundary")
    plt.axhline(y=0.5, color='grey', linestyle='--', linewidth=1.5, label="Probability Boundary")
    
    # Annotations for paradox zones
    plt.text(x=df_plot['Decision_Score_Raw'].min()*0.5, y=0.6, s="Paradox Zone\n(Geo says Palato, Prob says Epi)", 
             fontsize=8, color='darkred', ha='center', alpha=0.6)
    plt.text(x=df_plot['Decision_Score_Raw'].max()*0.5, y=0.4, s="Paradox Zone\n(Geo says Epi, Prob says Palato)", 
             fontsize=8, color='darkred', ha='center', alpha=0.6)

    plt.title(f"Confidence Landscape: {model_name}\n(Decision Score vs. Probability)", fontweight='bold', fontsize=14)
    plt.xlabel("Decision Score Confidence (Distance to Hyperplane)", fontweight='bold')
    plt.ylabel("Statistical Confidence (Platt Probability)", fontweight='bold')
    plt.ylim([-0.05, 1.05])
    
    plt.legend(title="Prediction Correct", loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_certainty_summary(df, value_col, threshold, title_suffix, output_path):
    """
    Generates a stacked bar plot of certainty vs. correctness, with a detailed
    statistical table of class probabilities below it.
    """
    sns.set_style(SNS_STYLE)
    fig, (ax_plot, ax_table) = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), 
                                            gridspec_kw={'height_ratios': [4, 1]})
    
    plot_df = df.copy()

    # 1. Determine Correctness and Certainty
    plot_df['Is_Correct'] = plot_df['Actual'] == plot_df['Predicted']
    plot_df['Outcome'] = plot_df['Is_Correct'].map({True: 'Correct', False: 'Wrong'})
    plot_df['Abs_Value'] = plot_df[value_col].abs()
    plot_df['Certainty'] = plot_df['Abs_Value'].apply(
        lambda x: 'Certain' if x > threshold else 'Uncertain'
    )

    # --- BAR PLOT LOGIC ---
    agg_df = plot_df.groupby(['Outcome', 'Certainty']).size().reset_index(name='Count')
    pivot_df = agg_df.pivot(index='Outcome', columns='Certainty', values='Count').fillna(0)
    
    pivot_df = pivot_df.reindex(['Correct', 'Wrong'])
    if 'Certain' not in pivot_df.columns: pivot_df['Certain'] = 0
    if 'Uncertain' not in pivot_df.columns: pivot_df['Uncertain'] = 0
    pivot_df = pivot_df[['Certain', 'Uncertain']]

    colors = ['#1f77b4', '#d9d9d9'] # Blue, Gray
    
    pivot_df.plot(kind='bar', stacked=True, color=colors, ax=ax_plot, 
                  edgecolor='black', width=0.6, rot=0)
    
    ax_plot.set_title(f"{title_suffix}\n(Threshold > {threshold})", fontweight='bold', fontsize=12)
    ax_plot.set_ylabel("Count of Predictions", fontweight='bold')
    ax_plot.set_xlabel(None)
    ax_plot.legend(title="Confidence Status", loc='upper right')
    
    for c in ax_plot.containers:
        ax_plot.bar_label(c, label_type='center', color='black', weight='bold', fontsize=10)

    # --- SMART TABLE LOGIC ---
    prob_col = 'Prob_Epiglottide'
    stats_data = []
    
    if prob_col in df.columns:
        scenarios = [
            ("Epiglottide Correct (TP)", (plot_df['Actual'] == 1) & (plot_df['Predicted'] == 1), False),
            ("Epiglottide Wrong (FN)", (plot_df['Actual'] == 1) & (plot_df['Predicted'] == 0), False),
            ("Palato Correct (TN)", (plot_df['Actual'] == 0) & (plot_df['Predicted'] == 0), True),
            ("Palato Wrong (FP)", (plot_df['Actual'] == 0) & (plot_df['Predicted'] == 1), True)
        ]
        
        for name, mask, invert_prob in scenarios:
            subset = plot_df.loc[mask, prob_col]
            if not subset.empty:
                vals = 1.0 - subset if invert_prob else subset
                mean_val, std_val = vals.mean(), vals.std(ddof=0)
                stats_str = f"{mean_val:.3f} ± {std_val:.3f}"
            else:
                stats_str = "N/A"
            stats_data.append([name, stats_str])
    else:
        stats_data = [["Error", "Prob_Epiglottide missing"]]

    # Render Table
    ax_table.axis('off')
    col_labels = ["Scenario", "Avg Prob (True Class) ± SD"]
    
    tbl = ax_table.table(cellText=stats_data, colLabels=col_labels, 
                         loc='center', cellLoc='center')
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8) 
    
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#404040')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()