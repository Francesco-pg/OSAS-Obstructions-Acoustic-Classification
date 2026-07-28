# -*- coding: utf-8 -*-

"""
Project:     OSAS Classification from Acoustic Analysis
Module:      plotting/biomarker_plots.py
Purpose:     Plotting functions for biomarker and feature importance analysis.
Author:      Francesco Pietrogiacomi
Created:     2026-03-02
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from math import pi

# --- SETTINGS ---
SNS_STYLE = "whitegrid"
COLORS = ["#66c2a5", "#fc8d62"] 
FAMILY_COLOR = "#4c72b0"

COLOR_PALATE = "#08A39E"
COLOR_EPIGLOTTIS = "#AF7509"
PALETTE_KDE = {"Palate": COLOR_PALATE, "Epiglottis": COLOR_EPIGLOTTIS}


def _mix_colors(c1, c2):
    """Helper to mix two hex colors evenly."""
    rgb1 = np.array(mcolors.to_rgb(c1))
    rgb2 = np.array(mcolors.to_rgb(c2))
    return mcolors.to_hex((rgb1 + rgb2) / 2)


def save_feature_ranking_table(df, title, filename):
    """Renders a clean, academic-style table image for individual feature rankings."""
    sns.set_style("white")
    h = len(df) * 0.45 + 1.5 
    fig, ax = plt.subplots(figsize=(9, h))
    ax.axis('off')
    
    display_df = df.copy()
    if 'Score Sum' in display_df.columns:
        display_df['Score Sum'] = display_df['Score Sum'].apply(
            lambda x: round(x, 3) if isinstance(x, (int, float)) else x
        )
    
    tbl = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc='center', cellLoc='center')
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    
    header_color = '#34495e' 
    row_colors = ['#f8f9f9', '#ffffff'] 
    
    for k, cell in tbl.get_celld().items():
        row, col = k
        cell.set_edgecolor('#d5dbdb') 
        cell.set_linewidth(0.6)
        
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(header_color)
            cell.set_height(0.12)
        else:
            cell.set_facecolor(row_colors[row % 2])
            cell.set_height(0.09)
            cell.set_text_props(color='#2c3e50')

    plt.title(title, fontweight='bold', pad=15, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    sns.set_style(SNS_STYLE)


def plot_feature_distribution(df, feature, output_path):
    """Generates a boxplot for a single feature's distribution across classes."""
    sns.set_style(SNS_STYLE)
    plt.figure(figsize=(6, 5))
    plot_df = df.copy()
    
    plot_df['Class'] = plot_df['y'].map({0: 'Palato', 1: 'Epiglottide'})
    
    sns.boxplot(x='Class', y=feature, data=plot_df, palette=COLORS, width=0.5, linewidth=1.2, showfliers=False)
    sns.stripplot(x='Class', y=feature, data=plot_df, color=".3", size=3, alpha=0.5, jitter=True)
    
    plt.title(feature, fontweight='bold', fontsize=11, pad=10)
    plt.ylabel("Norm. Value", fontweight='bold')
    plt.xlabel(None)
    plt.xticks(fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_family_ranking_table(df, title, filename):
    """Renders the family summary as a clean, high-quality academic table."""
    sns.set_style("white")
    h = len(df) * 0.5 + 1.5
    fig, ax = plt.subplots(figsize=(11, h))
    ax.axis('off')
    
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    
    header_color = '#34495e'
    row_colors = ['#f8f9f9', '#ffffff']
    
    for k, cell in tbl.get_celld().items():
        row, col = k
        cell.set_edgecolor('#d5dbdb')
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(header_color)
            cell.set_height(0.12)
        else:
            cell.set_facecolor(row_colors[row % 2])
            cell.set_height(0.1)
            
    plt.title(title, fontweight='bold', pad=15, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    sns.set_style(SNS_STYLE)


def plot_family_importance(df_family, target_model, output_path):
    """Generates a horizontal bar plot for feature family importance."""
    sns.set_style(SNS_STYLE)
    plt.figure(figsize=(10, 8))
    
    top_plot = df_family.head(20).sort_values(by='Family Score Sum') 
    plt.barh(top_plot['Feature Family'], top_plot['Family Score Sum'], color=FAMILY_COLOR)
    
    plt.title(f"Acoustic Property Importance: {target_model}", fontweight='bold', fontsize=14)
    plt.xlabel("Cumulative Family Score (Sum of RFE Weights)", fontweight='bold')
    plt.ylabel("Acoustic Family", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feats_radar(df_top, output_path, n_feats):
    """Generates a Radar Chart for the top N features based on importance scores."""
    sns.set_style("white")
    
    categories = [f.replace("_", "\n") for f in df_top['Feature'].tolist()]
    N = len(categories)
    
    values = df_top['Score Sum'].values.flatten().tolist()
    max_val = max(values) if max(values) > 0 else 1
    values = [(v / max_val) * 100 for v in values]
    
    values += values[:1]
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    plt.xticks(angles[:-1], categories, color='black', size=9)
    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="grey", size=8)
    plt.ylim(0, 110) 
    
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#1f77b4')
    ax.fill(angles, values, '#1f77b4', alpha=0.3)
    ax.scatter(angles, values, color='#1f77b4', s=50, zorder=10)
    
    plt.title(f"Acoustic Signature: Top {n_feats} Features\n(Normalized by Importance Score)", 
              size=15, fontweight='bold', y=1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    sns.set_style(SNS_STYLE)


def plot_aligned_biomarker_summary(df_top, df_full, output_path):
    """
    Generates a combined GridSpec plot mapping individual top features (Bar) 
    to their data distributions (KDE).
    """
    with plt.rc_context({"font.family": "Cambria", "font.size": 14}):
        n_features = len(df_top)
        bar_color = _mix_colors(COLOR_PALATE, COLOR_EPIGLOTTIS)
        
        # --- Prepare Data ---
        def _split_and_clean_feature(feat):
            """Splits 'mfcc_mean' into 'Mfcc' and 'Mean'."""
            parts = feat.rsplit('_', 1)
            root = parts[0].replace("medlow", "Med-Low").replace("medhi", "Med-Hi").replace("_", " ").title()
            stat = parts[1] if len(parts) > 1 else 'Raw'
            return pd.Series([root, stat])

        df_plot = df_top.copy()
        df_plot[['Root Name', 'Stat Name']] = df_plot['Feature'].apply(_split_and_clean_feature)
        dist_features = df_plot['Feature'].tolist()
        
        plot_data = df_full.copy()
        plot_data['class'] = plot_data['y'].map({0: 'Palate', 1: 'Epiglottis'})
        
        # Scale the features for normalized plotting
        scaler = StandardScaler()
        plot_data[dist_features] = scaler.fit_transform(plot_data[dist_features])
        
        # --- Plotting ---
        fig = plt.figure(figsize=(18, max(12, n_features * 1.2)))
        gs = gridspec.GridSpec(n_features, 3, width_ratios=[1, 0.05, 1.5], hspace=0.0)

        # --- PANEL A: Bar Plot ---
        ax_bar = fig.add_subplot(gs[:, 0])
        y_pos = np.arange(n_features)
        
        ax_bar.barh(y_pos, df_plot['Score Sum'], height=0.5, color=bar_color, alpha=0.8, edgecolor='black')
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(df_plot['Root Name'], fontweight='bold', fontsize=24)
        
        # Invert y-axis so the highest score is at the top
        ax_bar.set_ylim(n_features - 0.5, -0.5)
        ax_bar.set_title("A. Top Feature Importance", loc='left', pad=30, fontweight='bold', fontsize=24)
        ax_bar.set_xlabel("Summed Importance Score", fontsize=24, fontweight='bold')
        sns.despine(ax=ax_bar)

        # --- MAPPING LABELS & PANEL B: KDE Distributions ---
        for i, feature in enumerate(dist_features):
            # Middle Labels (Statistic)
            ax_label = fig.add_subplot(gs[i, 1])
            stat_name = df_plot['Stat Name'].iloc[i]
            ax_label.text(2.5, 0.3, f"({stat_name})", 
                          transform=ax_label.transAxes, 
                          ha='center', va='center', 
                          fontsize=24, color='gray', fontstyle='italic')
            ax_label.axis('off')
            
            # Right KDE Plots
            ax_kde = fig.add_subplot(gs[i, 2])
            sns.kdeplot(data=plot_data, x=feature, hue="class", fill=True, 
                        common_norm=False, palette=PALETTE_KDE, alpha=0.6, ax=ax_kde, legend=(i==0))
            
            ax_kde.axhline(0, color='black', linewidth=1.5, alpha=0.7)
            ax_kde.set_yticks([])
            ax_kde.set_ylabel("")
            ax_kde.set_xlabel("")
            ax_kde.set_xlim(-4, 4)
            sns.despine(ax=ax_kde, left=True, bottom=True, top=True, right=True)
            ax_kde.patch.set_alpha(0)
            
            if i == 0:
                ax_kde.set_title("B. Feature Distributions", loc='left', pad=30, fontweight='bold', fontsize=24)
                sns.move_legend(ax_kde, "upper right", bbox_to_anchor=(1, 1.6), ncol=2, frameon=False, fontsize=18, title_fontsize=18)

        fig.get_axes()[-1].set_xlabel("Z-Score", fontsize=24, fontweight='bold')
        plt.subplots_adjust(left=0.1, right=0.95, wspace=0.05)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()