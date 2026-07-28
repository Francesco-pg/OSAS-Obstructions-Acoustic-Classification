# -*- coding: utf-8 -*-
"""
Module:      global_plots.py
Purpose:     Plotting functions for aggregated and global model results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import auc

sns.set_style("whitegrid")
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"] 


def save_global_table_png(df, title, filename):
    h = len(df) * 0.6 + 1.5 
    fig, ax = plt.subplots(figsize=(10, h))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    header_color = '#E0E0E0'
    
    for k, cell in tbl.get_celld().items():
        row, col = k
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(weight='bold', fontsize=12)
            cell.set_facecolor(header_color)
            cell.set_height(0.12)
        else:
            cell.set_fontsize(11)
            cell.set_height(0.1)
            
    plt.title(title, fontweight='bold', pad=25, fontsize=15)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_by_fold(fold_roc_data, target_model, output_path):
    plt.figure(figsize=(8, 7))
    for i, (fpr, tpr, f_auc) in enumerate(fold_roc_data):
        plt.plot(fpr, tpr, color=COLORS[i % len(COLORS)], lw=2, label=f'Fold {i+1} (AUC = {f_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    plt.title(f'ROC by Fold: {target_model}', fontweight='bold')
    plt.legend(loc="lower right")
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_mean_roc(mean_fpr, tprs_interp, fold_roc_data, target_model, output_path, show_ghost_lines=False, show_sd_shade=True):
    plt.figure(figsize=(8, 7))
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc_val = auc(mean_fpr, mean_tpr)
    std_auc_val = np.std([x[2] for x in fold_roc_data])

    if show_ghost_lines:
        for tpr in tprs_interp:
            plt.plot(mean_fpr, tpr, lw=1, color='grey', alpha=0.15)

    if show_sd_shade:
        std_tpr = np.std(tprs_interp, axis=0)
        plt.fill_between(
            mean_fpr, 
            np.maximum(mean_tpr - std_tpr, 0), 
            np.minimum(mean_tpr + std_tpr, 1), 
            color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.'
        )

    plt.plot(mean_fpr, mean_tpr, color='blue', lw=3, label=rf'Mean ROC (AUC = {mean_auc_val:.3f} $\pm$ {std_auc_val:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', lw=2)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    plt.title(f'Mean ROC Analysis: {target_model}', fontweight='bold')
    plt.legend(loc="lower right")
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_global_confusion_matrix(summed_cm, target_names, target_model, output_path):
    cm_norm = summed_cm.astype('float') / (summed_cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(7, 6))
    
    sns.heatmap(
        cm_norm, 
        annot=summed_cm.astype(int), 
        fmt='d', 
        cmap='Blues', 
        cbar=False,
        xticklabels=target_names, 
        yticklabels=target_names,
        annot_kws={"size": 14, "weight": "bold"}
    )
    
    plt.title(f"Global Confusion Matrix: {target_model}", fontweight='bold')
    plt.ylabel('Actual Class', fontweight='bold')
    plt.xlabel('Predicted Class', fontweight='bold')
    plt.savefig(output_path, dpi=300)
    plt.close()