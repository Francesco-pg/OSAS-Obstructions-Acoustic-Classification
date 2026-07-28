# -*- coding: utf-8 -*-
"""
Module:      eval_plots.py
Purpose:     Plotting functions for generating tables, feature importance bars, and confusion matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
 
# --- SETTINGS ---
sns.set_style("white")
CM_CMAP = plt.cm.Blues


def save_table_as_png(df, title, filename):
    """Saves a pandas DataFrame as a formatted table image."""
    h = len(df) * 0.5 + 1.2 
    fig, ax = plt.subplots(figsize=(8, h))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    
    header_color = '#E0E0E0'
    for k, cell in tbl.get_celld().items():
        row, col = k
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(weight='bold', fontsize=10)
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor('white')
            cell.set_fontsize(9)
    
    plt.title(title, fontweight='bold', pad=15, fontsize=12)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def save_summary_table_as_png(df, title, filename):
    """Saves a compact cohort summary table as an image."""
    df_styled = df.copy()
    column_map = {'Class': 'CLASS', 'Subjects': 'SUB', 'Videos': 'SAMPLE'}
    df_styled = df_styled.rename(columns=column_map)
    df_styled['CLASS'] = df_styled['CLASS'].map({'palato': 'P', 'epiglottide': 'E'})

    h = len(df_styled) * 0.5 + 0.8
    fig, ax = plt.subplots(figsize=(4, h)) 
    ax.axis('off')
    tbl = ax.table(cellText=df_styled.values, colLabels=df_styled.columns, loc='center', cellLoc='center')
    
    for k, cell in tbl.get_celld().items():
        row, col = k
        cell.set_linewidth(1.2)
        if row == 0:
            cell.set_text_props(weight='bold', fontsize=12)
        cell.set_facecolor('white')

    plt.title(title, fontweight='bold', pad=10, fontsize=10)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance_score(feat_list, title, filename):
    """Plots a horizontal bar chart of the top 20 feature importance scores."""
    top_n = sorted(feat_list, key=lambda x: x[2], reverse=True)[:20]
    top_n.reverse()
    
    names = [x[0] for x in top_n]
    scores = [x[2] for x in top_n]
    
    plt.figure(figsize=(10, 8))
    norm = plt.Normalize(min(scores) if scores else 0, max(scores) if scores else 1)
    colors = plt.cm.viridis(norm(scores))
    plt.barh(names, scores, color=colors)
    plt.xlabel('Importance Score', fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """
    Plots a Confusion Matrix where colors are normalized by the actual class (rows) 
    to reflect recall per class, annotated with raw counts.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,           
        annot=cm,          
        fmt='d',           
        cmap=CM_CMAP, 
        cbar=False,          
        xticklabels=labels, 
        yticklabels=labels, 
        annot_kws={"size": 18, "weight": "bold"}
    )
    
    plt.title(title, fontweight='bold', pad=15)
    plt.ylabel('Actual Class', fontweight='bold')
    plt.xlabel('Predicted Class', fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    return cm