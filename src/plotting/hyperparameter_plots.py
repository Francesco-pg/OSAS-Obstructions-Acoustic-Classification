# -*- coding: utf-8 -*-
"""
Module:      hyperparameter_plots.py
Purpose:     Plotting functions for hyperparameter analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def save_hyperparameter_grid_table(df, title, filename):
    """Renders a grid of hyperparameters per fold as a clean table."""
    sns.set_style("white")
    
    # Adjust height based on number of parameters (rows)
    h = len(df) * 0.5 + 1.5
    fig, ax = plt.subplots(figsize=(10, h))
    ax.axis('off')
    
    tbl = ax.table(
        cellText=df.values, 
        rowLabels=df.index, 
        colLabels=df.columns, 
        loc='center', 
        cellLoc='center'
    )
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    
    header_color = '#34495e'
    row_header_color = '#f2f2f2'
    
    for k, cell in tbl.get_celld().items():
        row, col = k
        cell.set_edgecolor('#d5dbdb')
        cell.set_linewidth(0.6)
        
        if row == 0 and col != -1:  # Column headers
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(header_color)
            cell.set_height(0.1)
        elif col == -1 and row != 0:  # Row headers
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor(row_header_color)
            cell.set_width(0.2)
        elif row == 0 and col == -1:  # Top-left corner
            cell.set_facecolor(header_color)
        else:  # Data cells
            cell.set_height(0.08)

    plt.title(title, fontweight='bold', pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    sns.set_style("whitegrid")


def save_hyperparameter_summary_table(df, title, filename):
    """Renders a summary table of hyperparameter selection counts."""
    sns.set_style("white")
    
    h = len(df) * 0.45 + 1.5
    fig, ax = plt.subplots(figsize=(8, h))
    ax.axis('off')

    tbl = ax.table(
        cellText=df.values, 
        colLabels=df.columns, 
        loc='center', 
        cellLoc='center'
    )
    
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

    plt.title(title, fontweight='bold', pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    sns.set_style("whitegrid")