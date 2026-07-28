import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import config
from pathlib import Path

# --- SETTINGS ---
TOP_N = 20  

def plot_feats_radar(df_top, output_path):
    """Generates the Radar Chart."""
    # 1. Prepare Data
    # Clean up feature names for display (replace underscores with spaces or newlines)
    categories = [f.replace("_", "\n") for f in df_top['Feature'].tolist()]
    N = len(categories)
    
    # Use 'Score Sum' for the values
    values = df_top['Score Sum'].values.flatten().tolist()
    # Normalize values relative to the maximum in the top set (scaled to 100)
    max_val = max(values)
    values = [(v / max_val) * 100 for v in values]
    
    # Close the radar loop
    values += values[:1]
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # 2. Plotting
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Draw axes
    plt.xticks(angles[:-1], categories, color='black', size=9)
    
    # Draw background circles (25%, 50%, 75%, 100%)
    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="grey", size=8)
    plt.ylim(0, 110) # Leave a little room at the top
    
    # Plot the area
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#1f77b4')
    ax.fill(angles, values, '#1f77b4', alpha=0.3)
    
    # Add dots at the points
    ax.scatter(angles, values, color='#1f77b4', s=50, zorder=10)
    
    plt.title(f"Acoustic Signature: Top {TOP_N} Features\n(Normalized by Importance Score)", 
              size=15, fontweight='bold', y=1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()