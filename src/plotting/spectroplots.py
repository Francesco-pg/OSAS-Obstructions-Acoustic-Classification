# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:05:51 2025

@author: Francesco
"""

# In[Imports]

import librosa
# import glob
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# from typing import List
import os
# import math
# from tqdm import tqdm
# import pickle
# from typing import Dict, Tuple

# In[Function to plot f0 overlaid on spectogram]

def spect_f0(signalpath, f0_array) -> None:
    """
    Plots the fundamental frequency (f0) over the spectrogram of the audio signal.

    Args:
       
    """
    
    # y, sr = librosa.load(signalpath)
    
    signal, sr = librosa.load(signalpath)
    print(sr, ".sr.........")
    sname      = os.path.basename(signalpath).split(".")[0]
    
    ### Generate the time axis for the f0
    times = librosa.times_like(f0_array, sr=sr)
    times = times/2
    print(times, "times------")
    
    ### Compute the spectrogram (amplitude to decibels)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    ax.set(title=f'pYIN Fundamental Frequency Estimation - Sample {sname}')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    
    ax.plot(times, f0_array, label='f0', color='cyan', linewidth=3)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.legend(loc='upper right')

    # Show the plot
    plt.show()