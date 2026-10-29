import fir_design as fir_design
import fir_filter as fir_filter
import numpy as np
from pathlib import Path

tsv = np.loadtxt("James_ecg_1.tsv", delimiter='\t')
ecg_signal = tsv[:,8].copy()

IMAGES = Path("images")
IMAGES.mkdir(parents = True, exist_ok = True)

# Performing FIR Filtering

num_samples = len(ecg_signal) # Number of Samples in ECG signal
sample_rate = 250 # Sampling rate of ECG signal (Hz)
width = 1 # desired width of the transition window
stop_Hz = [(0, 3), (40, sample_rate // 2)] # The cutoff frequency for the filter
A = 30 # stopband attenuation

# get the array of taps and number of taps (taps == coefficients)
taps, num_taps = fir_design.getTaps(sample_rate, stop_Hz, A, width)

# use FIRfilter to filter x with the FIR filter
filter = fir_filter.FIRfilter(taps)

# Simulate realtime filtering of the ECG
filtered_ecg_signal = np.zeros(num_samples)
for i in range(num_samples):
    filtered_ecg_signal[i] = filter.dofilter(ecg_signal[i])

