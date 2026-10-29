import fir_design as fir_design
import fir_filter as fir_filter
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMAGES = Path("images")
IMAGES.mkdir(parents = True, exist_ok = True)

tsv = np.loadtxt("James_ecg_1.tsv", delimiter='\t')
ecg_signal = tsv[:,8].copy()

# This data is not used for the adaptive filtering, but allows us to call getTaps to get the number of taps for the adaptive filter. 
# The adaptive filter will be initialized with all coefficients set to zero.
num_samples = len(ecg_signal) # Number of Samples in ECG signal
sample_rate = 250 # Sampling rate of ECG signal (Hz)
width = 1 # desired width of the transition window
nyquist_freq = sample_rate // 2
stop_Hz = [(0, 3), (45, nyquist_freq)] # The cutoff frequency for the filter
A = 30 # stopband attenuation

# get the number of taps (taps == coefficients). 
taps, num_taps = fir_design.getTaps(sample_rate, stop_Hz, A, width)

#Simulate realtime adaptive filtering of the ECG

# Initialize adaptive filter with num_taps coefficients all set to zero
adaptivefilter = fir_filter.FIRfilter(np.zeros(num_taps)) 

# Initialize an array to hold the adaptive filtered ECG signal
adaptive_filtered_ecg_signal = np.zeros(num_samples)

#Use 50Hz as the signal for adaptive filtering (to be filtered out)
fiftyHz = np.sin(2 * np.pi * 50 / sample_rate * np.arange(num_samples))

# Use the filtered ECG signal as the noise reference for adaptive filtering
for i in range(num_samples):
    adaptive_filtered_ecg_signal[i] = adaptivefilter.dofilteradaptive(fiftyHz[i], ecg_signal[i], learningRate=0.001) 

measurement_time = num_samples / sample_rate
t = np.linspace(0 , measurement_time , num_samples)  

plt.figure()
plt.title("ecg_adaptive_filtered_time_domain")
plt.plot(t, adaptive_filtered_ecg_signal)
plt.savefig(IMAGES / "ecg_adaptive_filtered_time_domain.svg")
plt.show()