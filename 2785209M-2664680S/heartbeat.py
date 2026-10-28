import fir_design as fd
import fir_filter as ff
import numpy as np
import matplotlib.pyplot as plt

tsv1 = np.loadtxt("James_ecg_1.tsv", delimiter='\t')
ecg_signal = tsv1[:,8].copy()

# Performing FIR Pre-Filtering to remove DC offset, baseline wander, and 50Hz mains interference

num_samples = len(ecg_signal) # Number of Samples in ECG signal
sample_rate = 250 # Sampling rate of ECG signal (Hz)
width = 1 # desired width of the transition window
stop_Hz = [(0, 3), (40, sample_rate // 2)] # The cutoff frequency for the filter
A = 30 # stopband attenuation

# get the array of taps and number of taps (taps == coefficients)
taps, num_taps = fd.getTaps(sample_rate, stop_Hz, A, width)

# use FIRfilter to filter x with the FIR filter
prefilter = ff.FIRfilter(taps)

# Simulate realtime filtering of the ECG
filtered_ecg_signal = np.zeros(num_samples)
for i in range(num_samples):
    filtered_ecg_signal[i] = prefilter.dofilter(ecg_signal[i])

tsv2 = np.loadtxt("Ewan_ecg_1.tsv", delimiter='\t')
ecg_template = tsv2[:,8].copy()

# use new prefilter to filter the ECG template signal. So delay line is empty
prefilter2 = ff.FIRfilter(taps)

# Simulate realtime filtering of the ECG template
filtered_ecg_template = np.zeros(len(ecg_template))
for i in range(len(ecg_template)):
    filtered_ecg_template[i] = prefilter2.dofilter(ecg_template[i])

#extract one PQRST complex from the filtered ECG template for matching
filtered_ecg_template = filtered_ecg_template[500:700]

# Reverse the filtered ECG template for cross-correlation
filtered_ecg_template = filtered_ecg_template[::-1]

matched_filter = ff.FIRfilter(filtered_ecg_template)

matched_filtered_ecg_signal = np.zeros(num_samples)

#Realtime matched filtering of the filtered ECG signal with the filtered ECG template
for i in range(num_samples):
    matched_filtered_ecg_signal[i] = matched_filter.dofilter(filtered_ecg_signal[i])

# Square the matched filtered signal to enhance the peaks for easier detection of R-peaks
matched_filtered_ecg_signal = matched_filtered_ecg_signal * matched_filtered_ecg_signal

plt.figure()
plt.title("ecg_matched_filtered_time_domain")
plt.plot(matched_filtered_ecg_signal)
plt.savefig("ecg_matched_filtered_time_domain.svg")


