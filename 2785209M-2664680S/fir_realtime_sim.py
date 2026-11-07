import fir_design as fir_design
import fir_filter as fir_filter
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

tsv = np.loadtxt("James_ecg_1.tsv", delimiter='\t')
ecg_signal = tsv[:,8].copy()

IMAGES = Path("images")
IMAGES.mkdir(parents = True, exist_ok = True)

# Performing FIR Filtering

num_samples = len(ecg_signal) # Number of Samples in ECG signal
sample_rate = 250 # Sampling rate of ECG signal (Hz)
width = 1 # desired width of the transition window
stop_Hz = [(0, 1), (45, sample_rate // 2)] # The cutoff frequency for the filter
A = 30 # stopband attenuation

# get the array of taps and number of taps (taps == coefficients)
taps, num_taps = fir_design.getTaps(sample_rate, stop_Hz, A, width)

# use FIRfilter to filter x with the FIR filter
filter = fir_filter.FIRfilter(taps)

# Simulate realtime filtering of the ECG
filtered_ecg_signal = np.zeros(num_samples)
for i in range(num_samples):
    filtered_ecg_signal[i] = filter.dofilter(ecg_signal[i])

# Delay introduced by the linear-phase FIR filter
group_delay = (num_taps - 1) // 2
delay_time = group_delay / sample_rate
measurement_time = num_samples / sample_rate

# Remove the samples before the window is filled (garbage data)
t_original = np.linspace(0 , measurement_time , num_samples)
t = np.linspace(0 , measurement_time , num_samples)[num_taps:]
filtered_ecg_signal = filtered_ecg_signal[num_taps:]

plt.figure()
plt.title("ecg_filtered_time_domain")
plt.plot((t - delay_time)[1000:3000], filtered_ecg_signal[1000:3000])
plt.ylabel("Amplitude (V)")
plt.xlabel("Time (s)")
plt.show()
plt.savefig(IMAGES / "ecg_filtered_time_domain_zoomed.svg", bbox_inches='tight')

plt.figure()
plt.title("ecg_unfiltered_time_domain")
plt.plot(t_original[1000:3000], ecg_signal[1000:3000])
plt.ylabel("Amplitude (V)")
plt.xlabel("Time (s)")
plt.show()
plt.savefig(IMAGES / "ecg_unfiltered_time_domain_zoomed.svg", bbox_inches='tight')
