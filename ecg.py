import matplotlib.pyplot as plt
import scipy.signal as scipy
import numpy as np

tsv = np.loadtxt("James_ecg_1.tsv", delimiter='\t')
ecg_signal = tsv[:,8].copy()
ecg_signal = ecg_signal - np.mean(ecg_signal) # remove DC shift from the signal

num_samples = len(ecg_signal)
sample_rate = 250 #(Hz)

nyq_samples = num_samples // 2
nyq_rate = sample_rate / 2

measurement_time = num_samples / sample_rate

# Creates an x axis for the time domain with the correct number of samples in the measurement time
t = np.linspace(0 , measurement_time , num_samples)

# Generate the discrete frequency bins for the fourier transform
frequency = np.linspace(0, nyq_rate , nyq_samples , endpoint = False)

# desired width of the transition window
width = 5/nyq_rate

# The desired attenuation in the stop band, in dB
ripple_dB = 60

# Compute the order and Kaiser parameter for the FIR filter
N, beta = scipy.kaiserord(ripple_dB, width)

# The cutoff frequency for the filter
cutoff_Hz = 10

# use the firwin function with the kaiser window to generate a lowpass FIR filter
taps = scipy.firwin(N, cutoff_Hz/nyq_rate, window=('kaiser', beta))

# use lfilter to filter x with the FIR filter
filtered_ecg_signal = scipy.lfilter(taps, 1, ecg_signal)

# Generate frequency Spectrum
raw_nyq_fft = np.abs(np.fft.fft(ecg_signal))[:num_samples // 2]
filtered_nyq_fft = np.abs(np.fft.fft(filtered_ecg_signal))[:num_samples // 2]

plt.figure()
plt.plot(t,ecg_signal)
plt.savefig("ecg_raw_time_domain.png")

plt.figure()
plt.plot(t, filtered_ecg_signal)
plt.savefig("ecg_filtered_time_domain.png")

plt.figure()
plt.plot(frequency, raw_nyq_fft)
plt.grid(True)
plt.savefig("ecg_raw_frequency_domain")

plt.figure()
plt.plot(frequency, filtered_nyq_fft)
plt.grid(True)
plt.savefig("ecg_filtered_frequency_domain")

plt.show()
plt.close()