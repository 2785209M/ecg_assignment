import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import FIRfilter

IMAGES = Path("images")
IMAGES.mkdir(parents = True, exist_ok = True)

tsv = np.loadtxt("James_ecg_1.tsv", delimiter='\t')
ecg_signal = tsv[:,8].copy()


# Defining Functions

# sampling_rate, frequency ranges to accept, width of transition window
def getTaps(sampling_rate, stop_bands, A, transition_width):

    # 1. Calculate the stopband attenuation and filter order for the kaiser window
    if A > 50:
        beta = 0.1102 * (A - 8.7)
    elif 21 <= A <= 50:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
    else:
        beta = 0

    filt_order = int((A - 8) / (2.285 * (transition_width / sampling_rate)))

    # 2. Define the impulse Response of the ideal frequency response from stop_bands
    m = (filt_order - 1) // 2
    h = np.zeros(filt_order)
    h[m] = 1 # Initialize h as an all-pass filter (unit impulse)
    n = np.arange(-m, m + 1)

    for f1, f2 in stop_bands:
        # Normalize Frequencies
        f1 = f1 / sampling_rate
        f2 = f2 / sampling_rate

        # IFFT Formula for bandpass filter: h(n) = 1 / pi * n * [sin(w2 * n) - sin(w1 * n)]
        # Simplified using sinc function: h(n) = [2 * f2 * sinc (2 * f2 * n)] - [2 * f1 * sinc (2 * f1 * n)]
        h_band = 2 * f2 * np.sinc(2 * f2 * n) - (2 * f1 * np.sinc(2 * f1 * n))

        # Principle of superposition - subtract bandpass filter from allpass filter to create a bandstop filter
        # WARNING: Ensure stop-band ranges don't overlap to avoid signal distortion
        h -= h_band

    # 3. Apply a window function to smooth the sharp edges
    window = np.kaiser(filt_order, beta)
    taps = h * window

    return taps, filt_order


# Performing FIR Filtering

num_samples = len(ecg_signal) # Number of Samples in ECG signal
sample_rate = 250 # Sampling rate of ECG signal (Hz)
width = 1 # desired width of the transition window
pass_Hz = [(0, 1), (40, sample_rate // 2)] # The cutoff frequency for the filter
A = 30 # stopband attenuation

# get the array of taps and number of taps (taps == coefficients)
taps, filt_order = getTaps(sample_rate, pass_Hz, A, width)

# use FIRfilter to filter x with the FIR filter
filter = FIRfilter.FIRfilter(taps)

# Simulate realtime filtering of the ECG
filtered_ecg_signal = np.zeros(num_samples)
for i in range(num_samples):
    filtered_ecg_signal[i] = filter.dofilter(ecg_signal[i])

#Simulate realtime adaptive filtering of the ECG
adaptivefilter = FIRfilter.FIRfilter(np.zeros(filt_order)) # Initialize adaptive filter with zero coefficients

adaptive_filtered_ecg_signal = np.zeros(num_samples)
#Use 50Hz as the signal for adaptive filtering (to be filtered out)
fiftyHz = np.sin(2 * np.pi * 50 / sample_rate * np.arange(num_samples))

# Use the filtered ECG signal as the noise reference for adaptive filtering
for i in range(num_samples):
    adaptive_filtered_ecg_signal[i] = adaptivefilter.dofilteradaptive(fiftyHz[i], ecg_signal[i], mu=0.001)   


# Plotting Data

nyq_samples = num_samples // 2 # Number of Samples below Nyquist frequency
nyq_rate = sample_rate / 2 # Nyquist Adjusted Sampling Rate
measurement_time = num_samples / sample_rate

# Generate frequency Spectrum
raw_nyq_fft = np.abs(np.fft.fft(ecg_signal))[:nyq_samples]
filtered_nyq_fft = np.abs(np.fft.fft(filtered_ecg_signal))[:nyq_samples]

# Create an x axis for the time domain with the correct number of samples in the measurement time
t = np.linspace(0 , measurement_time , num_samples)

# Generate the discrete frequency bins for the fourier transform
frequency_bins = np.linspace(0, nyq_rate , nyq_samples , endpoint = False)

# Plot the FIR Impulse Response
plt.figure()
plt.plot(taps)
plt.title("Impulse Response (%d taps)" % filt_order)
plt.grid(True)
plt.savefig(IMAGES / "filter_coefficients.svg")

# Plot the time domain signals
plt.figure()
plt.title("ecg_raw_time_domain")
plt.plot(t,ecg_signal)
plt.savefig(IMAGES / "ecg_raw_time_domain.svg")

plt.figure()
plt.title("ecg_filtered_time_domain")
plt.plot(t, filtered_ecg_signal)
plt.savefig(IMAGES / "ecg_filtered_time_domain.svg")

plt.figure()
plt.title("ecg_adaptive_filtered_time_domain")
plt.plot(t, adaptive_filtered_ecg_signal)
plt.savefig(IMAGES / "ecg_adaptive_filtered_time_domain.svg")

# Plot the frequency domain signals
plt.figure()
plt.title("ecg_raw_frequency_domain")
plt.plot(frequency_bins, raw_nyq_fft)
plt.grid(True)
plt.savefig(IMAGES / "ecg_raw_frequency_domain.svg")

plt.figure()
plt.title("ecg_filtered_frequency_domain")
plt.plot(frequency_bins, filtered_nyq_fft)
plt.grid(True)
plt.savefig(IMAGES / "ecg_filtered_frequency_domain.svg")

plt.show()
plt.close()