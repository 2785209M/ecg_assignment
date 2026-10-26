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
def getTaps(sampling_rate, stop_bands, transition_width):

    # 1. Calculate the number of taps (General formula for a Hamming Window)
    # Formula: N = 3.3 * sampling_rate / transition_width
    num_taps = int(3.3 * (sampling_rate / transition_width))
    if num_taps % 2 == 0:
        num_taps += 1 # Ensure num_taps is odd so there is a central value

    # 2. Define the impulse Response of the ideal frequency response from the stop_bands
    m = (num_taps - 1) // 2
    h = np.zeros(num_taps)
    h[m] = 1 # Initialize h as an all-pass filter (unit impulse)
    n = np.arange(-m, m + 1)

    for f_low, f_high in stop_bands:
        w1 = (f_low / sampling_rate) * 2 * np.pi
        w2 = (f_high / sampling_rate) * 2 * np.pi

        h_band = np.zeros(num_taps)

        # Use L' Hospital's rule to avoid dividing by zero at index M
        h_band[m] = (w2 - w1) / np.pi

        # Iterate from 0 to M-1, then from M + 1 to num_taps - 1, and apply the IFFT to each index
        # IFFT Formula for pass-band filter: h(n) = 1/pi*n * [sin(w2 * n) - sin(w1 * n)]
        for i in range(m):
            h_band[i] = (1 / (n[i] * np.pi)) * (np.sin(w2 * n[i]) - np.sin(w1 * n[i]))
        for i in range(m + 1, num_taps):
            h_band[i] = (1 / (n[i] * np.pi)) * (np.sin(w2 * n[i]) - np.sin(w1 * n[i]))

        # Principle of superposition - ensure pass_band ranges do not overlap to avoid signal distortion
        h -= h_band

    # 3. Apply a window function to smooth the sharp edges
    window = np.hamming(num_taps)
    taps = h * window

    return taps, num_taps


# Performing FIR Filtering

num_samples = len(ecg_signal) # Number of Samples in ECG signal
sample_rate = 250 # Sampling rate of ECG signal (Hz)
width = 1 # desired width of the transition window
pass_Hz = [(0, 1), (40, sample_rate // 2)] # The cutoff frequency for the filter

# get the array of taps and number of taps (taps == coefficients)
taps, num_taps = getTaps(sample_rate, pass_Hz, width)

# use FIRfilter to filter x with the FIR filter
filter = FIRfilter.FIRfilter(taps)

# Simulate realtime filtering of the ECG
filtered_ecg_signal = np.zeros(num_samples)
for i in range(num_samples):
    filtered_ecg_signal[i] = filter.dofilter(ecg_signal[i])



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
plt.title("Impulse Response (%d taps)" % num_taps)
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