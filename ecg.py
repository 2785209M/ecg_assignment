import matplotlib.pyplot as plt
import scipy.signal as scipy
import numpy as np
from pathlib import Path

IMAGES = Path("images")
IMAGES.mkdir(parents = True, exist_ok = True)

tsv = np.loadtxt("James_ecg_1.tsv", delimiter='\t')
ecg_signal = tsv[:,8].copy()

num_samples = len(ecg_signal)
sample_rate = 250 #(Hz)

nyq_samples = num_samples // 2
nyq_rate = sample_rate / 2

measurement_time = num_samples / sample_rate


# number of samples, sampling_rate, frequency ranges to accept
def getTaps(num_bins, sampling_rate, pass_bands, transition_width):
    # 1. Calculate the number of taps
    # Formula: N = 3.3 * sampling_rate / transition_width
    num_taps = int(3.3 * (sampling_rate / transition_width))
    if num_taps % 2 == 0:
        num_taps += 1 # Force an odd number of taps to maintain a centered peak
    
    # 2. Define the ideal frequency Response
    h_ideal = np.zeros(num_bins)

    # 3. Apply the pass bands
    for low, high in pass_bands:
        low = int(low * (num_bins / sampling_rate))
        high = int(high * (num_bins / sampling_rate))

        for i in range(low, high):
            h_ideal[i] = 1
            if i > 0: # ensure we don't try to mirror DC (0Hz)
                h_ideal[num_bins - i] = 1

    # 2. Apply the IFFT to get the time-domain impulse response
    h_time = np.fft.ifft(h_ideal)
    h_real = np.real(h_time)

    # 3. Shift Zero-frequency to center
    h_shifted = np.fft.fftshift(h_real)

    # 4. Truncate to N taps and apply a cutoff window
    center_index = num_bins // 2
    half_index = num_taps // 2
    h_truncated = h_shifted[center_index - half_index : center_index + half_index + 1]

    # Apply a window function to smooth the sharp edges
    window = scipy.windows.hamming(num_taps)
    taps = h_truncated * window

    return taps, num_taps

# Create an x axis for the time domain with the correct number of samples in the measurement time
t = np.linspace(0 , measurement_time , num_samples)

# Generate the discrete frequency bins for the fourier transform
frequency_bins = np.linspace(0, nyq_rate , nyq_samples , endpoint = False)

# desired width of the transition window
width = 3

# The cutoff frequency for the filter
pass_Hz = [(1, 30)]

# use the firwin function with the kaiser window to find the taps of the FIR filter
taps, num_taps = getTaps(num_samples, sample_rate, pass_Hz, width)

# use filtfilt to filter x with the FIR filter
filtered_ecg_signal = scipy.filtfilt(taps, 1, ecg_signal)

# Generate frequency Spectrum
raw_nyq_fft = np.abs(np.fft.fft(ecg_signal))[:nyq_samples]
filtered_nyq_fft = np.abs(np.fft.fft(filtered_ecg_signal))[:nyq_samples]


# Plot the FIR filter taps
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