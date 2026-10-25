import matplotlib.pyplot as plt
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
def getTaps(sampling_rate, pass_bands, transition_width):
    # 1. Calculate the number of taps
    # Formula: N = 3.3 * sampling_rate / transition_width
    num_taps = int(3.3 * (sampling_rate / transition_width))
    if num_taps % 2 == 0:
        num_taps += 1 # Force an odd number of taps to maintain a centered peak
    
    # 2. Define the ideal frequency Response
    num_bins = 2000
    H = np.zeros(num_bins)

    # 3. Apply the pass bands
    for low, high in pass_bands:
        low = int(low * (num_bins / sampling_rate))
        high = int(high * (num_bins / sampling_rate))

        for i in range(low, high):
            H[i] = 1
            if i > 0: # ensure we don't try to mirror DC (0Hz)
                H[num_bins - i] = 1

    # 2. Apply the IFFT to get the time-domain impulse response
    h = np.real(np.fft.ifft(H))

    # 3. Shift Zero-frequency to center
    buffer = h.copy()
    h[0:int(num_bins/2)] = buffer[int(num_bins/2):num_bins]
    h[int(num_bins/2):num_bins] = buffer[0:int(num_bins/2)]

    # 4. Truncate to N taps and apply a cutoff window
    center_index = num_bins // 2
    half_index = num_taps // 2
    h = h[center_index - half_index : center_index + half_index + 1]

    # Apply a window function to smooth the sharp edges
    window = np.hamming(num_taps)
    taps = h * window

    return taps, num_taps

def convolution(signal, taps):
    # 1. Setup the output
    length = len(signal) + len(taps) - 1

    # 2. Find the frequency domain response of each function (padding with zeros so they can be multiplied together)
    signal_fft = np.fft.fft(signal, n = length)
    taps_fft = np.fft.fft(taps, n=length)

    # 3. Multiply the frequency domain
    frequency_product = signal_fft * taps_fft

    # 4. Take the real values of the time domain representation of the convolved product
    output = np.real(np.fft.ifft(frequency_product))

    # 5. Truncate the output array to undo the padding
    output = output[:len(signal)]
            
    return output

# Create an x axis for the time domain with the correct number of samples in the measurement time
t = np.linspace(0 , measurement_time , num_samples)

# Generate the discrete frequency bins for the fourier transform
frequency_bins = np.linspace(0, nyq_rate , nyq_samples , endpoint = False)

# desired width of the transition window
width = 1

# The cutoff frequency for the filter
pass_Hz = [(0.8, 30)]

# get num_taps
taps, num_taps = getTaps(sample_rate, pass_Hz, width)

# use filtfilt to filter x with the FIR filter
filtered_ecg_signal = convolution(ecg_signal, taps)

# Generate frequency Spectrum
raw_nyq_fft = np.abs(np.fft.fft(ecg_signal))[:nyq_samples]
filtered_nyq_fft = np.abs(np.fft.fft(filtered_ecg_signal))[:nyq_samples]


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