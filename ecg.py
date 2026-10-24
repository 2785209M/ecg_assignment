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


# Order of filter, number of samples, sampling_rate, cutoff frequency
def getCoefficients(num_taps, num_bins, sampling_rate, cutoff_frequency):
    # Convert the cutoff frequency to a bin index
    cutoff_bin = int(cutoff_frequency * (num_samples / sampling_rate))

    # Define the ideal frequency Response
    h_ideal = np.zeros(num_bins)
    # Passband for positive frequencies
    h_ideal[0 : cutoff_bin] = 1
    # Passband for negative frequencies
    h_ideal[num_bins - cutoff_bin : num_bins] = 1

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
    coefficients = h_truncated * window

    return coefficients

# Remove the DC shift by taking the mean of the ecg signal (== the DC shift) and shifting it doen by that value
shifted_ecg_signal = ecg_signal - np.mean(ecg_signal) # remove DC shift from the signal

# Create an x axis for the time domain with the correct number of samples in the measurement time
t = np.linspace(0 , measurement_time , num_samples)

# Generate the discrete frequency bins for the fourier transform
frequency_bins = np.linspace(0, nyq_rate , nyq_samples , endpoint = False)

# desired width of the transition window
width = 5/nyq_rate

# The desired attenuation in the stop band, in dB
ripple_dB = 60

# Compute the order and Kaiser parameter for the FIR filter
N, beta = scipy.kaiserord(ripple_dB, width)

# The cutoff frequency for the filter
cutoff_Hz = 40

# use the firwin function with the kaiser window to find the coefficients of the FIR filter
taps = getCoefficients(31, num_samples, sample_rate, 30)

# use lfilter to filter x with the FIR filter
filtered_ecg_signal = scipy.filtfilt(taps, 1, shifted_ecg_signal)

# Generate frequency Spectrum
raw_nyq_fft = np.abs(np.fft.fft(ecg_signal))[:nyq_samples]
filtered_nyq_fft = np.abs(np.fft.fft(filtered_ecg_signal))[:nyq_samples]



# Plot the FIR filter coefficients
plt.figure()
plt.plot(taps)
plt.title("Impulse Response (%d taps)" % N)
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