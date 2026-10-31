from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fir_filter as ff

IMAGES = Path("images")
IMAGES.mkdir(parents=True, exist_ok=True)

tsv = np.loadtxt("James_ecg_1.tsv", delimiter="\t")
ecg_signal = tsv[:, 8].copy()


# Defining Functions

# sampling_rate, frequency ranges to block, desired stop-band attenuation, width of transition window
def getTaps(sampling_rate, stop_bands, A, transition_width):

    # Formula for num_taps of Kaiser window: (A - 8) / (2.285 * d_omega), where d_omega is the normalized transition width in radians
    num_taps = int((A - 8) / (2.285 * 2 * np.pi * (transition_width / sampling_rate)))

    if num_taps % 2 == 0:
        num_taps += 1 # Ensure the number of taps is odd so the filter order will be even
    
    # Define the impulse Response of the ideal frequency response from stop_bands
    m = (num_taps - 1) // 2
    h = np.zeros(num_taps)
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
    
    # Calculate beta
    if A > 50:
        beta = 0.1102 * (A - 8.7)
    elif 21 <= A <= 50:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
    else:
        beta = 0

    # Calculate the Kaiser Window function using a modified bessel function of the first kind
    # Kaiser window formula: I0 [beta * sqrt(1 - {(n - alpha) / alpha} ** 2)] / I0 * beta
    kaiser_window = np.i0(beta * np.sqrt(1 - (n / m) ** 2)) / np.i0(np.array([beta]))[0]

    # Window the stop-band filter with our kaiser window
    taps = h * kaiser_window

    return taps, num_taps


# ---------------------------------------------------------------------------
# FIR filtering
# ---------------------------------------------------------------------------

num_samples = len(ecg_signal)  # Number of samples in the ECG signal
sample_rate = 250              # Sampling rate of the ECG signal (Hz)
width = 1                      # Width of the transition window (Hz)
stop_Hz = [(40, sample_rate // 2)]  # Stop bands
A = 30                         # Stop-band attenuation (dB)

# Get the filter coefficients (taps) and the number of taps
taps, num_taps = getTaps(sample_rate, stop_Hz, A, width)

fir = ff.FIRfilter(taps)

# Simulate real-time filtering of the ECG
filtered_ecg_signal = np.zeros(num_samples)
for i in range(num_samples):
    filtered_ecg_signal[i] = fir.dofilter(ecg_signal[i])


# ---------------------------------------------------------------------------
# Adaptive filtering
# ---------------------------------------------------------------------------

# Start with all-zero coefficients
adaptivefilter = ff.FIRfilter(np.zeros(num_taps))

# 50 Hz reference signal used to adaptively remove the mains interference
fiftyHz = np.sin(2 * np.pi * 50 / sample_rate * np.arange(num_samples))

# Simulate real-time adaptive filtering of the ECG
adaptive_filtered_ecg_signal = np.zeros(num_samples)
for i in range(num_samples):
    adaptive_filtered_ecg_signal[i] = adaptivefilter.dofilteradaptive(
        fiftyHz[i], ecg_signal[i], learningRate=0.001)


# ---------------------------------------------------------------------------
# Data for plotting
# ---------------------------------------------------------------------------

nyq_samples = num_samples // 2  # Number of samples below the Nyquist frequency
nyq_rate = sample_rate / 2      # Nyquist frequency (Hz)
measurement_time = num_samples / sample_rate

# Frequency spectra (positive frequencies only)
raw_nyq_fft = np.abs(np.fft.fft(ecg_signal))[:nyq_samples]
filtered_nyq_fft = np.abs(np.fft.fft(filtered_ecg_signal))[:nyq_samples]

# Time axis covering the full measurement
t = np.linspace(0, measurement_time, num_samples)

# Discrete frequency bins for the Fourier transform
frequency_bins = np.linspace(0, nyq_rate, nyq_samples, endpoint=False)

# Delay introduced by the linear-phase FIR filter
group_delay = (num_taps - 1) // 2
delay_time = group_delay / sample_rate


# ---------------------------------------------------------------------------
# ECG template
# ---------------------------------------------------------------------------

tsv2 = np.loadtxt("Ewan_ecg_1.tsv", delimiter="\t")
ecg_template = tsv2[:, 8].copy()

num_samples = len(ecg_template)
sample_rate = 250
width = 1
stop_Hz = [(0, 3), (40, sample_rate // 2)]  # Stop bands: below 3 Hz and above 40 Hz
A = 30


# Get the filter coefficients (taps) and the number of taps
template_taps, num_template_taps = getTaps(sample_rate, stop_Hz, A, width)

# Create a new filter for the template to avoid state contamination from the previous filter's delay line
filter2 = ff.FIRfilter(taps)

# Simulate real-time filtering of the ECG template
filtered_ecg_template = np.zeros(len(ecg_template))
for i in range(len(ecg_template)):
    filtered_ecg_template[i] = filter2.dofilter(ecg_template[i])

filtered_ecg_template = filtered_ecg_template[num_taps:]  # Extract one PQRST complex for matching
filtered_ecg_template = filtered_ecg_template[::-1]       # Reverse for cross-correlation
t2 = np.linspace(0, measurement_time, len(ecg_template))
t3 = np.linspace(0, measurement_time, len(filtered_ecg_template))



# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# FIR impulse response
plt.figure()
plt.plot(taps)
plt.title("Impulse Response (%d taps)" % num_taps)
plt.grid(True)
plt.savefig(IMAGES / "filter_coefficients.svg")

# FIR frequency response
frequency_domain_filter = np.fft.fft(taps)

plt.figure()
plt.plot(frequency_domain_filter)
plt.title("Filter (%d taps)" % num_taps)
plt.grid(True)
plt.savefig(IMAGES / "filter.svg")

# Time-domain signals
plt.figure()
plt.title("ECG_Raw")
plt.plot(t, ecg_signal, label="Raw ECG")

plt.figure()
plt.title("Filtered signal")
plt.plot(t - delay_time, filtered_ecg_signal, label="Filtered signal")
plt.savefig(IMAGES / "ecg_overlay_time_domain.svg")

"""
# Time-domain signals
plt.figure()
plt.title("ecg_template_raw_time_domain")
plt.plot(t2, ecg_template)
plt.savefig(IMAGES / "ecg_template_raw_time_domain.svg")

plt.figure()
plt.title("ecg_filtered_time_domain")
plt.plot(t3, filtered_ecg_template)
plt.savefig(IMAGES / "ecg_filtered_template_time_domain.svg")

plt.figure()
plt.title("ecg_adaptive_filtered_time_domain")
plt.plot(t, adaptive_filtered_ecg_signal)
plt.savefig(IMAGES / "ecg_adaptive_filtered_time_domain.svg")

# Frequency-domain signals
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
"""

plt.show()
plt.close()