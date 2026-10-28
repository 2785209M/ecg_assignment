import numpy as np

# sampling_rate, frequency ranges to block, desired stop-band attenuation, width of transition window
def getTaps(sampling_rate, stop_bands, A, transition_width):

    # 1. Calculate the stopband attenuation and filter order for the kaiser window
    if A > 50:
        beta = 0.1102 * (A - 8.7)
    elif 21 <= A <= 50:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
    else:
        beta = 0

    # Formula for num_taps of Kaiser window: (A - 8) / (2.285 * d_omega), where d_omega is the normalized transition width in radians
    num_taps = int((A - 8) / (2.285 * 2 * np.pi * (transition_width / sampling_rate)))

    if num_taps % 2 == 0:
        num_taps += 1 # Ensure the number of taps is odd so the filter order will be even
    
    # 2. Define the impulse Response of the ideal frequency response from stop_bands
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

    # 3. Calculate the Kaiser Window function using a modified bessel function of the first kind
    # Kaiser window formula: I0 [beta * sqrt(1 - {(n - alpha) / alpha} ** 2)] / I0 * beta
    kaiser_window = np.i0(beta * np.sqrt(1 - (n / m) ** 2)) / np.i0(np.array([beta]))[0]

    # 4. Window the stop-band filter with our kaiser window
    taps = h * kaiser_window

    return taps, num_taps