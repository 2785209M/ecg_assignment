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
    n = np.arange(-m, m + 1)
    H = np.ones(num_taps)

    # Apply the pass bands
    for low, high in stop_bands:
        low = int(low * (num_taps / sampling_rate))
        high = int(high * (num_taps / sampling_rate))

        for i in range(low, high):
            H[i] = 0
            if i > 0: # ensure we don't try to mirror DC (0Hz)
                H[num_taps - i] = 0

    # Apply the IFFT to get the time-domain impulse response
    h = np.real(np.fft.ifft(H))

    # Shift Zero-frequency to center
    buffer = h.copy()
    split = num_taps - m

    # Shift edges to the middle
    h[0:m] = buffer[split:num_taps]
    h[m:num_taps] = buffer[0:split]

    # 3. Calculate the Kaiser Window function using a modified bessel function of the first kind
    # Kaiser window formula: I0 [beta * sqrt(1 - {(n - alpha) / alpha} ** 2)] / I0 * beta
    kaiser_window = np.i0(beta * np.sqrt(1 - (n / m) ** 2)) / np.i0(np.array([beta]))[0]

    # 4. Window the stop-band filter with our kaiser window
    taps = h * kaiser_window

    return taps, num_taps