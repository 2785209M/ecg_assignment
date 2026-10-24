import matplotlib.pyplot as plt
import numpy as np

tsv = np.loadtxt("James_ecg_1.tsv", delimiter='\t')
ecg_signal = tsv[:,8].copy()
ecg_signal = ecg_signal - np.mean(ecg_signal) # remove DC shift from the signal
N = len(ecg_signal)

sample_rate = 250 #(Hz)
measurement_time = N // sample_rate

ft = np.fft.fft(ecg_signal)
nyquist_mag = np.abs(ft)[:N // 2]

frequency = np.linspace(0, sample_rate // 2 , N // 2 , endpoint = False) # Generate the discrete frequency bins for the fourier transform
t=np.linspace(0 , measurement_time , N) # Creates an x axis for the time domain with the correct number of samples in the measurement time

plt.figure()
plt.plot(t,ecg_signal)
plt.savefig("ecg_time_domain.png")
plt.figure()
plt.plot(frequency,nyquist_mag)
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.grid(True)
plt.savefig("ecg_frequency_domain.png")
plt.show()
plt.close()