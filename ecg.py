import matplotlib.pyplot as plt
import numpy as np

tsv = np.loadtxt("James_ecg_1.tsv", delimiter='\t')
ft = np.fft.fft(tsv[:,8].copy())
ft_mag = np.abs(ft)
sample_rate = 250
frequency = np.linspace(0,sample_rate,len(tsv))
t=np.linspace(0,len(tsv)/sample_rate,len(tsv))
plt.figure()
plt.plot(t,tsv[:,8])
plt.figure()
plt.plot(frequency,ft_mag)
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()