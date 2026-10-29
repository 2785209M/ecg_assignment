import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

IMAGES = Path("images")
IMAGES.mkdir(parents = True, exist_ok = True)

file_name = "James_ecg_1.tsv"
tsv = np.loadtxt(file_name, delimiter='\t')
ecg_signal = tsv[:,8].copy()


sample_rate = 250
num_samples = len(ecg_signal)
measurement_time = num_samples / sample_rate
t = np.linspace(0, measurement_time, num_samples)

plt.figure()
plt.title("time_domain_ecg")
plt.plot(t,ecg_signal)
plt.savefig(IMAGES / f"time_domain_{file_name}.svg")
plt.show()