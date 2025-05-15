import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('/Users/mikkelodeon/optomechanics/scope/scope_4.csv', delimiter=",", skip_header=4)

time = data[:,0]
signal = data[:,1]
piezo = data[:,3]

plt.figure(figsize=(10,7))
plt.plot(time, signal/np.max(signal), color="orange", label="transmission signal")
plt.plot(time, piezo/np.max(piezo), color="royalblue", label="freq. generator signal")
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.ylabel("norm. voltage [arb. u.]", fontsize=24)
plt.xlabel("time [s]", fontsize=24)
plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)
plt.subplots_adjust(bottom=0.3)
plt.show()


