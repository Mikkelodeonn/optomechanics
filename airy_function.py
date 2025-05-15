import numpy as np
import matplotlib.pyplot as plt

def airy_function(δ, r1, r2):
    T = (1-np.abs(r1)**2) * (1-np.abs(r2)**2) / ((1-np.abs(r1*r2))**2 + 4*np.abs(r1*r2) * (np.sin(δ*np.pi/2))**2)
    return T

def airy_function2(λ, r, l):
    F = 4*np.abs(r)**2 / ((1-np.abs(r)**2)**2)
    T = 1/(1+F*(np.sin((4*np.pi*l/λ)/2))**2)
    return T

xs = np.linspace(940,960,10000)*1e-9

plt.figure(figsize=(10,7))

#plt.plot(xs, airy_function2(xs, np.sqrt(0.5), np.sqrt(0.5)), color="royalblue", linestyle="-.", label="$|r|^2=50$%")
plt.plot(xs*1e9, airy_function2(xs, np.sqrt(0.9), 100e-6), color="royalblue", label="$|r|^2 = 90\% \:\:\:\:\:\: l = 100 \mu m$")
plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)
plt.subplots_adjust(bottom=0.3)
plt.xticks(fontsize=19)
plt.yticks(fontsize=21)
plt.xlabel("wavelength [nm]", fontsize=24)
plt.ylabel("$|E_{out}|^2/|E_{0,in}|^2$ [arb. u.]", fontsize=24) 
plt.show()
