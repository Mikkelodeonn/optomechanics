from fano_class import fano
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250220/25um/25fsr.txt")[3:-1]
PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250220/25um/25fsr_PI.txt")[3:-1]

PI0 = PI[0,1]
data[:,1] = [d/(pi/PI0) for d,pi in zip(data[:,1], PI[:,1])]

def fabry_perot(λ, r, t, l, φ):
    T = np.abs((t**2)/(1-r**2*np.exp(2j* ((2 * np.pi / λ) * l + φ))))**2
    return T

p0 = [np.sqrt(0.85),np.sqrt(0.1),20e3,np.pi/2]
popt, pcov = curve_fit(fabry_perot, data[:,0], data[:,1], p0=p0, maxfev=1000000)
xs = np.linspace(data[:,0][0], data[:,0][-1], 10000)

l = round(np.abs(popt[2])*1e-3,3)
l_err = round(np.sqrt(np.diag(pcov))[2]*1e-3,3)

length = [l, l_err]

plt.figure(figsize=(10,6))
plt.scatter(data[:,0], data[:,1], marker="o", color="royalblue", label="data")
plt.plot(xs, fabry_perot(xs, *popt), color="firebrick", label="fit: cavity length $\\approx$ %5.3f +/- %5.3fμm" % tuple(length)) 
plt.title("M3/M5 off-resonance transmission")  
plt.xlabel("wavelength [nm]")
plt.ylabel("transmission [V]")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
plt.subplots_adjust(bottom=0.2)
plt.show()