from fano_class import fano
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


data = np.loadtxt("/Users/mikkelodeon/optomechanics/Single fano cavity/Data/20250512/5um/fsr1.txt")#[2:-3]
PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Single fano cavity/Data/20250512/5um/fsr1_PI.txt")#[2:-3]
norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Single fano cavity/Data/20250512/normalization/fsr2.txt")#[2:-3]
norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Single fano cavity/Data/20250512/normalization/fsr2.txt")#[2:-3]

#PI0 = norm_PI[0,1]
#data[:,1] = [d/(pi/PI0) for d,pi in zip(data[:,1], PI[:,1])]
data[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(data[:,1], PI[:,1], norm[:,1], norm_PI[:,1])]

def fabry_perot(λ, r, t, l, φ):
    T = np.abs((t**2)/(1-r**2*np.exp(2j* ((2 * np.pi / λ) * l + φ))))**2
    return T

p0 = [np.sqrt(0.3), np.sqrt(0.7), 20e3, np.pi/2]
popt, pcov = curve_fit(fabry_perot, data[:,0], data[:,1], p0=p0, maxfev=1000000)
xs = np.linspace(data[:,0][0], data[:,0][-1], 10000)

l = round(np.abs(popt[2])*1e-3,2)
l_err = round(np.sqrt(np.diag(pcov))[2]*1e-3,2)

length = [l, l_err]

print(l, "+/-", l_err)

plt.figure(figsize=(10,7))
plt.scatter(data[:,0], data[:,1], marker="o", color="royalblue", label="data")
plt.plot(xs, fabry_perot(xs, *popt), color="firebrick", label="$l \\approx$%5.2f$\\pm$%5.2fμm" % tuple(length)) 
#plt.title("M3/M5 off-resonance transmission")  
plt.xlabel("wavelength [nm]", fontsize=28)
plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.ylabel("norm. trans. [arb. u.]", fontsize=28)
plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)
plt.subplots_adjust(bottom=0.3)
plt.show()