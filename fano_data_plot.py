from fano_class import fano
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/565um/565s.txt")[15:-11]
PI_data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/565um/565s_PI.txt")[15:-11]
PI_0 = PI_data[0,1]

data[:,1] = [dat/(PI/PI_0) for dat,PI in zip(data[:,1], PI_data[:,1])]

def fabry_perot(λ, r, t, l, φ):
    T = np.abs((t**2)/(1-r**2*np.exp(2j* ((2 * np.pi / λ) * l + φ))))**2
    return T

p0 = [np.sqrt(0.90),np.sqrt(0.05),565e3,np.pi/2]

popt, pcov = curve_fit(fabry_perot, data[:,0], data[:,1], p0=p0, maxfev=1000000)
print(popt)

xs = np.linspace(data[:,0][0], data[:,0][-1], 10000)

plt.figure(figsize=(10,6))
plt.scatter(data[:,0], data[:,1], color="royalblue", label="data")
plt.plot(xs, fabry_perot(xs, *popt), color="firebrick", label="fit")
plt.title("double fano transmission at ~565μm (M3 + M5)") 
plt.xlabel("wavelength [nm]")
plt.ylabel("transmission [V]")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.subplots_adjust(right=0.70)
plt.show()