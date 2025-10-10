import numpy as np
import matplotlib.pyplot as plt

def airy_function(δ, r1, r2):
    T = (1-np.abs(r1)**2) * (1-np.abs(r2)**2) / ((1-np.abs(r1*r2))**2 + 4*np.abs(r1*r2) * (np.sin(δ*np.pi/2))**2)
    return T

def airy_function2(λ, r, t, L, l):

    F = 4*np.abs(r)**2 / (1 - np.abs(r)**2 - 2*np.abs(t)**2 - L)**2
    T = (1-L)/(1+F*(np.sin((4*np.pi*l/λ)/2))**2)
    return T

def airy_function3(λ, r, t, l, φ):
    t = np.sqrt(1 - r**2)
    T = np.abs((t**2)/(1-r**2*np.exp(2j* ((2 * np.pi*3e-1 / λ) * l + φ))))**2
    return T

λs = np.linspace(940,960,10000)*1e-9
#xs = np.linspace(941, 959, 10000)*3e-1 ## omskriv til frekvens!!!
δs = np.linspace(0,8,10000)

plt.figure(figsize=(10,7))

#plt.plot(xs, airy_function3(xs, np.sqrt(0.75), 200e3, 0), color="royalblue", linestyle="-.", label="200μm Fabry-Perot cavity")
#plt.plot(xs, airy_function3(xs, np.sqrt(0.75), 100e3, 0), color="firebrick", label="100μm Fabry-Perot cavity")

#plt.plot(δs, airy_function(δs, np.sqrt(0.5), np.sqrt(0.5)), color="royalblue", linestyle="-.", label="$|r|^2=50$%")
#plt.plot(δs, airy_function(δs, np.sqrt(0.9), np.sqrt(0.9)), color="firebrick", linestyle="-", label="$|r|^2=90$%")
#plt.plot(λs*1e9, airy_function2(λs, np.sqrt(0.9), np.sqrt(0.1), np.sqrt(0.0), 100e-6), color="royalblue", linestyle="-.", label="lossless trans.")
#plt.plot(λs*1e9, airy_function2(λs, np.sqrt(0.7), np.sqrt(0.1), np.sqrt(0.2), 100e-6), color="firebrick", linestyle="-", label="lossy trans. (L = 20%)")

plt.plot(λs*1e9, airy_function2(λs, np.sqrt(0.9), np.sqrt(0.1), np.sqrt(0.0), 100e-6), color="firebrick", linestyle="-", label="high finesse") 
plt.plot(λs*1e9, airy_function2(λs, np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.0), 100e-6), color="royalblue", linestyle="-.", label="low finesse")
plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)
plt.subplots_adjust(bottom=0.3)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.xlabel("wavelength [nm]", fontsize=28)
#plt.xlabel("δ [π]", fontsize=28) 
#plt.xlabel("frequency [GHz]", fontsize=28)
plt.ylabel("norm. trans.", fontsize=28)
#plt.ylabel("$|E_{out}|^2/|E_{in}|^2$", fontsize=24) 
plt.locator_params(axis='x', tight=True, nbins=7)
plt.show()
