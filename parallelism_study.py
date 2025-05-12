from fano_class import fano
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


data = np.loadtxt("/Users/mikkelodeon/optomechanics/fabry perot/data/M3+M5/short scan/peak data.txt")#[2:-3]
PI = np.loadtxt("/Users/mikkelodeon/optomechanics/fabry perot/data/M3+M5/short scan/peak ref.txt")#[2:-3]
norm = np.loadtxt("/Users/mikkelodeon/optomechanics/fabry perot/data/M3+M5/short scan/p data.txt")#[2:-3]
norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/fabry perot/data/M3+M5/short scan/p ref.txt")#[2:-3]
bg = np.loadtxt("/Users/mikkelodeon/optomechanics/fabry perot/data/M3+M5/short scan/bg peak.txt")
#bg_val = np.mean(bg[:,1])

#data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250422/20um/fsr.txt")[0:-26]
#PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250422/20um/fsr_PI.txt")[0:-26]
#norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250422/normalization/fsr4.txt")[0:-26]#
#norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250422/normalization/fsr4.txt")[0:-26]

data[:,1] = [((d-bg_val)/(pi-bg_val))/((n-bg_val)/(pi_-bg_val)) for d,pi,n,pi_,bg_val in zip(data[:,1], PI[:,1], norm[:,1], norm_PI[:,1], bg[:,1])]

def fabry_perot(λ, r, t, l, φ):
    T = np.abs((t**2)/(1-r**2*np.exp(2j* ((2 * np.pi / λ) * l + φ))))**2
    return T

def fabry_perot2(λ, r, r_, t, t_, l, φ):
    T = np.abs((t*t_)/(1-r*r_*np.exp(2j* ((2 * np.pi / λ) * l + φ))))**2
    return T

p0 = [np.sqrt(0.3), np.sqrt(0.7), 240e3, np.pi/2]
popt, pcov = curve_fit(fabry_perot, data[:,0], data[:,1], p0=p0, maxfev=1000000)
xs = np.linspace(data[:,0][0], data[:,0][-1], 10000)

l = round(np.abs(popt[2])*1e-3,2)
l_err = round(np.sqrt(np.diag(pcov))[2]*1e-3,2)

length = [l, l_err]

t_max = round(np.max(fabry_perot2(data[:,0], 0.57, 0.575, 0.814, 0.814, l*1e3, popt[3])),3)*1e2
t_max_data = round(np.max(fabry_perot(data[:,0], *popt)),3)*1e2
print(t_max, t_max_data)

idx = np.argmax(data[:,1])
λres = list(data[:,0])[idx]

F = (4*0.57*0.575) / (1 - 0.57*0.575)**2

eps = np.sqrt((t_max*1e-2 - t_max_data*1e-2)) * ((λres*1e-9)/(F*np.pi*160e-6))*1e3
print("eps = ", eps, "in mrad")
print("eps = ", eps*1e-3*(180/np.pi), "in degrees")

print(l, "+/-", l_err)

plt.figure(figsize=(10,7))
plt.scatter(data[:,0], data[:,1], marker="o", color="royalblue", label="data")
plt.plot(xs, fabry_perot2(xs, 0.57, 0.575, 0.814, 0.814, l*1e3, popt[3]), color="firebrick", alpha=0.6, label="$T_{MAX}^{optimal} = $ %s%%" % str(t_max))
plt.plot(xs, fabry_perot(xs, *popt), color="royalblue", alpha=0.6, label="$T_{MAX}^{measured} = $ %s%%" % str(t_max_data)) 
#plt.title("M3/M5 off-resonance transmission")  
plt.fill_between(xs, fabry_perot(xs, *popt), fabry_perot2(xs, 0.57, 0.575, 0.814, 0.814, l*1e3, popt[3]), alpha=0.05, color="firebrick")
plt.xlabel("wavelength [nm]", fontsize=28)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.ylabel("norm. trans. [arb. u.]", fontsize=28)
plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)
plt.subplots_adjust(bottom=0.3)
plt.show()