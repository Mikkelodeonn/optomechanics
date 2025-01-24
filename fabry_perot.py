import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.loadtxt("/Users/mikkelodeon/optomechanics/fabry perot/data/M8+M9/fp fringes.txt")[1:-8]
pi_cavity = np.loadtxt("/Users/mikkelodeon/optomechanics/fabry perot/data/M8+M9/p fringes.txt")[1:-8]
laser_data = np.loadtxt("/Users/mikkelodeon/optomechanics/fabry perot/data/M8+M9/laser.txt")[1:-8]
pi_laser = np.loadtxt("/Users/mikkelodeon/optomechanics/fabry perot/data/M8+M9/p laser3.txt")[1:-8]
background = np.loadtxt("/Users/mikkelodeon/optomechanics/fabry perot/data/M8+M9/bg.txt")[1:-8]

pi0_cavity = pi_cavity[:,1][0]

pi_cavity[:,1] = [pi/pi0_cavity for pi in pi_cavity[:,1]]
pi_laser[:,1] = [pi/pi0_cavity for pi in pi_laser[:,1]]

data[:,1] = [((dat)/(pi))/((laser)/(pi_))-bg for dat,laser,pi,pi_,bg in zip(data[:,1],laser_data[:,1],pi_cavity[:,1],pi_laser[:,1],background[:,1])]

def fabry_perot(λ, r, l, φ):
    t = np.sqrt(1-r**2)
    T = np.abs((t**2)/(1-r**2*np.exp(2j* ((2 * np.pi / λ) * l + φ))))**2
    return T

p0 = [np.sqrt(0.26),145e3,np.pi/2]

popt, pcov = curve_fit(fabry_perot, data[:,0], data[:,1], p0=p0, maxfev=1000000)
print(popt)

xs = np.linspace(data[:,0][0], data[:,0][-1], 10000)

plt.figure(figsize=(10,6))
plt.scatter(data[:,0], data[:,1], color="royalblue", label="data")
plt.plot(xs, fabry_perot(xs, *popt), color="firebrick", label="fit")
plt.title("off-resonance fabry-perot signal (M8 + M9)")
plt.xlabel("wavelength [nm]")
plt.ylabel("normalized transmission [arb. u.]")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.subplots_adjust(right=0.70)
plt.show()