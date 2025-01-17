import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#### IMPORTANT!! -> FIT WITH FABRY-PEROT TRANSMISSION EQUATION INSTEAD OF A SERIES OF LORENTZIANS!

x0s = [959,962.5,966,969.5]

data = np.loadtxt("/Users/mikkelodeon/optomechanics/fabry perot/data/cavity.txt")[2:-12]
pi_cavity = np.loadtxt("/Users/mikkelodeon/optomechanics/fabry perot/data/pi_laser.txt")[2:-12]
laser_data = np.loadtxt("/Users/mikkelodeon/optomechanics/fabry perot/data/laser.txt")[2:-12]
pi_laser = np.loadtxt("/Users/mikkelodeon/optomechanics/fabry perot/data/pi_cavity.txt")[2:-12]
background = np.loadtxt("/Users/mikkelodeon/optomechanics/fabry perot/data/bg.txt")[2:-12]

pi0_cavity = pi_cavity[:,1][0]

pi_cavity[:,1] = [pi/pi0_cavity for pi in pi_cavity[:,1]]
pi_laser[:,1] = [pi/pi0_cavity for pi in pi_laser[:,1]]

data[:,1] = [((dat-bg)/(pi-bg))/((laser-bg)/(pi_-bg)) for dat,laser,pi,pi_,bg in zip(data[:,1],laser_data[:,1],pi_cavity[:,1],pi_laser[:,1],background[:,1])]

def lorentz(x,x0,Γ,A):
    return A/(1 + ((x-x0)/Γ)**2)

def lorentzians(x,Γ,B,A,*x0s):
    func = B
    for x0 in x0s:
        func += lorentz(x,x0,Γ,A)
    return func

p0 = [0.52, 0.35, 1] + x0s ## Γ, B, A + x0s

popt, pcov = curve_fit(lorentzians, data[:,0], data[:,1], maxfev=10000, p0=p0)

xs = np.linspace(data[:,0][0], data[:,0][-1], 10000)

plt.figure(figsize=(10,6))
plt.scatter(data[:,0], data[:,1], color="royalblue", label="data")
plt.plot(xs, lorentzians(xs, *popt), color="firebrick", label="fit")
plt.title("off-resonance fabry-perot signal (M8 + blank membrane)")
plt.xlabel("wavelength [nm]")
plt.ylabel("normalized transmission [arb. u.]")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.subplots_adjust(right=0.70)
plt.show()