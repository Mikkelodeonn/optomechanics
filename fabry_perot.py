import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

zoom = True

data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/Data/94um/off_resonance_fp_high_resolution.txt")[1:-1]
pi_cavity = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/Data/94um/off_resonance_fp_high_resolution_PI.txt")[1:-1]
laser_data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/Data/94um/laser_power_fp_high_resolution.txt")[1:-1]
pi_laser = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/Data/94um/laser_power_fp_high_resolution_PI.txt")[1:-1]
#background = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/Data/94um/bg_high_resolution.txt")[1:-1]

pi0_cavity = pi_cavity[:,1][0]

pi_cavity[:,1] = [pi/pi0_cavity for pi in pi_cavity[:,1]]
pi_laser[:,1] = [pi/pi0_cavity for pi in pi_laser[:,1]]

data[:,1] = [((dat)/(pi))/((laser)/(pi_)) for dat,laser,pi,pi_ in zip(data[:,1],laser_data[:,1],pi_cavity[:,1],pi_laser[:,1])]

def fabry_perot(λ, r, t, l, φ):
    #t = np.sqrt(1-r**2)
    T = np.abs((t**2)/(1-r**2*np.exp(2j* ((2 * np.pi / λ) * l + φ))))**2
    return T

p0 = [np.sqrt(0.26),np.sqrt(0.29),94e3,np.pi/2]
xs = np.linspace(data[:,0][0], data[:,0][-1], 10000)


popt, pcov = curve_fit(fabry_perot, data[:,0], data[:,1], p0=p0, maxfev=1000000)
errs = np.sqrt(np.diag(pcov))

fig, ax = plt.subplots(figsize=(10,7))
if zoom == True:
    data_zoom = data[160:195]
    axins = ax.inset_axes([0.3, 0.3, 0.35, 0.35])
    axins.scatter(data_zoom[:,0], data_zoom[:,1], color="royalblue")
    axins.plot(data_zoom[:,0], fabry_perot(data_zoom[:,0], *popt), color="firebrick")
    #axins.set_xlim(951.65, 951.95)
    #axins.set_xlim(950, 953.5)
    #axins.set_ylim(0.02, 0.5)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    mark_inset(ax, axins, loc1=2, loc2=4, edgecolor="black", alpha=0.3)
ax.scatter(data[:,0], data[:,1], color="royalblue", label="data")
ax.plot(xs, fabry_perot(xs, *popt), color="firebrick", label="fit")
plt.title("off-resonance fabry-perot signal (M3 + M5)")
plt.xlabel("wavelength [nm]")
plt.ylabel("normalized transmission [arb. u.]")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.subplots_adjust(right=0.70)
#ax.grid()
plt.show()